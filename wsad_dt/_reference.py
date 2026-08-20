"""Reference implementation of WSAD-DT (ICML 2025), cleaned for packaging.

The functions below are logic-identical to the original ``WSAD_DT.py``
released with the paper (verified by ``tests/test_equivalence.py``). Changes
are non-behavioral only: imports consolidated at the top, module-level side
effects (``torch.use_deterministic_algorithms``, ``mp.set_start_method``)
moved into the functions that need them, and unused imports removed.
See IMPLEMENTATION_NOTES.md for documented reference semantics.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class NNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(NNetwork, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim, 100, bias=False)
        self.encoder_fc2 = nn.Linear(100, 50, bias=False)
        self.encoder_fc3 = nn.Linear(50, 128, bias=False)  # Latent space

    def forward(self, x):
        x = F.selu(self.encoder_fc1(x))
        x = F.selu(self.encoder_fc2(x))
        latent = self.encoder_fc3(x)
        return latent, None


def calculate_centers(model, data_loader, device):
    model.eval()
    latent_vectors_normal = []
    latent_vectors_abnormal = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            latent, _ = model(inputs)
            normal_vectors = latent[labels == 0]
            abnormal_vectors = latent[labels == 1]
            if len(normal_vectors) > 0:
                latent_vectors_normal.append(normal_vectors)
            if len(abnormal_vectors) > 0:
                latent_vectors_abnormal.append(abnormal_vectors)

    normal_center = torch.cat(latent_vectors_normal, dim=0).mean(dim=0) \
        if latent_vectors_normal else None
    abnormal_center = torch.cat(latent_vectors_abnormal, dim=0).mean(dim=0) \
        if latent_vectors_abnormal else None
    eps = 0.1
    c = normal_center
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    a = abnormal_center
    a[(abs(a) < eps) & (a < 0)] = -eps
    a[(abs(a) < eps) & (a > 0)] = eps
    return c, a


def gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian kernel between two sets of points."""
    dist = torch.cdist(x, y, p=2)
    return torch.exp(-dist ** 2 / (2 * sigma ** 2))


def tdist_kernel(x, y, nu=1.0):
    """Compute t-distribution kernel between two sets of points."""
    dist = torch.cdist(x, y, p=2) ** 2
    return (1 + dist / nu) ** (-(nu + 1) / 2)


def mm(latents_a, latents_b, sigma=1.0, nu=1.0):
    # Subsample up to 8 points per class (consumes torch RNG).
    indices = torch.randperm(latents_a.size(0))[:8]
    latents_a = latents_a[indices]

    indices = torch.randperm(latents_b.size(0))[:8]
    latents_b = latents_b[indices]

    k_aa = gaussian_kernel(latents_a, latents_a, 0.1)

    dist_matrix = torch.cdist(latents_a, latents_a, p=2)
    duplicate_mask = dist_matrix < 1e-6
    k_aa = torch.where(duplicate_mask,
                       torch.tensor(0.0, device=k_aa.device), k_aa)
    k_aa = torch.where(torch.eye(k_aa.size(0), device=k_aa.device).bool(),
                       torch.tensor(0.0, device=k_aa.device), k_aa)
    valid_count = (~duplicate_mask).float().sum()
    k_aa_mean = k_aa.sum() / valid_count.clamp(min=1.0)

    k_bb = gaussian_kernel(latents_b, latents_b, 1)
    dist_matrix_b = torch.cdist(latents_b, latents_b, p=2)
    duplicate_mask_b = dist_matrix_b < 1e-6
    k_bb = torch.where(duplicate_mask_b,
                       torch.tensor(0.0, device=k_bb.device), k_bb)
    k_bb = torch.where(torch.eye(k_bb.size(0), device=k_bb.device).bool(),
                       torch.tensor(0.0, device=k_bb.device), k_bb)
    valid_count_b = (~duplicate_mask_b).float().sum()
    k_bb_mean = k_bb.sum() / valid_count_b.clamp(min=1.0)

    if torch.isnan(k_bb_mean):
        k_bb_mean = torch.tensor(0.0, device=k_bb.device)

    return k_aa_mean + k_bb_mean


class TDistributionLoss(nn.Module):
    def __init__(self, normal_center, abnormal_center, alpha=100,
                 epsilon=1e-6, weight_close=1):
        super(TDistributionLoss, self).__init__()
        self.normal_center = nn.Parameter(normal_center)
        self.abnormal_center = nn.Parameter(abnormal_center)
        self.epsilon = epsilon
        self.alpha = 0.2  # NOTE: init arg `alpha` is overridden (reference).
        self.weight_close = weight_close

    def light_trail(self, dist_sq):
        self.alpha = 0.2
        return torch.exp(-dist_sq)

    def lightt_tail_n(self, dist_sq):
        self.alpha = 0.2
        return torch.exp(-dist_sq / .5)

    def heavy_trail(self, dist_sq):
        self.alpha = 0.2
        t_dist = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
        return t_dist

    def compute_similarity(self, dist_sq, labels, t):
        if t == 0:
            heavy_tail_similarity = self.lightt_tail_n(dist_sq)
        else:
            heavy_tail_similarity = self.light_trail(dist_sq)
        light_tail_similarity = self.heavy_trail(dist_sq)
        similarity = torch.where(labels != t, light_tail_similarity,
                                 heavy_tail_similarity)
        return similarity

    def forward(self, latent_vectors, labels, unique_latent_vectors,
                unique_labels):
        normal_dist = ((latent_vectors - self.normal_center) ** 2).sum(dim=1)
        abnormal_dist = ((latent_vectors - self.abnormal_center) ** 2).sum(dim=1)

        t_dist_normal = self.compute_similarity(normal_dist, labels, 0)
        t_dist_abnormal = self.compute_similarity(abnormal_dist, labels, 1)

        q_normal = t_dist_normal / (t_dist_normal + t_dist_abnormal + self.epsilon)
        q_abnormal = t_dist_abnormal / (t_dist_normal + t_dist_abnormal + self.epsilon)

        q_normal = torch.clamp(q_normal, min=self.epsilon, max=1.0 - self.epsilon)
        q_abnormal = torch.clamp(q_abnormal, min=self.epsilon, max=1.0 - self.epsilon)

        loss_normal = -torch.log(q_normal[labels == 0]).mean() \
            - torch.log(1 - q_abnormal[labels == 0]).mean()

        reg_loss = mm(latent_vectors[labels == 0], latent_vectors[labels == 1])

        loss_abnormal = -torch.log(q_abnormal[labels == 1]).mean() \
            - torch.log(1 - q_normal[labels == 1]).mean()
        if torch.isnan(loss_abnormal):
            loss_abnormal = torch.tensor(0.0)
        if torch.isnan(loss_normal):
            loss_normal = torch.tensor(0.0)
        total_loss = loss_normal + loss_abnormal + reg_loss
        return total_loss


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def evaluate_ensemble(models, centers, data_loader, device):
    all_scores = []
    labels = []
    alpha = 0.2

    for inputs, lbls in data_loader:
        inputs = inputs.to(device)
        lbls = lbls.to(device)
        batch_scores = []

        with torch.no_grad():
            for model, (normal_center, abnormal_center) in zip(models, centers):
                model.eval()
                latent, _ = model(inputs)

                t_dist_normal = ((latent - normal_center) ** 2).sum(dim=1)
                t_dist_abnormal = ((latent - abnormal_center) ** 2).sum(dim=1)

                p_normal = (1 + t_dist_normal / alpha) ** -((alpha + 1) / 2)
                p_abnormal = (1 + t_dist_abnormal / alpha) ** -((alpha + 1) / 2)

                q_normal = p_normal / (p_normal + p_abnormal + 1e-6)
                q_anomaly = 1 - q_normal

                batch_scores.append(q_anomaly.cpu().numpy())

        aggregated_scores = np.nanmean(batch_scores, axis=0)
        all_scores.extend(aggregated_scores)
        labels.extend(lbls.cpu().numpy())

    all_scores = np.array(all_scores)
    return all_scores


def f(subset_size_class_0, num_splits, X_train_class_0, y_train_class_0,
      X_train_class_1, y_train_class_1, y_train_semi_supervised,
      X_train_contaminated, batch_size, seed, input_dim, latent_dim,
      device, i):
    start_idx = i * subset_size_class_0
    end_idx = (i + 1) * subset_size_class_0 if i < num_splits - 1 \
        else len(X_train_class_0)

    X_train_subset_class_0 = X_train_class_0[start_idx:end_idx]
    y_train_subset_class_0 = y_train_class_0[start_idx:end_idx]

    X_train_subset_class_1 = X_train_class_1[:]
    y_train_subset_class_1 = y_train_class_1[:]

    X_train_subset = np.vstack((X_train_subset_class_0, X_train_subset_class_1))
    y_train_subset = np.concatenate((y_train_subset_class_0, y_train_subset_class_1))

    subset_indices = np.arange(len(y_train_subset))
    np.random.shuffle(subset_indices)
    X_train_subset = X_train_subset[subset_indices]
    y_train_subset = y_train_subset[subset_indices]

    X_train_tensor = torch.tensor(X_train_subset, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_subset, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    counter = Counter(y_train_semi_supervised)
    weight_map = {0: 2. / counter[0], 1: 1. / counter[1]}

    sampler = WeightedRandomSampler(
        weights=[weight_map[label.item()] for data, label in train_dataset],
        num_samples=len(X_train_contaminated), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler)

    set_seed(seed)

    model = NNetwork(input_dim, latent_dim).to(device)
    normal_center, abnormal_center = calculate_centers(model, train_loader,
                                                       device)
    criterion_supervised = TDistributionLoss(normal_center, abnormal_center)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3,
                                 weight_decay=1e-5)

    model.train()
    for epoch in range(100):
        eoch_loss = 0
        for (labeled_inputs, labels) in train_loader:
            labeled_inputs = labeled_inputs.to(device)
            labels = labels.to(device)

            latent_vectors_labeled, _ = model(labeled_inputs)
            unique_labeled_inputs, unique_indices = torch.unique(
                labeled_inputs, dim=0, return_inverse=True)
            unique_labels = labels[unique_indices]

            uni_latent_vectors_labeled, _ = model(labeled_inputs)
            optimizer.zero_grad()
            supervised_loss = criterion_supervised(
                latent_vectors_labeled, labels,
                uni_latent_vectors_labeled, unique_labels)
            total_loss = supervised_loss
            eoch_loss += total_loss.item()
            if torch.isnan(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model, normal_center, abnormal_center


def train_and_append_model(i, seed, subset_size_class_0, num_splits,
                           X_train_class_0, y_train_class_0, X_train_class_1,
                           y_train_class_1, y_train_semi_supervised,
                           X_train_contaminated, batch_size, input_dim,
                           latent_dim, device):
    set_seed(seed)
    model, normal_center, abnormal_center = f(
        subset_size_class_0, num_splits, X_train_class_0, y_train_class_0,
        X_train_class_1, y_train_class_1, y_train_semi_supervised,
        X_train_contaminated, batch_size, seed, input_dim, latent_dim,
        device, i)
    return model, (normal_center, abnormal_center)


def train(num_splits, X_train_contaminated, y_train_semi_supervised, s,
          parallel=True):
    """Train the WSAD-DT ensemble (paper entry point).

    ``parallel=True`` reproduces the paper script's multiprocessing pool;
    ``parallel=False`` runs the identical per-split training serially
    (numerically equivalent, since each split reseeds all RNGs).
    """
    class_0_indices = np.where(y_train_semi_supervised == 0)[0]
    class_1_indices = np.where(y_train_semi_supervised == 1)[0]

    X_train_class_0 = X_train_contaminated[class_0_indices]
    y_train_class_0 = y_train_semi_supervised[class_0_indices]
    X_train_class_1 = X_train_contaminated[class_1_indices]
    y_train_class_1 = y_train_semi_supervised[class_1_indices]
    subset_size_class_0 = len(X_train_class_0) // num_splits

    input_dim = X_train_contaminated.shape[1]
    latent_dim = 128
    device = 'cpu'
    batch_size = 64

    seeds = [s * (j + 1) for j in range(num_splits)]
    args = [(i, seed, subset_size_class_0, num_splits, X_train_class_0,
             y_train_class_0, X_train_class_1, y_train_class_1,
             y_train_semi_supervised, X_train_contaminated, batch_size,
             input_dim, latent_dim, device)
            for i, seed in enumerate(seeds)]

    if parallel:
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=num_splits) as pool:
            results = [pool.apply_async(train_and_append_model, a)
                       for a in args]
            ensemble_models_centers = [r.get() for r in results]
    else:
        ensemble_models_centers = [train_and_append_model(*a) for a in args]

    ensemble_models, ensemble_centers = zip(*ensemble_models_centers)
    return ensemble_models, ensemble_centers


def test(ensemble_models, ensemble_centers, test_loader, device):
    s = evaluate_ensemble(ensemble_models, ensemble_centers, test_loader,
                          device)
    return s
