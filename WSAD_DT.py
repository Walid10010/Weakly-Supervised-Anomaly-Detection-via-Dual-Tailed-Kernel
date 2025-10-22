import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import glob
import itertools
import torch.nn as nn

torch.use_deterministic_algorithms(True)

###
class NNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(NNetwork, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim, 100, bias=False)
        self.encoder_fc2 = nn.Linear(100, 50, bias=False)
        self.encoder_fc3 = nn.Linear(50, 128, bias=False)  # Latent space


    def forward(self, x):
        x = F.selu(self.encoder_fc1(x))
        x = F.selu(self.encoder_fc2(x))
        latent = (self.encoder_fc3(x))  # Latent space output
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

    normal_center = torch.cat(latent_vectors_normal, dim=0).mean(dim=0) if latent_vectors_normal else None
    abnormal_center = torch.cat(latent_vectors_abnormal, dim=0).mean(dim=0) if latent_vectors_abnormal else None
    eps = 0.1
    c = normal_center
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
   # c = c+ model.eps

    a = abnormal_center
    a[(abs(a) < eps) & (a < 0)] = -eps
    a[(abs(a) < eps) & (a > 0)] = eps

   # a = a- model.eps
    return c, a




def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute Gaussian kernel between two sets of points.
    """
    dist = torch.cdist(x, y, p=2)
    return torch.exp(-dist ** 2 / (2 * sigma ** 2))

def tdist_kernel(x, y, nu=1.0):
    """
    Compute t-distribution kernel between two sets of points.
    """
    dist = torch.cdist(x, y, p=2) ** 2
    return (1 + dist / nu) ** (-(nu + 1) / 2)



def mm(latents_a, latents_b, sigma=1.0, nu=1.0):
    # Compute Gaussian kernel for latents_a
    indices = torch.randperm(latents_a.size(0))[:8]
    latents_a = latents_a[indices]

    indices = torch.randperm(latents_b.size(0))[:8]
    latents_b = latents_b[indices]

    k_aa = gaussian_kernel(latents_a, latents_a, 0.1)

    # Identify off-diagonal duplicates in latents_a
    dist_matrix = torch.cdist(latents_a, latents_a, p=2)
    duplicate_mask = dist_matrix < 1e-6  # Mask for near-zero distances
    # duplicate_mask.fill_diagonal_(False)  # Ignore self-comparisons (diagonal)

    # Zero out duplicate contributions
    k_aa = torch.where(duplicate_mask, torch.tensor(0.0, device=k_aa.device), k_aa)
    k_aa = torch.where(torch.eye(k_aa.size(0), device=k_aa.device).bool(), torch.tensor(0.0, device=k_aa.device), k_aa)

    # Adjust mean to account for valid comparisons
    valid_count = (~duplicate_mask).float().sum()  # Count non-duplicates
    k_aa_mean = k_aa.sum() / valid_count.clamp(min=1.0)

    # Repeat for latents_b
    k_bb = gaussian_kernel(latents_b, latents_b, 1)
    dist_matrix_b = torch.cdist(latents_b, latents_b, p=2)
    duplicate_mask_b = dist_matrix_b < 1e-6
    # duplicate_mask_b.fill_diagonal_(False)
    k_bb = torch.where(duplicate_mask_b, torch.tensor(0.0, device=k_bb.device), k_bb)
    k_bb = torch.where(torch.eye(k_bb.size(0), device=k_bb.device).bool(), torch.tensor(0.0, device=k_bb.device), k_bb)

    # Adjust mean for latents_b
    valid_count_b = (~duplicate_mask_b).float().sum()
    k_bb_mean = k_bb.sum() / valid_count_b.clamp(min=1.0)

    # Handle NaNs
    if torch.isnan(k_bb_mean):
        k_bb_mean = torch.tensor(0.0, device=k_bb.device)

    return k_aa_mean + k_bb_mean





class TDistributionLoss(nn.Module):
    def __init__(self, normal_center, abnormal_center, alpha=100, epsilon=1e-6, weight_close=1):
        """
        Loss function using heavy-tailed and light-tailed distributions without additional parameters.

        Args:
        - normal_center (Tensor): Center for normal points.
        - abnormal_center (Tensor): Center for abnormal points.
        - epsilon (float): Small value to prevent instability.
        """
        super(TDistributionLoss, self).__init__()
        self.normal_center = nn.Parameter(normal_center)
        self.abnormal_center = nn.Parameter(abnormal_center)
        self.epsilon = epsilon
        self.alpha = 0.2
        self.epsilon = epsilon
        self.weight_close = weight_close


    def light_trail(self, dist_sq):
        """
        Heavy-tailed distribution: Decays slowly at large distances.
        """
        # self.alpha = 0.2
        # t_dist = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)

        self.alpha = 0.2
        t_dist = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
        # return t_dist

        return torch.exp(-dist_sq)

    def lightt_tail_n(self, dist_sq):
        """
        Heavy-tailed distribution: Decays slowly at large distances.
        """
        # self.alpha = 0.2
        # t_dist = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)

        self.alpha = 0.2
        t_dist = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
        # return t_dist

        return torch.exp(-dist_sq/.5)

    def heavy_trail(self, dist_sq):
        """
        Light-tailed distribution: Decays rapidly at large distances.
        """
        # self.alpha = 0.5
        # t_dist = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)

        self.alpha = 0.2
        t_dist = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)

        return  t_dist

    def compute_similarity(self, dist_sq, labels, t):
        """
        Use heavy-tailed for anomalies and light-tailed for normal points.

        Args:
        - dist_sq (Tensor): Squared distances from the center.
        - labels (Tensor): Labels indicating normal (0) or abnormal (1).

        Returns:
        - Tensor: Similarity scores.
        """

        if t ==0:
            heavy_tail_similarity  = self.lightt_tail_n(dist_sq)
        else:
            heavy_tail_similarity = self.light_trail(dist_sq)
        light_tail_similarity = self.heavy_trail(dist_sq)

        # Apply light tail for normal points (label 0), heavy tail for abnormal points (label 1)
        similarity = torch.where(labels != t, light_tail_similarity, heavy_tail_similarity)
        return similarity

    def forward(self, latent_vectors, labels, unique_latent_vectors, unique_labels):
        """
        Compute the loss.

        Args:
        - latent_vectors (Tensor): Latent space representations.
        - labels (Tensor): Labels indicating normal (0) or abnormal (1).

        Returns:
        - Tensor: Total loss value.
        """
        # Compute squared distances
        normal_dist = ((latent_vectors - self.normal_center) ** 2).sum(dim=1)
        abnormal_dist = ((latent_vectors - self.abnormal_center) ** 2).sum(dim=1)

        # Compute similarity using dynamic distributions
        t_dist_normal = self.compute_similarity(normal_dist, labels, 0)
        t_dist_abnormal = self.compute_similarity(abnormal_dist, labels, 1)

        # Calculate probabilities
        q_normal = t_dist_normal / (t_dist_normal + t_dist_abnormal + self.epsilon)
        q_abnormal = t_dist_abnormal / (t_dist_normal + t_dist_abnormal + self.epsilon)

        # Clamp probabilities for numerical stability
        q_normal = torch.clamp(q_normal, min=self.epsilon, max=1.0 - self.epsilon)
        q_abnormal = torch.clamp(q_abnormal, min=self.epsilon, max=1.0 - self.epsilon)

        # Calculate loss for normal samples
        loss_normal = -torch.log(q_normal[labels == 0]).mean() - torch.log(1 - q_abnormal[labels == 0]).mean()
        # Regularization term

        reg_loss = mm(latent_vectors[labels == 0], latent_vectors[labels == 1])

        # Calculate loss for abnormal samples
        loss_abnormal = -torch.log(q_abnormal[labels == 1]).mean() - torch.log(1 - q_normal[labels == 1]).mean()
        if torch.isnan(loss_abnormal):
            loss_abnormal = torch.tensor(0.0)
        if torch.isnan(loss_normal):
                loss_normal = torch.tensor(0.0)
        # Combine losses
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
# Evaluate the ensemble of models
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

                # Calculate t-distribution probabilities
                t_dist_normal = ((latent - normal_center) ** 2).sum(dim=1)
                t_dist_abnormal = ((latent - abnormal_center) ** 2).sum(dim=1)

                p_normal = (1 + t_dist_normal / alpha) ** -((alpha+1)/2)
                p_abnormal = (1 + t_dist_abnormal / alpha) ** -((alpha+1)/2)

                q_normal = p_normal / (p_normal + p_abnormal + 1e-6)
                q_anomaly = 1 - q_normal  # Use this as the anomaly score

                batch_scores.append(q_anomaly.cpu().numpy())

        # Aggregate the scores from each model in the ensemble (e.g., by averaging)
        aggregated_scores = np.nanmean(batch_scores, axis=0)
        all_scores.extend(aggregated_scores)
        labels.extend(lbls.cpu().numpy())

    all_scores = np.array(all_scores)
    return  all_scores
    # labels = np.array(labels)
	#
    # # Calculate ROC AUC and Average Precision (AP) scores
    # roc_auc = roc_auc_score(labels, all_scores)
    # ap_score = average_precision_score(labels, all_scores)
	#
    # return roc_auc, ap_score
	#



from torch.utils.data import WeightedRandomSampler


def f(subset_size_class_0, num_splits, X_train_class_0, y_train_class_0, X_train_class_1, y_train_class_1, y_train_semi_supervised,
      X_train_contaminated, batch_size, seed, input_dim, latent_dim, device,  i):
    # Define the start and end indices for the current subset of class 0 data
    start_idx = i * subset_size_class_0
    end_idx = (i + 1) * subset_size_class_0 if i < num_splits - 1 else len(X_train_class_0)

    start_idx1 = i * len(y_train_class_1) // num_splits
    end_idx1 = (i + 1) * subset_size_class_0 if i < num_splits - 1 else len(X_train_class_0)

    # Extract the subset for class 0
    X_train_subset_class_0 = X_train_class_0[start_idx:end_idx]
    y_train_subset_class_0 = y_train_class_0[start_idx:end_idx]

    #X_train_subset_class_0 = X_train_class_0[:]
   # y_train_subset_class_0 = y_train_class_0[:]

    # indices = np.random.choice(len(X_train_subset_class_0), size=min(100, len(X_train_subset_class_0)), replace=False)
    # X_train_subset_class_0 = X_train_subset_class_0[indices]
    # y_train_subset_class_0 = y_train_subset_class_0[indices]
    X_train_subset_class_1 = X_train_class_1[:]
    y_train_subset_class_1 = y_train_class_1[:]


    # Combine class 0 subset with the full class 1 data to ensure all anomalies are present
    X_train_subset = np.vstack((X_train_subset_class_0, X_train_subset_class_1))
    y_train_subset = np.concatenate((y_train_subset_class_0, y_train_subset_class_1))

    # Shuffle the combined subset to mix class 0 and class 1 samples
    subset_indices = np.arange(len(y_train_subset))
    np.random.shuffle(subset_indices)
    X_train_subset = X_train_subset[subset_indices]
    y_train_subset = y_train_subset[subset_indices]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_subset, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_subset, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    counter = Counter(y_train_semi_supervised
                      )

    weight_map = {0: 2. / counter[0], 1: 1. / counter[1]}
    # weight_map = {0: 0.5, 1: 0.5}

    sampler = WeightedRandomSampler(
        weights=[weight_map[label.item()] for data, label in train_dataset],
        num_samples=len(X_train_contaminated), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    set_seed(seed)


    # end_idx = start_idx + subset_size if i < 1 else len(remaining_indices)
    #
    # unlabeled_indices = remaining_indices[start_idx:end_idx]
    # if len(unlabeled_indices) == 0:
    #     continue

    # X_unlabeled = X[unlabeled_indices]
    # X_unlabeled_tensor = X_train_tensor[i * subset_size:(i + 1) * subset_size]

    # unlabeled_dataset = TensorDataset(X_unlabeled_tensor, torch.zeros(len(X_unlabeled_tensor)))
    # unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

    model = NNetwork(input_dim, latent_dim).to(device)
    normal_center, abnormal_center = calculate_centers(model, train_loader, device)
    criterion_supervised = TDistributionLoss(normal_center, abnormal_center)

    optimizer = torch.optim.Adam(list(model.parameters()) , lr=1e-3, weight_decay=1e-5)


    model.train()
    for epoch in range(100):
        eoch_loss = 0
        for (labeled_inputs, labels) in (train_loader):
            labeled_inputs = labeled_inputs.to(device)
            labels = labels.to(device)
            # print(labels[labels==1].sum(), 'count')
            # unlabeled_inputs = unlabeled_inputs.to(device)


            latent_vectors_labeled, _ = model(labeled_inputs)
            unique_labeled_inputs, unique_indices = torch.unique(labeled_inputs, dim=0, return_inverse=True)

            # Use the indices to filter corresponding labels
            unique_labels = labels[unique_indices]

            uni_latent_vectors_labeled, _ =  model(labeled_inputs)
            optimizer.zero_grad()
            supervised_loss = criterion_supervised(latent_vectors_labeled, labels, uni_latent_vectors_labeled, unique_labels)
            total_loss = supervised_loss
            eoch_loss += total_loss.item()
            if torch.isnan(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model, normal_center, abnormal_center



from sklearn.model_selection import train_test_split
from collections import Counter

import torch.multiprocessing as mp

# Ensure this code is at the start of your script to avoid conflicts
mp.set_start_method('spawn', force=True)
import torch.multiprocessing as mp

# Ensure this code is at the start of your script to avoid conflicts
mp.set_start_method('spawn', force=True)

def main_with_exhaustive_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_values = [0.2]
    reconstruction_loss_factors = [0]
    reconstruction_loss_2_factors = [0]
    recalculate_intervals = [25]

    for path in glob.glob('Classical/*.npz'):
     try:
        if 'jpg' in path or 'donor' in path:
            continue
        for alpha, rec_loss_factor, rec_loss_2_factor, recalculate_interval in itertools.product(
                alpha_values, reconstruction_loss_factors, reconstruction_loss_2_factors, recalculate_intervals):

            print(f"Testing configuration: alpha={alpha}, rec_loss_factor={rec_loss_factor}, "
                  f"rec_loss_2_factor={rec_loss_2_factor}, recalculate_interval={recalculate_interval}")
            data = np.load(path, allow_pickle=True)
            X, Y = data['X'], data['y']
            X = MinMaxScaler().fit_transform(X)

            input_dim = X.shape[1]
            latent_dim = 128
            if   X.shape[0]  > 20000:
                continue

            seedsF = [100, 1000]
            roc_list = []
            ap_list = []

            set_seed(1)

            for ss in [1, 5, 10]:
                auc_roc_scores = []
                avg_precision_scores = []

                # Initial train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=ss, stratify=Y)

                # Step 2: Identify normal and anomalous indices in the training data
                normal_indices_train = np.where(y_train == 0)[0]
                anomalous_indices_train = np.where(y_train == 1)[0]

                # Step 3: Select a small percentage of labeled anomalies
                label_percentage = 0.05
                num_labeled_anomalies = max(int(label_percentage * len(anomalous_indices_train)), 5)
                labeled_anomalous_indices = np.random.choice(anomalous_indices_train,
                                                             min(num_labeled_anomalies, len(anomalous_indices_train)),
                                                             replace=False)

                # Step 4: Identify remaining non-labeled anomalies
                non_labeled_anomalous_indices = np.setdiff1d(anomalous_indices_train, labeled_anomalous_indices)

                # Step 5: Contaminate the training data
                contamination_ratio = 0.
                num_non_labeled_anomalies_to_add = 0#int(num_labeled_anomalies)  # int(contamination_ratio * len(non_labeled_anomalous_indices))
                contaminated_anomalous_indices = np.random.choice(non_labeled_anomalous_indices,
                                                                  num_non_labeled_anomalies_to_add, replace=False)

                # Step 6: Combine normal and anomalous data into the contaminated dataset
                contaminated_indices_train = np.concatenate(
                    (normal_indices_train, labeled_anomalous_indices, contaminated_anomalous_indices))
                X_train_contaminated = X_train[contaminated_indices_train]

                # Step 7: Create initial semi-supervised labels
                y_train_semi_supervised = np.zeros_like(y_train[contaminated_indices_train])
                y_train_semi_supervised[
                len(normal_indices_train):len(normal_indices_train) + len(labeled_anomalous_indices)] = 1

                # Step 8: Mislabel a small percentage of normal samples as anomalies
                mislabel_percentage = 0.0  # 5% of normal samples misclassified as anomalies
                num_mislabeled_normals =0# num_non_labeled_anomalies_to_add  # int(mislabel_percentage * len(normal_indices_train))

                # Randomly select normal indices to mislabel as anomalies
                mislabeled_normal_indices = np.random.choice(normal_indices_train,
                                                             num_mislabeled_normals,
                                                             replace=False)

                # Update the labels for the mislabeled normal samples
                mislabeled_indices_in_contaminated = np.isin(contaminated_indices_train, mislabeled_normal_indices)
                y_train_semi_supervised[mislabeled_indices_in_contaminated] = 1
                indices = np.random.permutation(len(X_train_contaminated))
                X_train_contaminated = X_train_contaminated[indices]
                y_train_semi_supervised = y_train_semi_supervised[indices]

                X_train_tensor = torch.tensor(X_train_contaminated, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_semi_supervised, dtype=torch.long)

                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test, dtype=torch.long)

                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

                batch_size = 64
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



                # Define number of splits and create subsets
                num_splits = 1
                class_0_indices = np.where(y_train_semi_supervised == 0)[0]
                class_1_indices = np.where(y_train_semi_supervised == 1)[0]

                X_train_class_0 = X_train_contaminated[class_0_indices]
                y_train_class_0 = y_train_semi_supervised[class_0_indices]
                X_train_class_1 = X_train_contaminated[class_1_indices]
                y_train_class_1 = y_train_semi_supervised[class_1_indices]
                subset_size_class_0 = len(X_train_class_0) // num_splits

                # Iterate over the seed values in seedsF
                for s in seedsF:
                    print(f"Seed: {s}")
                    set_seed(s)

                    ensemble_models, ensemble_centers  = train(num_splits, X_train_contaminated, y_train_semi_supervised, s)
                    roc_auc, ap_score = test(ensemble_models, ensemble_centers, test_loader, device)


#                     # Use a process pool with 5 workers for parallel training
#                     with mp.Pool(processes=num_splits) as pool:
#                         results = []
# #                        for i, seed in enumerate([s * (j + 1) for j in range(num_splits)]):
#
#                         # Create separate seeds for each split in the ensemble using s as a base
#                         for i, seed in enumerate([s * (j + 1)  for j in range(num_splits)]):
#                             print(seed)
#                             result = pool.apply_async(
#                                 train_and_append_model,
#                                 (i, seed, subset_size_class_0, num_splits, X_train_class_0, y_train_class_0,
#                                  X_train_class_1, y_train_class_1, y_train_semi_supervised, X_train_contaminated,
#                                  batch_size, input_dim, latent_dim, device, alpha, recalculate_interval)
#                             )
#                             results.append(result)
#
#                         # Wait for all processes to finish and collect results
#                         ensemble_models_centers = [result.get() for result in results]
#
#                     # Unpack models and centers
#                     ensemble_models, ensemble_centers = zip(*ensemble_models_centers)
                    #roc_auc, ap_score = evaluate_ensemble(ensemble_models, ensemble_centers, test_loader, device, alpha)
                    roc_list.append(roc_auc)
                    ap_list.append(ap_score)


            avg_roc_auc = np.mean(roc_list)
            avg_ap_score = np.mean(ap_list)

            result_str = (
                f"Path: {path} | Ensemble ROC AUC: {avg_roc_auc:.4f}, Ensemble AP Score: {avg_ap_score:.4f}\n")
            print(result_str)

            with open("{}erspilt_1_abgabe.txt".format(num_splits), "a") as file:
                file.write(result_str)

     except Exception as e:
       print(e)
       pass



def train(num_splits, X_train_contaminated, y_train_semi_supervised, s):
    # Define number of splits and create subsets
    num_splits = num_splits
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

    with mp.Pool(processes=num_splits) as pool:
        results = []
        #                        for i, seed in enumerate([s * (j + 1) for j in range(num_splits)]):

        # Create separate seeds for each split in the ensemble using s as a base
        for i, seed in enumerate([s * (j + 1) for j in range(num_splits)]):
            print(seed)
            result = pool.apply_async(
                train_and_append_model,
                (i, seed, subset_size_class_0, num_splits, X_train_class_0, y_train_class_0,
                 X_train_class_1, y_train_class_1, y_train_semi_supervised, X_train_contaminated,
                 batch_size, input_dim, latent_dim, device)
            )
            results.append(result)

        # Wait for all processes to finish and collect results
        ensemble_models_centers = [result.get() for result in results]

    # Unpack models and centers
    ensemble_models, ensemble_centers = zip(*ensemble_models_centers)

    return ensemble_models, ensemble_centers


def test(ensemble_models, ensemble_centers, test_loader, device):
    s = evaluate_ensemble(ensemble_models, ensemble_centers, test_loader, device)
    return s


def train_and_append_model(i, seed, subset_size_class_0, num_splits, X_train_class_0, y_train_class_0, X_train_class_1,
                           y_train_class_1, y_train_semi_supervised, X_train_contaminated, batch_size, input_dim,
                           latent_dim, device):

    set_seed(seed)

    # Train the model using the subset
    model, normal_center, abnormal_center = f(
        subset_size_class_0, num_splits, X_train_class_0, y_train_class_0, X_train_class_1,
        y_train_class_1, y_train_semi_supervised, X_train_contaminated, batch_size,
        seed, input_dim, latent_dim, device, i)

    # Return the model and its centers for ensemble evaluation
    return model, (normal_center, abnormal_center)


# if __name__ == "__main__":
#     main_with_exhaustive_search()

