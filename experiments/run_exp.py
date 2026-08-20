import glob
import random
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from deepod.models.tabular import PReNet, DeepSAD, RoSAS, DevNet
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from WSAD_DT import train, test
from pyod.models.xgbod import XGBOD
# List of seeds for reproducibility
seeds = [100, 1000]
from adbench.baseline.GANomaly.run import GANomaly
WSAD_DT_ = None
# Dictionary containing the algorithms to be tested
algo_dic = {'WSAD_DT': WSAD_DT_ ,  'DeepSAD': DeepSAD, 'DevNet': DevNet, 'FeaWAD': DeepSAD, 'GANanomaly': GANomaly, 'PreNet': PReNet, 'RoSAS':RoSAS, 'XBGOD':XGBOD}
###

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    # Iterate over each algorithm
    for key, algo in algo_dic.items():
        results = {}  # Dictionary to store results for each dataset

        # Iterate over each dataset file
        for data_name in glob.glob('Classical/*.npz'):

            try:
                if 'jpg' in data_name or 'donor' in data_name:
                    continue

                # Load the dataset
                data = np.load(data_name, allow_pickle=True)
                X, Y = data['X'], data['y']
                if X.shape[0] > 20000: continue
                X = MinMaxScaler().fit_transform(X)
                set_seed(1)
                device = 'cpu'

                auc_roc_scores = []
                avg_precision_scores = []
                for ss in [1, 5, 10]:
                    # Step 1: Initial train-test split
                    # Step 1: Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=ss,
                                                                        stratify=Y)

                    # Step 2: Identify normal and anomalous indices in the training data
                    normal_indices_train = np.where(y_train == 0)[0]
                    anomalous_indices_train = np.where(y_train == 1)[0]

                    # Step 3: Select a small percentage of labeled anomalies
                    label_percentage = 0.05
                    num_labeled_anomalies = max(int(label_percentage * len(anomalous_indices_train)), 5)
                    labeled_anomalous_indices = np.random.choice(anomalous_indices_train,
                                                                 min(num_labeled_anomalies,
                                                                     len(anomalous_indices_train)),
                                                                 replace=False)

                    # Step 4: Identify remaining non-labeled anomalies
                    non_labeled_anomalous_indices = np.setdiff1d(anomalous_indices_train, labeled_anomalous_indices)

                    # Step 5: Contaminate the training data
                    contamination_ratio = 0.
                    num_non_labeled_anomalies_to_add = 0 #int(contamination_ratio * len(non_labeled_anomalous_indices)) comment in for contamination in the training data
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
                    num_mislabeled_normals = 0  # num_non_labeled_anomalies_to_add  # int(mislabel_percentage * len(normal_indices_train))

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

                    # y = Y.copy()
                    # y[remaining_indices] = 0

                    # Iterate over each seed
                    for seed in seeds:
                        set_seed(seed)
                        # Initialize and train the classifier

                        if key == 'WSAD_DT':
                            ensemble_models, ensemble_centers = train(5, X_train_contaminated, y_train_semi_supervised,
                                                                      seed)
                            scores = test(ensemble_models, ensemble_centers, test_loader, device)

                        else:
                         clf = algo(device='cpu', random_state=seed, verbose=0)
                         clf.fit(X_train_contaminated, y_train_semi_supervised)  # full data
                         scores = clf.decision_function(X_test)






                        auc_roc = roc_auc_score(y_test, scores)
                        avg_precision = average_precision_score(y_test, scores)

                        auc_roc_scores.append(auc_roc)
                        avg_precision_scores.append(avg_precision)

                # Calculate the average metrics across all seeds
                avg_auc_roc = np.mean(auc_roc_scores)
                avg_avg_precision = np.mean(avg_precision_scores)

                print(f"\nDataset: {data_name}")
                print(f"{key} - Average AUC-ROC: {avg_auc_roc:.4f}")
                print(f"{key} - Average Precision Score: {avg_avg_precision:.4f}")

                # Store results in the dictionary
                results[data_name] = {
                    'avg_auc_roc': avg_auc_roc,
                    'avg_avg_precision': avg_avg_precision
                }

            except Exception as e:
                print(f"Error processing {data_name}: {e}")
                continue

    # Save results to a CSV file
    # with open(f'icml_results/{key}_average_results_light_weight_nn.csv', 'a') as file:
    #     file.write('Dataset,Avg AUC-ROC,Avg AP\n')
    #     for data_name, metrics in results.items():
    #         file.write(f"{data_name},{metrics['avg_auc_roc']:.4f},{metrics['avg_avg_precision']:.4f}\n")