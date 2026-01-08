import os
import torch
import torch.nn as nn
import torchio as tio
from resnet_MM import resnet18
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from torch.distributions import Bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 1) UNCERTAINTY DECOMPOSITION
# ---------------------------
def uncertainty_decomposition_bernoulli(probs):
    """
    probs: Tensor shape (M, N) → M modelli, N samples
    """
    M, N = probs.shape

    mean_probs = torch.mean(probs, dim=0)

    total_entropy = Bernoulli(probs=mean_probs).entropy()          # (N,)
    entropies = Bernoulli(probs=probs).entropy()                   # (M, N)
    aleatoric_entropy = torch.mean(entropies, dim=0)               # (N,)
    epistemic_entropy = total_entropy - aleatoric_entropy          # (N,)

    return total_entropy, aleatoric_entropy, epistemic_entropy


# ---------------------------
# PATHS
# ---------------------------
results_dir = r"C:\Users\ludov\Scripts\results\optuna_mm_resnet18"
excel_path = r"C:\Users\ludov\Scripts\dati_cnr_new.xlsx"
base_dir = r"C:\Users\ludov\Scripts\registered_linear_l\cropped\hist_normed_2"
num_folds = 5

# ---------------------------
# LOAD BEST TRIAL ID AND PARAMS
# ---------------------------
best_id_file = os.path.join(results_dir, "optuna_best_trial_id.txt")
with open(best_id_file, "r") as f:
    best_trial_id = f.read().strip()
print(f"Best trial ID: {best_trial_id}")

best_params_file = os.path.join(results_dir, "optuna_best_params.json")
with open(best_params_file, "r") as f:
    best_trial_params = json.load(f)["params"]

dropout = best_trial_params["dropout"]


# ---------------------------
# LABELS
# ---------------------------
def load_label_mapping(excel_path):
    df = pd.read_excel(excel_path, usecols=['ID CLOUD', 'TLE'])
    return {row['ID CLOUD']: int(row['TLE']) for _, row in df.iterrows()}

label_mapping = load_label_mapping(excel_path)


# ---------------------------
# DATASET TEST
# ---------------------------
def create_multimodal_subjects(root_dirs, label_mapping, split_name="test"):
    """
    Create multi-modal subjects (MT, PD, R1) from separate folders.

    Args:
        root_dirs (dict): dictionary like {'MT': path1, 'PD': path2, 'R1': path3}
        label_mapping (dict): maps base_id -> label
        split_name (str): 'train' or 'test' (for logging)
    Returns:
        list[tio.Subject]
    """
    contrast_order = ['MT', 'PD', 'R1', 'R2']
    subjects = []
    norm_func = tio.ZNormalization()

    # get list of files from one folder (assuming same filenames across contrasts)
    ref_dir = root_dirs['R1']
    all_files = [f for f in os.listdir(ref_dir) if f.endswith('.nii')]

    for file in all_files:
        base_id = os.path.splitext(file)[0]
        tensors = []
        affines = []

        missing_modalities = []
        for contrast in contrast_order:
            path = os.path.join(root_dirs[contrast], file)
            if not os.path.exists(path):
                missing_modalities.append(contrast)
                continue

            # --- TorchIO-style canonical orientation and noise cleanup ---
            tio_img = tio.ScalarImage(path)
            #data = tio_img.data
            #tio_img = tio.ToCanonical()(tio_img)
            #data[data < 1e-5] = 0

            tio_img_norm = norm_func(tio_img)
            data_norm = tio_img_norm.data
            tensor = data_norm.clone() 
            tensors.append(tensor)
            affines.append(tio_img_norm.affine)

        if missing_modalities:
            print(f"⚠️ {split_name} subject {base_id} missing {missing_modalities} → skipped")
            continue

        image_tensor = torch.cat(tensors, dim=0)  # [3, D, H, W]
        print(image_tensor.shape)
        affine = affines[0]
        label = label_mapping.get(base_id, 0)

        image = tio.ScalarImage(tensor=image_tensor, affine=affine)
        subject = tio.Subject(
            image=image,
            label=torch.tensor(label, dtype=torch.long),
            participant_id=base_id
        )
        subjects.append(subject)

    print(f"✅ {split_name}: {len(subjects)} multi-modal subjects loaded.")
    return subjects

test_dir = {
        'MT': os.path.join(base_dir, 'MT\\test'),
        'PD': os.path.join(base_dir, 'PD\\test'),
        'R1': os.path.join(base_dir, 'R1\\test'),
        'R2': os.path.join(base_dir, 'R2\\test')

    }

test_subjects = create_multimodal_subjects(
    test_dir,
    label_mapping,
    split_name="test"
)

test_dataset = tio.SubjectsDataset(test_subjects)
test_loader = tio.SubjectsLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)


# ---------------------------
# EVALUATE ONE FOLD
# ---------------------------
def evaluate_fold(fold, weight_path):
    print(f"\n=== Testing FOLD {fold} ===")

    model = resnet18(
        sample_input_D=163,
        sample_input_H=193,
        sample_input_W=166,
        num_seg_classes=1,
        dropout=dropout
    ).to(device)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    output_file = os.path.join(results_dir, f"test_predictions_fold_{fold}.txt")
    open(output_file, 'w').close()  # clear old file

    fold_probs = []   # <---- raccolgo qui tutte le prob per i sample test
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'][tio.DATA].to(device)
            labels = batch['label'].float().unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds = (probs >= 0.5).float()

            fold_probs.extend(probs.squeeze().cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            with open(output_file, 'a') as f:
                for pid, label, prob, pred in zip(
                    batch['participant_id'],
                    labels.squeeze(),
                    probs.squeeze(),
                    preds.squeeze()
                ):
                    f.write(f"{pid}, True: {label.item()}, Pred: {pred.item()}, Prob: {prob.item():.4f}\n")


    acc = correct / total
    print(f"Fold {fold} Accuracy: {acc*100:.2f}%")

    with open(output_file, 'a') as f:
        f.write(f"\n[Fold {fold}] Test Accuracy: {acc*100:.2f}%\n")

    return acc, np.array(fold_probs)


# ---------------------------
# MAIN TESTING LOOP
# ---------------------------
all_acc = []
all_probs_folds = []  # <---- ensemble


for fold in range(num_folds):

    weight_path = os.path.join(
        results_dir,
        f"best_model_fold_{fold}_trial_{best_trial_id}.pth"
    )

    if not os.path.exists(weight_path):
        print(f"ERRORE: Non trovo i pesi per il fold {fold}: {weight_path}")
        continue

    acc, fold_probs = evaluate_fold(fold, weight_path)

    all_acc.append(acc)
    all_probs_folds.append(fold_probs)


# ---------------------------
# SAVE SUMMARY FILE
# ---------------------------
summary_path = os.path.join(results_dir, "accuracy_per_fold.txt")
with open(summary_path, "w") as f:
    for fold, acc in enumerate(all_acc):
        f.write(f"Fold {fold}: {acc*100:.2f}%\n")
    f.write(f"\nAccuracy media: {np.mean(all_acc)*100:.2f}%\n")

print("\n=== RISULTATI FINALI ===")
print("Accuracy per fold:", [f"{a*100:.2f}%" for a in all_acc])
print(f"Accuracy media: {np.mean(all_acc)*100:.2f}%")
print(f"Salvato summary in: {summary_path}")


# -------------------------------------------------------
#  ENSEMBLE UNCERTAINTY CALCULATION
# -------------------------------------------------------
print("\n=== COMPUTING ENSEMBLE UNCERTAINTIES ===")

ensemble_probs = torch.tensor(np.vstack(all_probs_folds), dtype=torch.float32)  # (M, N)

total_unc, aleatoric_unc, epistemic_unc = uncertainty_decomposition_bernoulli(ensemble_probs)
mean_probs = torch.mean(ensemble_probs, dim=0)

ids = [s['participant_id'] for s in test_subjects]
y_true = np.array([s['label'].item() for s in test_subjects])


df_out = pd.DataFrame({
    "ID": ids,
    "y_true": y_true,
    "prob_mean": mean_probs.numpy(),
    "unc_total": total_unc.numpy(),
    "unc_aleatoric": aleatoric_unc.numpy(),
    "unc_epistemic": epistemic_unc.numpy(),
})

out_path = os.path.join(results_dir, "uncertainty_results.csv")
df_out.to_csv(out_path, index=False)

print("Ensemble uncertainty saved in:")
print(out_path)

# -------------------------------------------------------
# CALCOLO NLL (Negative Log-Likelihood)
# -------------------------------------------------------
import torch.nn.functional as F

labels_tensor = torch.tensor(y_true, dtype=torch.float32)
probs_tensor = mean_probs.clone().detach().cpu()

# NLL
nll_value = F.binary_cross_entropy(probs_tensor, labels_tensor).item()
print(f"\nNegative Log-Likelihood (NLL): {nll_value:.4f}")

# -------------------------------------------------------
# CALCOLO ECE (Expected Calibration Error)
# -------------------------------------------------------
from torchmetrics.classification import BinaryCalibrationError

labels_tensor_long = torch.tensor(y_true, dtype=torch.long)

ece_l1 = BinaryCalibrationError(n_bins=3, norm='l1')
ece_l2 = BinaryCalibrationError(n_bins=3, norm='l2')
ece_max = BinaryCalibrationError(n_bins=3, norm='max')

ece_l1_val = ece_l1(probs_tensor, labels_tensor_long).item()
ece_l2_val = ece_l2(probs_tensor, labels_tensor_long).item()
ece_max_val = ece_max(probs_tensor, labels_tensor_long).item()

print("\n=== CALIBRAZIONE MODELLO (ECE) ===")
print(f"ECE (L1):  {ece_l1_val:.4f}")
print(f"ECE (L2):  {ece_l2_val:.4f}")
print(f"ECE (MAX): {ece_max_val:.4f}")

# -------------------------------------------------------
# SALVATAGGIO NEL CSV
# -------------------------------------------------------
# Ripetiamo NLL e ECE in tutte le righe (per semplicità)
df_out["NLL"] = nll_value
df_out["ECE_l1"] = ece_l1_val
df_out["ECE_l2"] = ece_l2_val
df_out["ECE_max"] = ece_max_val

df_out.to_csv(out_path, index=False)
print(f"\nCSV aggiornato con NLL e ECE salvato in: {out_path}")
