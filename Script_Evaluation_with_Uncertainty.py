import os
import torch
import torch.nn as nn
import torchio as tio
from resnet_modified import resnet18
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
results_dir = r"C:\Users\ludov\Scripts\results\optuna_singolo_R1_resnet18_FIN"
excel_path = r"C:\Users\ludov\Scripts\dati_cnr_new.xlsx"
test_dir = r"C:\Users\ludov\Scripts\registered_linear_l\cropped\hist_normed_2\R1\test"
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
def create_subject(img_path, label, pid):
    image = tio.ScalarImage(img_path)
    return tio.Subject(
        image=image,
        label=torch.tensor(label, dtype=torch.long),
        participant_id=pid,
    )


test_files = [f for f in os.listdir(test_dir) if f.endswith(".nii")]
test_subjects = [
    create_subject(
        os.path.join(test_dir, f),
        label_mapping[os.path.splitext(f)[0]],
        os.path.splitext(f)[0]
    )
    for f in test_files if os.path.splitext(f)[0] in label_mapping
]

test_transform = tio.Compose([tio.ZNormalization()])
test_dataset = tio.SubjectsDataset(test_subjects, transform=test_transform)
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

            preds = probs.round()

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

