import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, f1_score

# Percorso alla cartella
folder = r"C:\Users\ludov\Scripts\results\final_training_R2_resnet18"

files = sorted(
    [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("test_predictions_fold_")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

# Dizionario per salvare i risultati
fold_data = {}

for i, file in enumerate(files):
    data = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if "," not in line:  # Ignora righe come "[Fold 5]
                continue
            parts = line.split(", ")
            subj = parts[0]
            true = float(parts[1].split(": ")[1])
            pred = float(parts[2].split(": ")[1])
            prob = float(parts[3].split(": ")[1])
            data.append([subj, true, pred, prob])
    df = pd.DataFrame(data, columns=["ID", "True", "Pred", "Prob"])
    fold_data[i] = df

# 1️⃣ Calcola metriche per ogni fold
metrics = []
for i, df in fold_data.items():
    y_true = df["True"]
    y_pred = df["Pred"]
    y_prob = df["Prob"]
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan  # Se c'è solo una classe, AUROC non può essere calcolato
    metrics.append([acc, sens, spec, f1, auc])

metrics_df = pd.DataFrame(metrics, columns=["Accuracy", "Sensitivity", "Specificity", "F1", "AUROC"])
# Calcola media e deviazione standard
mean_vals = metrics_df.mean()
std_vals = metrics_df.std()

print("\n📊 Metriche sui 5 fold (media ± std):")
for m in metrics_df.columns:
    print(f"{m}: {mean_vals[m]:.4f} ± {std_vals[m]:.4f}")

# 2️⃣ ENSEMBLE – media delle probabilità
# Unisci tutti i fold per ID
ensemble_df = pd.concat(fold_data.values()).groupby("ID").agg({
    "True": "first",
    "Prob": "mean"
})
ensemble_df["Pred_ensemble"] = (ensemble_df["Prob"] > 0.5).astype(float)

# Metriche ensemble
acc_ens = accuracy_score(ensemble_df["True"], ensemble_df["Pred_ensemble"])
sens_ens = recall_score(ensemble_df["True"], ensemble_df["Pred_ensemble"], pos_label=1)
spec_ens = recall_score(ensemble_df["True"], ensemble_df["Pred_ensemble"], pos_label=0)
f1_ens = f1_score(ensemble_df["True"], ensemble_df["Pred_ensemble"])
auc_ens = roc_auc_score(ensemble_df["True"], ensemble_df["Prob"])

print("\n🤖 Metriche ENSEMBLE (media probabilità):")
print(f"Accuracy: {acc_ens:.4f}, Sensitivity: {sens_ens:.4f}, Specificity: {spec_ens:.4f}, F1: {f1_ens:.4f}, AUROC: {auc_ens:.4f}")

# 3️⃣ MAJORITY VOTING
# Per ogni soggetto, maggioranza tra le predizioni dei 5 fold
vote_df = pd.concat(fold_data.values()).groupby("ID").agg({
    "True": "first",
    "Pred": lambda x: 1.0 if np.mean(x) > 0.5 else 0.0
}).reset_index()

acc_mv = accuracy_score(vote_df["True"], vote_df["Pred"])
sens_mv = recall_score(vote_df["True"], vote_df["Pred"], pos_label=1)
spec_mv = recall_score(vote_df["True"], vote_df["Pred"], pos_label=0)
f1_mv = f1_score(vote_df["True"], vote_df["Pred"])

print("\n🗳️ Metriche MAJORITY VOTING:")
print(f"Accuracy: {acc_mv:.4f}, Sensitivity: {sens_mv:.4f}, Specificity: {spec_mv:.4f}, F1: {f1_mv:.4f}")
