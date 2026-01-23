# optimize_with_optuna.py
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn.model_selection import StratifiedKFold
import torchio as tio
from scipy import stats
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pickle

# --- import your model (adjust import path if needed) ---
from resnet_modified import resnet18

# -------------------------
# GLOBAL CONFIG (tweakable)
# -------------------------
num_epochs = 50
# default values (will be overridden by Optuna during trials)
default_lr = 3e-4
default_batch_size = 4
default_weight_decay = 0.0
patience = 20
patience_lr = 20
num_folds = 5

use_pretrained = True
pretrained_path = r"/home/labbioimm/Scrivania/resnet_18_23dataset.pth"
contrast = 'MT'

results_dir = os.path.join('results', f"optuna_singolo_{contrast}_resnet18_tuning_fold_1")
plots_dir = os.path.join('plots', f"optuna_singolo_{contrast}_resnet18_tuning_fold_1")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

EPS = 1e-12

# -------------------------
# UTILITIES
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_label_mapping(excel_path):
    df = pd.read_excel(excel_path, usecols=['ID CLOUD', 'TLE'])
    return {row['ID CLOUD']: int(row['TLE']) for _, row in df.iterrows()}

def create_subject(img_path, label, participant_id):
    image = tio.ScalarImage(img_path)
    return tio.Subject(
        image=image,
        label=torch.tensor(label, dtype=torch.long),
        participant_id=participant_id,
    )

def get_transforms():
    val_transform = tio.Compose([tio.ZNormalization()])
    train_transform = tio.Compose([
        tio.RandomFlip(axes=('LR'), flip_probability=0.5),
        tio.RandomAffine(degrees=15, scales=(0.9, 1.1), translation=5),
        tio.ZNormalization()
    ])
    return train_transform, val_transform

# -------------------------
# MODEL SETUP (refactored to accept args)
# -------------------------
def setup_model(device, train_mode='partial', pretrained_path=None, use_pretrained=False, dropout=0.3):
    model = resnet18(
        sample_input_D=163,
        sample_input_H=193,
        sample_input_W=166,
        num_seg_classes=1,
        dropout=dropout  # <--- passaggio qui
    ).to(device)

    if use_pretrained and pretrained_path and os.path.exists(pretrained_path):
        pretrain = torch.load(pretrained_path, map_location=device)
        pretrain_names = pretrain.get('state_dict', pretrain)
        renamed_dict = {k.replace('module.', ''): v for k, v in pretrain_names.items()}
        missing, unexpected = model.load_state_dict(renamed_dict, strict=False)
        if missing: print("⚠️ Missing keys:", missing)
        if unexpected: print("⚠️ Unexpected keys:", unexpected)
        del pretrain, pretrain_names, renamed_dict
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if train_mode == 'partial':
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith('fc') or name.startswith('layer4') or name.startswith('layer3') or "downsample" in name
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm3d) and (
                name == 'bn1' or name.startswith('layer1') or name.startswith('layer2')
            ):
                module.eval()
                module.train = lambda _: module
        print("Training only some layers (partial)")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("Training all layers (full)")

    return model

def save_augmented_images(dataset, fold, max_images=10):
    import nibabel as nib
    augmented_dir = os.path.join(results_dir, f"augmented_fold_{fold}")
    os.makedirs(augmented_dir, exist_ok=True)
    
    for i, subj in enumerate(dataset):
        img = subj['image'][tio.DATA].numpy().squeeze()
        filename = f"{subj['participant_id']}_augmented.nii.gz"
        path = os.path.join(augmented_dir, filename)
        nib.save(nib.Nifti1Image(img, affine=np.eye(4)), path)
        if i >= max_images - 1:
            break

def save_fold_subjects_txt(fold_idx, train_subjects, val_subjects, out_dir, num_folds):
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.join(out_dir, f"fold_{fold_idx+1}_subjects.txt")

    with open(filename, "w") as f:
        f.write(f"Fold {fold_idx+1}/{num_folds}\n")
        f.write("=" * 40 + "\n\n")

        # ---- TRAIN ----
        f.write(f"Train subjects ({len(train_subjects)}):\n")
        for s in train_subjects:
            pid = s['participant_id']
            label = int(s['label'].item())
            f.write(f"  {pid} | label={label}\n")

        f.write("\n")

        # ---- VALIDATION ----
        f.write(f"Validation subjects ({len(val_subjects)}):\n")
        for s in val_subjects:
            pid = s['participant_id']
            label = int(s['label'].item())
            f.write(f"  {pid} | label={label}\n")

    print(f"✅ Fold subjects saved to: {filename}")

# -------------------------
# TRAIN ONE FOLD (uses passed hyperparams)
# -------------------------
def train_one_fold(fold, train_subjects, val_subjects, train_transform, val_transform, device,
                   lr, batch_size, weight_decay, num_epochs_local, patience_local, patience_lr_local, train_mode_local, use_pretrained_local, pretrained_path_local, trial=None, dropout=0.3):
    train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)

    history = {
    'loss_tr': [],
    'loss_val': [],
    'acc_tr': [],
    'acc_val': []
}

    # salva immagini augmented
    # save_augmented_images(train_dataset, fold)

    val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)

    train_loader = tio.SubjectsLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = tio.SubjectsLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    model = setup_model(
        device,
        train_mode=train_mode_local,
        pretrained_path=pretrained_path_local,
        use_pretrained=use_pretrained_local,
        dropout=dropout
    )

    train_labels_tensor = torch.tensor([s['label'].item() for s in train_subjects], dtype=torch.float)
    
    num_pos = (train_labels_tensor == 1).sum()
    num_neg = (train_labels_tensor == 0).sum()

    pos_weight = (num_neg / (num_pos + EPS)).to(torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_lr_local)

    best_val_loss_epoch = float("inf")   # early stopping
    best_val_loss_optuna = float("inf")  # pruning / objective
    patience_counter = 0

    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs_local):
        # TRAIN
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs_local}", unit="batch"):
            images = batch['image'][tio.DATA].to(device)
            labels = batch['label'].float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).round()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # VALIDATION
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'][tio.DATA].to(device)
                labels = batch['label'].float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs).round()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        #scheduler.step(avg_val_loss)

        history['loss_tr'].append(avg_train_loss)
        history['loss_val'].append(avg_val_loss)
        history['acc_tr'].append(avg_train_acc)
        history['acc_val'].append(avg_val_acc)

        print(f"[Fold {fold+1}] Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        best_val_loss_optuna = min(best_val_loss_optuna, avg_val_loss)

        if trial is not None:
            trial.report(best_val_loss_optuna, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch + 1}")
                raise optuna.exceptions.TrialPruned()

        # Save best weights for this fold (optional)
        if avg_val_loss < best_val_loss_epoch:
            best_val_loss_epoch = avg_val_loss
            patience_counter = 0
            # save checkpoint of best weights for fold (named by trial id if present)
            trial_tag = f"trial_{trial.number}" if trial is not None else "no_trial"
            ckpt_name = os.path.join(results_dir, f"best_model_fold_{fold}_{trial_tag}.pth")
            torch.save({'state_dict': model.state_dict()}, ckpt_name)
        else:
            patience_counter += 1

        if patience_counter >= patience_local:
            print(f"Early stopping at epoch {epoch+1} for fold {fold}")
            break

    # Save last weights
    trial_tag = f"trial_{trial.number}" if trial is not None else "no_trial"
    #ckpt_last = os.path.join(results_dir, f"last_model_fold_{fold}_{trial_tag}.pth")
    #torch.save({'state_dict': model.state_dict()}, ckpt_last)

    # Save plots per fold (loss + accuracy)
    try:
        early_stop_epoch = epoch - patience_counter + 1 if patience_counter >= patience_local else None

        # ---- LOSS ----
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        if early_stop_epoch is not None:
            plt.axvline(x=early_stop_epoch-1, color='red', linestyle='--', label='Early Stop')
        plt.legend()
        plt.title(f"Fold {fold+1} Loss (trial {trial_tag})")
        plt.ylim(0.4, 0.8)  # <-- limiti per la loss
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(plots_dir, f"loss_fold_{fold+1}_{trial_tag}.png"))
        plt.close()

        # ---- ACCURACY ----
        plt.figure()
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Val Acc")
        if early_stop_epoch is not None:
            plt.axvline(x=early_stop_epoch-1, color='red', linestyle='--', label='Early Stop')
        plt.legend()
        plt.title(f"Fold {fold+1} Accuracy (trial {trial_tag})")
        plt.ylim(0.4, 1)  # <-- limiti per l'accuracy
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(plots_dir, f"acc_fold_{fold+1}_{trial_tag}.png"))
        plt.close()
    except Exception as e:
        print("Could not save plot:", e)

    trial_tag = f"trial_{trial.number}" if trial is not None else "no_trial"

    history_path = os.path.join(
        results_dir,
        f"train_history_fold_{fold}_{trial_tag}.p"
    )

    with open(history_path, 'wb') as f:
        pickle.dump(history, f)


    return best_val_loss_optuna  # we return the fold's best validation loss

FIXED_FOLD = 1  # usa il fold 0 (puoi cambiarlo)

# -------------------------
# PREPARE FIXED FOLD SPLITS (ONCE)
# -------------------------
GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)

excel_path = r'/home/labbioimm/Scrivania/dati_cnr_new.xlsx'
base_dir = r'/home/labbioimm/Scrivania/registered_linear_l/cropped/hist_normed_2/MT'
train_val_dir = os.path.join(base_dir, 'train')

label_mapping = load_label_mapping(excel_path)
all_files = [f for f in os.listdir(train_val_dir) if f.endswith(".nii")]

train_val_subjects = [
    create_subject(
        os.path.join(train_val_dir, f),
        label_mapping[os.path.splitext(f)[0]],
        os.path.splitext(f)[0]
    )
    for f in all_files
    if os.path.splitext(f)[0] in label_mapping
]

train_val_labels = [s['label'].item() for s in train_val_subjects]

kf = StratifiedKFold(
    n_splits=num_folds,
    shuffle=True,
    random_state=GLOBAL_SEED
)

fold_splits = list(kf.split(train_val_subjects, train_val_labels))

# scegli il fold fisso
train_idx, val_idx = fold_splits[FIXED_FOLD]

FIXED_TRAIN_SUBJS = [train_val_subjects[i] for i in train_idx]
FIXED_VAL_SUBJS   = [train_val_subjects[i] for i in val_idx]

save_fold_subjects_txt(
    fold_idx=FIXED_FOLD,
    train_subjects=FIXED_TRAIN_SUBJS,
    val_subjects=FIXED_VAL_SUBJS,
    out_dir=results_dir,
    num_folds=num_folds
)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    set_seed(GLOBAL_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"\nTrial {trial.number} | "
        f"lr={lr:.2e}, wd={weight_decay:.2e}, "
        f"bs={batch_size}, dropout={dropout:.2f}"
    )

    train_transform, val_transform = get_transforms()

    try:
        best_val_loss = train_one_fold(
            fold=FIXED_FOLD,
            train_subjects=FIXED_TRAIN_SUBJS,
            val_subjects=FIXED_VAL_SUBJS,
            train_transform=train_transform,
            val_transform=val_transform,
            device=device,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            num_epochs_local=num_epochs,
            patience_local=patience,
            patience_lr_local=patience_lr,
            train_mode_local='partial',
            use_pretrained_local=use_pretrained,
            pretrained_path_local=pretrained_path,
            trial=trial,
            dropout=dropout
        )
    except optuna.exceptions.TrialPruned:
        print(f"⚠️ Trial {trial.number} è stato interrotto (pruned) da Optuna.")
        raise  # Rilancia l'eccezione per farla gestire a Optuna correttamente

    return best_val_loss

# -------------------------
# ENTRYPOINT: create study & run optimization
# -------------------------
if __name__ == "__main__":
    # Optuna study configuration
    N_TRIALS = 50

    contrast = "MT"

    contrast_seeds = {
    "R1": 100,
    "R2": 200,
    "PD": 300,
    "MT": 400,
}

    sampler = TPESampler(seed=contrast_seeds[contrast])

    pruner = MedianPruner(
    n_startup_trials=10,  # first 15 trials not pruned
    n_warmup_steps=20,   # don’t prune before epoch 30
)
    # === PATH ASSOLUTO AL DB ===
    db_path = os.path.abspath(
        os.path.join(results_dir, "optuna_study.db")
    )

    print("Optuna DB:", db_path)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{db_path}",
        study_name="resnet18_optuna",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=None, gc_after_trial=True)

    print("Study finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    # Save best params to disk
    best_params_path = os.path.join(results_dir, "optuna_best_params.json")
    import json
    with open(best_params_path, "w") as f:
        json.dump({"value": trial.value, "params": trial.params}, f, indent=2)

    print(f"Best params saved to {best_params_path}")

    # Save ALSO the best trial ID
    best_trial_path = os.path.join(results_dir, "optuna_best_trial_id.txt")
    with open(best_trial_path, "w") as f:
        f.write(str(trial.number))

    print(f"Best trial ID saved to {best_trial_path}")
