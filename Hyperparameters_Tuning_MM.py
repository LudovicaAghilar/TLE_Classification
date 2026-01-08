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

# --- import your model (adjust import path if needed) ---
from resnet_MM import resnet18

# -------------------------
# GLOBAL CONFIG (tweakable)
# -------------------------
num_epochs = 90
# default values (will be overridden by Optuna during trials)
default_lr = 3e-4
default_batch_size = 4
default_weight_decay = 0.0
patience = 30
patience_lr = 20
num_folds = 5

use_pretrained = True
pretrained_path = r"C:\Users\ludov\Scripts\resnet_18_23dataset.pth"

results_dir = os.path.join('results', f"optuna_mm_resnet18")
plots_dir = os.path.join('plots', f"optuna_mm_resnet18")
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

def create_multimodal_subjects(root_dirs, label_mapping, split_name="train"):
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
            tensor = data_norm.clone()  # already shape [1, D, H, W]
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

def get_transforms():
    val_transform = tio.Compose([])
    train_transform = tio.Compose([
        tio.RandomFlip(axes=('LR'), flip_probability=0.5),
        tio.RandomAffine(degrees=15, scales=(0.9, 1.1), translation=5),
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
        
        # Replico pesi conv1 monocanale sui tre canali
        conv1_weight = renamed_dict['conv1.weight']  # copia dei pesi per layer input
        conv1_weight_replicated = conv1_weight.repeat(1, 4, 1, 1, 1)  # Replicare il peso per i 4 canali
        renamed_dict['conv1.weight'] = conv1_weight_replicated  # Aggiorniamo i pesi del modello

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

# -------------------------
# TRAIN ONE FOLD (uses passed hyperparams)
# -------------------------
def train_one_fold(fold, train_subjects, val_subjects, train_transform, val_transform, device,
                   lr, batch_size, weight_decay, num_epochs_local, patience_local, patience_lr_local, train_mode_local, use_pretrained_local, pretrained_path_local, trial=None, dropout=0.3):
    train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)

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

    best_val_loss = float('inf')
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

        print(f"[Fold {fold+1}] Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # report intermediate result to optuna for pruning
        if trial is not None:
            # report validation loss (lower is better)
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                print(f"Trial pruned at fold {fold}, epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        # Save best weights for this fold (optional)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
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
        plt.ylim(0, max(max(train_losses + [0]), max(val_losses + [0])) + 0.1)
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
        plt.ylim(0, 1)  # <-- range 0-1 per tutte le accuracy
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(plots_dir, f"acc_fold_{fold+1}_{trial_tag}.png"))
        plt.close()
    except Exception as e:
        print("Could not save plot:", e)


    return best_val_loss  # we return the fold's best validation loss

# -------------------------
# OBJECTIVE for OPTUNA
# -------------------------
def objective(trial):
    # --- Suggest hyperparameters
    # loguniform for learning rate and weight decay
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)  # <--- nuovo parametro

     # GLOBAL SEED & DEVICE
    # -------------------------
    GLOBAL_SEED = 42
    set_seed(GLOBAL_SEED)  # seed globale per riproducibilità

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nStarting trial {trial.number}: lr={lr:.2e}, weight_decay={weight_decay:.2e}, batch_size={batch_size}, device={device}")

    # --- Load dataset and prepare folds (paths hard-coded like in your main)
    excel_path = r'C:\Users\ludov\Scripts\dati_cnr_new.xlsx'
    # Dataset paths
    # === Define paths for multimodal data ===
    base_dir = r"C:\Users\ludov\Scripts\registered_linear_l\cropped\hist_normed_2"

    train_dirs = {
        'MT': os.path.join(base_dir, 'MT\\train'),
        'PD': os.path.join(base_dir, 'PD\\train'),
        'R1': os.path.join(base_dir, 'R1\\train'),
        'R2': os.path.join(base_dir, 'R2\\train')

    }

    label_mapping = load_label_mapping(excel_path)
    ref_dir = train_dirs['R1']
    all_files = [f for f in os.listdir(ref_dir) if f.endswith(".nii")]

    train_val_subjects = create_multimodal_subjects(train_dirs, label_mapping, split_name="train")

    if len(train_val_subjects) < num_folds:
        raise RuntimeError(f"Not enough samples ({len(train_val_subjects)}) for {num_folds}-fold CV")

    train_transform, val_transform = get_transforms()
    train_val_labels = [s['label'].item() for s in train_val_subjects]

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=GLOBAL_SEED)

    # definisco seed per fold fisso
    fold_seeds = [GLOBAL_SEED + i for i in range(num_folds)]

    fold_val_losses = []
   
    try:
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_subjects, train_val_labels)):

            # seed specifico per il fold
            set_seed(fold_seeds[fold])
            print(f"\nTrial {trial.number} - Fold {fold+1}/{num_folds}")
            train_subjs = [train_val_subjects[i] for i in train_idx]
            val_subjs = [train_val_subjects[i] for i in val_idx]

            # Save fold subject IDs for reproducibility (per trial & fold)
            train_ids = [s['participant_id'] for s in train_subjs]
            val_ids = [s['participant_id'] for s in val_subjs]
            fold_log_path = os.path.join(results_dir, f"fold_{fold}_trial_{trial.number}_splits.txt")
            with open(fold_log_path, "w") as f:
                f.write(f"Trial {trial.number} - Fold {fold+1}/{num_folds}\n")
                f.write(f"Train subjects ({len(train_ids)}): {train_ids}\n")
                f.write(f"Val subjects ({len(val_ids)}): {val_ids}\n")
                f.write(f"Train labels: {[s['label'].item() for s in train_subjs]}\n")
                f.write(f"Val labels: {[s['label'].item() for s in val_subjs]}\n")
            print(f"  Fold splits saved: {fold_log_path}")

            best_val_loss = train_one_fold(
                fold=fold,
                train_subjects=train_subjs,
                val_subjects=val_subjs,
                train_transform=train_transform,
                val_transform=val_transform,
                device=device,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                num_epochs_local=num_epochs,
                patience_local=patience,
                patience_lr_local=patience_lr,
                train_mode_local='partial',  # or 'partial' depending on your preference
                use_pretrained_local=use_pretrained,
                pretrained_path_local=pretrained_path,
                trial=trial,
                dropout=dropout
            )
            fold_val_losses.append(best_val_loss)

    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} pruned at fold {fold+1}")
        raise

    mean_val_loss = float(np.mean(fold_val_losses))
    print(f"Trial {trial.number} completed. Mean val loss across folds: {mean_val_loss:.6f}")

    # return the metric to minimize (validation loss)
    return mean_val_loss

# -------------------------
# ENTRYPOINT: create study & run optimization
# -------------------------
if __name__ == "__main__":
    # Optuna study configuration
    N_TRIALS = 8
    sampler = TPESampler(seed=500)
    pruner = MedianPruner(n_warmup_steps=2)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
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
