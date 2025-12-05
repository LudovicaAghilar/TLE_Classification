import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torchio as tio
import nibabel as nib
from resnet_modified import resnet18   # Assicurati che resnet.py sia nel PYTHONPATH
import json

os.chdir(r"C:\Users\ludov\Scripts")

# --------------------- PARAMETRI ---------------------
DATA_DIR       = r"C:\Users\ludov\OneDrive\Desktop\CNR\TLE\R1\test_R1_histogram_normalized"
OUT_DIR        = r"C:\Users\ludov\OneDrive\Desktop\CNR\TLE\R1\test_R1_histogram_normalized\HeatMap_avg_prova"
TARGET_LAYER   = "layer4"           # Cambia se necessario
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA, FPS     = 0.4, 10
results_dir = r"C:\Users\ludov\OneDrive\Desktop\CNR\TLE\R1"
os.makedirs(OUT_DIR, exist_ok=True)
# -----------------------------------------------------


# ---------------- LEGGI ID DI TEST -------------------
def load_test_labels(txt_path):
    labels = {}
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # ------ Fermati quando trovi la parte delle metriche ------
            if "accuracy" in line.lower():
                break
            
            parts = line.split(",")
            subj_id = parts[0].strip()
            label = parts[1].strip() if len(parts) > 1 else None
            try:
                label = int(label)
            except:
                # Se non è numerica, la tieni come stringa
                pass
            labels[subj_id] = label
    return labels


# Usa uno qualunque dei file test_predictions (contengono gli stessi ID)
TEST_TXT_PATH  = r"C:\Users\ludov\OneDrive\Desktop\CNR\TLE\R1\test_predictions_fold_0.txt"
TEST_LABELS = load_test_labels(TEST_TXT_PATH)
TEST_IDS = set(TEST_LABELS.keys())
print(f"Trovati {len(TEST_IDS)} ID con label di test.")

# -----------------------------------------------------


# ---------------- CLASSE GRAD-CAM 3D -----------------
# ---------------- CLASSE GRAD-CAM 3D -----------------
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.tlayer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.tlayer.register_forward_hook(fwd_hook))
        self.hooks.append(self.tlayer.register_full_backward_hook(bwd_hook))

    def generate(self, x, class_idx):
        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward(retain_graph=True)

        pooled = self.gradients.mean(dim=[0, 2, 3, 4])       # [C]
        heatmap = torch.einsum('cdhw,c->dhw', self.activations[0], pooled)
        heatmap = torch.relu(heatmap)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.cpu().numpy()
    
    def close(self):
        for h in self.hooks:
            h.remove()



# -----------------------------------------------------


# ---------------- PREPROCESSING ----------------------
preproc = tio.Compose([
    tio.ZNormalization(),
])

padding = tio.CropOrPad(
    (163,193,166),
    only_pad='True'
)

def load_volume(path):
    subject = tio.Subject(img=tio.ScalarImage(path))

    """ if "3TLE_NIGUARDA" in path or "3TLE_HC" in path:
        tensor = subject['img'].data
        affine = subject['img'].affine
        tensor = tensor.permute(0, 3, 1, 2)  # (1, x, y, z)
        subject['img'] = tio.ScalarImage(tensor=tensor, affine=affine)
        subject = padding(subject)
        print(f"Immagine trasformata: {path}: {subject['img'].shape}") """

    return preproc(subject)['img'].data  # (1, D, H, W)


def upsample_heatmap(hm, target_shape, order=3):
    zoom_factors = [t / s for t, s in zip(target_shape, hm.shape)]
    hm_up = zoom(hm, zoom_factors, order=order)
    if hm_up.max() > 0:
        hm_up /= (hm_up.max() + 1e-8)
    return hm_up

import math

def plot_orthogonal_planes(volume, heatmap, alpha=0.2, save_path=None, title=None):
    D, H, W = volume.shape

    # Slice centrali
    sag_idx = W // 2    # piano sagittale
    cor_idx = H // 2    # piano coronale
    ax_idx  = D // 2     # piano assiale

    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    # Sagittal
    axes[2].imshow(np.rot90(vol_norm[:, :, sag_idx]), cmap='gray')
    axes[2].imshow(np.rot90(heatmap[:, :, sag_idx]), cmap='jet', alpha=alpha)
    axes[2].set_title('Axial')
    axes[2].axis('off')

    # Coronal
    axes[1].imshow(np.rot90(vol_norm[:, cor_idx, :]), cmap='gray')
    axes[1].imshow(np.rot90(heatmap[:, cor_idx, :]), cmap='jet', alpha=alpha)
    axes[1].set_title('Coronal')
    axes[1].axis('off')

    # Axial
    axes[0].imshow(np.rot90(vol_norm[ax_idx, :, :]), cmap='gray')
    axes[0].imshow(np.rot90(heatmap[ax_idx, :, :]), cmap='jet', alpha=alpha)
    axes[0].set_title('Sagittal')
    axes[0].axis('off')

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# -----------------------------------------------------


# ---------------- CARICO I 5 MODELLI -----------------
FOLDS = range(5)
MODELS = []
modelli = []

best_id_file = os.path.join(results_dir, "optuna_best_trial_id.txt")
with open(best_id_file, "r") as f:
    best_trial_id = f.read().strip()
print(f"Best trial ID: {best_trial_id}")

best_params_file = os.path.join(results_dir, "optuna_best_params.json")
with open(best_params_file, "r") as f:
    best_trial_params = json.load(f)["params"]

dropout = best_trial_params["dropout"]  # usa il dropout ottimale


for f in FOLDS:
    wpath = os.path.join(results_dir,f"best_model_fold_{f}_trial_{best_trial_id}.pth")
    state = torch.load(wpath, map_location=DEVICE)
    model = resnet18(sample_input_D=163, sample_input_H=193, sample_input_W=166,
                     num_seg_classes=1, dropout=dropout).to(DEVICE)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    MODELS.append(GradCam3D(model, model.layer4))
    modelli.append(model)
print(f"Caricati {len(MODELS)} modelli (fold 0–4).")
# -----------------------------------------------------


# ---------------- LOOP SUI VOLUMI --------------------
nii_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii*")))
print(f"Nella cartella immagini trovati {len(nii_files)} file.")

# --- Carica label vere ---
TEST_LABELS = load_test_labels(TEST_TXT_PATH)
TEST_IDS = set(TEST_LABELS.keys())

for path in nii_files:
    base = os.path.basename(path)
    subj_prefix = base.split(".")[0]

    if subj_prefix not in TEST_IDS:
        print(f"{subj_prefix}: non è nel set di test → skip.")
        continue 

    label_true = TEST_LABELS[subj_prefix]

    hm_raw_path = os.path.join(OUT_DIR, f"{subj_prefix}_gradcam_mean.nii.gz")
    #out_gif = os.path.join(OUT_DIR, f"{subj_prefix}_gradcam_mean.gif")
    if os.path.exists(hm_raw_path):
        print(f"{subj_prefix}: Heatmap media già esistente → skip.")
        continue 

    
    try:
        print(f"{subj_prefix}: preprocessing & Grad-CAM media…")
        img_nii = nib.load(path)
        affine = img_nii.affine
        vol = load_volume(path).unsqueeze(0).to(DEVICE)   # (1,1,D,H,W)
        vol_np = vol.cpu().numpy()[0, 0]

        probs = []
        # First pass: get probabilities
        for model in modelli:
            with torch.no_grad():
                output = model(vol)
                prob = torch.sigmoid(output).item()
                probs.append(prob)

        # Compute ensemble probability
        avg_prob = np.mean(probs)
        pred_class = 1 if avg_prob > 0.5 else 0

        print(f"{subj_prefix}: true={label_true}, pred={pred_class}, prob={avg_prob:.3f}")

        heatmaps = []

        for gcam in MODELS:
            hm = gcam.generate(vol, class_idx=0)  # Guided Backprop
            hm = upsample_heatmap(hm, vol_np.shape)
            heatmaps.append(hm)

        # Media ensemble
        heatmap_mean = np.mean(heatmaps, axis=0)
        vol_np = vol.cpu().numpy()[0, 0]

        # --- Save NIfTI heatmap ---
        heatmap_nifti = nib.Nifti1Image(
            heatmap_mean.astype(np.float32),
            affine=affine
        )
        nib.save(heatmap_nifti, hm_raw_path)

        # --- Save visualization PNG ---
        out_path = os.path.join(OUT_DIR, f"{subj_prefix}_gradcam.png")
        title = f"{subj_prefix} | true={label_true} | pred={pred_class})"
        plot_orthogonal_planes(vol_np, heatmap_mean, alpha=ALPHA, 
                       save_path=out_path, title=title)

        print(f"Saved Grad-CAM for {subj_prefix}")
        

    except Exception as e:
        print(f"Errore con {subj_prefix}: {e}")


# -----------------------------------------------------

# Chiudo hook
for gcam in MODELS:
    gcam.close()

