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

# --------------------- PARAMETRI ---------------------
DATA_DIR    = r"C:\Users\ludov\OneDrive\Desktop\CNR\TLE\R1\test_R1_histogram_normalized"
OUT_DIR     = r"C:\Users\ludov\OneDrive\Desktop\CNR\TLE\R1\test_R1_histogram_normalized\HeatMap_avg_vanilla_smooth_fast_prova"
TARGET_LAYER = "layer4"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA       = 0.4
results_dir = r"C:\Users\ludov\OneDrive\Desktop\CNR\TLE"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LEGGI ID DI TEST -------------------
def load_test_labels(txt_path):
    labels = {}
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "accuracy" in line.lower():
                break
            parts = line.split(",")
            subj_id = parts[0].strip()
            label = parts[1].strip() if len(parts) > 1 else None
            try:
                label = int(label)
            except Exception:
                pass
            labels[subj_id] = label
    return labels

TEST_TXT_PATH = r"C:\Users\ludov\OneDrive\Desktop\CNR\TLE\R1\test_predictions_fold_0.txt"
TEST_LABELS = load_test_labels(TEST_TXT_PATH)
TEST_IDS = set(TEST_LABELS.keys())
print(f"Trovati {len(TEST_IDS)} ID con label di test.")

# ---------------- GRAD-CAM 3D ------------------------
class GradCAM3D:
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _get_target_layer(self):
        module = self.model
        for name in self.target_layer_name.split("."):
            module = getattr(module, name)
        return module

    def _register_hooks(self):
        target_layer = self._get_target_layer()

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_backward_hook(bwd_hook)

    def generate(self, x, class_idx=0):
        """
        x: (1, 1, D, H, W)
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if logits.dim() == 1 or logits.shape[-1] == 1:
            val = logits.view(-1)[0]
        else:
            val = logits[0, class_idx]

        val.backward(retain_graph=True)

        activations = self.activations   # (1, C, D, H, W)
        gradients   = self.gradients     # (1, C, D, H, W)

        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)   # (1, C, 1, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=False) # (1, D, H, W)
        cam = cam.squeeze(0)

        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy()

# ---------------- GUIDED BP + SMOOTH -----------------
class GuidedBackprop3D_Smooth:
    def __init__(self, model, std_spatial=0.1, n_samples=5):
        self.model = model
        self.model.eval()
        self.std_spatial = std_spatial
        self.n_samples = n_samples
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def relu_hook(module, grad_in, grad_out):
            return tuple(torch.clamp(g, min=0.0) for g in grad_in)
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_backward_hook(relu_hook))

    def generate(self, x, class_idx=0):
        """
        x: (1, 1, D, H, W) su DEVICE
        """
        x = x.detach()
        grads_acc = None

        with torch.no_grad():
            std = x.std()
        noise_std = self.std_spatial * std

        for _ in range(self.n_samples):
            noisy = x + torch.randn_like(x) * noise_std
            noisy.requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(noisy)

            if logits.dim() == 1 or logits.shape[-1] == 1:
                val = logits.view(-1)[0]
            else:
                val = logits[0, class_idx]

            val.backward(retain_graph=True)

            grad = noisy.grad.detach()  # (1,1,D,H,W)
            if grads_acc is None:
                grads_acc = grad
            else:
                grads_acc += grad

        grads_acc = grads_acc / float(self.n_samples)
        grads_acc = grads_acc.detach().cpu().squeeze(0)  # (1,D,H,W) -> (D,H,W)
        if grads_acc.shape[0] == 1:
            grads_acc = grads_acc[0]

        grads_acc = grads_acc - grads_acc.min()
        if grads_acc.max() > 0:
            grads_acc = grads_acc / grads_acc.max()

        return grads_acc.numpy()

    def close(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass

# ---------------- PREPROCESSING ----------------------
preproc = tio.Compose([tio.ZNormalization()])

def load_volume(path):
    subject = tio.Subject(img=tio.ScalarImage(path))
    return preproc(subject)['img'].data  # (1, D, H, W)

def upsample_heatmap(hm, target_shape, order=1):
    zoom_factors = [t / s for t, s in zip(target_shape, hm.shape)]
    hm_up = zoom(hm, zoom_factors, order=order)
    if hm_up.max() > 0:
        hm_up /= (hm_up.max() + 1e-8)
    return hm_up

def plot_orthogonal_planes(volume, heatmap, alpha=0.2, save_path=None, title=None):
    D, H, W = volume.shape

    # Slice centrali
    sag_idx = W // 2      # piano sagittale
    cor_idx = H // 2      # piano coronale
    ax_idx  = D // 2      # piano assiale

    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sagittal
    axes[0].imshow(np.rot90(vol_norm[:, :, sag_idx]), cmap='gray')
    axes[0].imshow(np.rot90(heatmap[:, :, sag_idx]), cmap='jet', alpha=alpha)
    axes[0].set_title('Sagittal')
    axes[0].axis('off')

    # Coronal
    axes[1].imshow(np.rot90(vol_norm[:, cor_idx, :]), cmap='gray')
    axes[1].imshow(np.rot90(heatmap[:, cor_idx, :]), cmap='jet', alpha=alpha)
    axes[1].set_title('Coronal')
    axes[1].axis('off')

    # Axial
    axes[2].imshow(np.rot90(vol_norm[ax_idx, :, :]), cmap='gray')
    axes[2].imshow(np.rot90(heatmap[ax_idx, :, :]), cmap='jet', alpha=alpha)
    axes[2].set_title('Axial')
    axes[2].axis('off')

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

# ---------------- CARICO UN MODELLO ------------------
best_id_file = os.path.join(results_dir, "optuna_best_trial_id.txt")
with open(best_id_file, "r") as f:
    best_trial_id = f.read().strip()

best_params_file = os.path.join(results_dir, "optuna_best_params.json")
with open(best_params_file, "r") as f:
    best_trial_params = json.load(f)["params"]

dropout = best_trial_params["dropout"]

# qui uso solo il fold 0 come "best model" per velocità
wpath = os.path.join(results_dir, f"best_model_fold_0_trial_{best_trial_id}.pth")
state = torch.load(wpath, map_location=DEVICE)
model = resnet18(sample_input_D=163, sample_input_H=193, sample_input_W=166,
                 num_seg_classes=1, dropout=dropout).to(DEVICE)
model.load_state_dict(state["state_dict"], strict=False)
model.eval()

print("Caricato modello singolo (fold 0).")

gradcam = GradCAM3D(model, target_layer_name=TARGET_LAYER)
gbp     = GuidedBackprop3D_Smooth(model, std_spatial=0.1, n_samples=5)

# ---------------- LOOP SUI VOLUMI --------------------
nii_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii*")))
print(f"Nella cartella immagini trovati {len(nii_files)} file.")

for path in nii_files:
    base = os.path.basename(path)
    subj_prefix = base.split(".")[0]

    if subj_prefix not in TEST_IDS:
        print(f"{subj_prefix}: non è nel set di test → skip.")
        continue

    label_true = TEST_LABELS[subj_prefix]

    hm_raw_path = os.path.join(OUT_DIR, f"{subj_prefix}_guided_gradcam.nii.gz")
    if os.path.exists(hm_raw_path):
        print(f"{subj_prefix}: Heatmap già esistente → skip.")
        continue

    try:
        print(f"{subj_prefix}: preprocessing & Guided Grad-CAM…")
        img_nii = nib.load(path)
        affine = img_nii.affine

        vol = load_volume(path).unsqueeze(0).to(DEVICE)   # (1,1,D,H,W)
        vol_np = vol.cpu().numpy()[0, 0]

        # probabilità del modello singolo (utile per log)
        with torch.no_grad():
            output = model(vol)
            prob = torch.sigmoid(output).item()
        pred_class = 1 if prob > 0.5 else 0
        print(f"{subj_prefix}: true={label_true}, pred={pred_class}, prob={prob:.3f}")

        # --------- Grad-CAM ---------
        cam_small = gradcam.generate(vol, class_idx=0)   # (D_cam,H_cam,W_cam)
        cam_up    = upsample_heatmap(cam_small, vol_np.shape)

        # --------- Guided BP + SmoothGrad ---------
        gbp_map = gbp.generate(vol, class_idx=0)         # (D,H,W)

        # --------- Guided Grad-CAM ---------
        guided_cam = cam_up * gbp_map
        guided_cam = guided_cam - guided_cam.min()
        if guided_cam.max() > 0:
            guided_cam = guided_cam / guided_cam.max()

        # --- Save NIfTI heatmap ---
        heatmap_nifti = nib.Nifti1Image(
            guided_cam.astype(np.float32),
            affine=affine
        )
        nib.save(heatmap_nifti, hm_raw_path)

        # --- Save visualization PNG ---
        out_path = os.path.join(OUT_DIR, f"{subj_prefix}_guided_gradcam.png")
        title = f"{subj_prefix} | true={label_true} | pred={pred_class} | prob={prob:.3f}"
        plot_orthogonal_planes(vol_np, guided_cam, alpha=ALPHA,
                               save_path=out_path, title=title)

        print(f"Saved Guided Grad-CAM per {subj_prefix}")

    except Exception as e:
        print(f"Errore con {subj_prefix}: {e}")

# -----------------------------------------------------
gbp.close()
