"""
inference.py — Inférence autonome (n'importe PAS TransformerInterraction)

Sélectionne N_GROUPS groupes aléatoires de 20 frames consécutives
parmi les données de validation, lance le générateur sur chaque groupe
et compare les prédictions au ground truth (ADE / FDE).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
import torchvision.transforms as T
from PIL import Image
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════
#  CONSTANTES  (identiques à TransformerInterraction.py)
# ══════════════════════════════════════════════════════
SEQ_PAST     = 8
SEQ_FUT      = 12
NOISE_DIM    = 64
D_MODEL      = 128
MAX_NEIGH    = 10
NEIGH_RADIUS = 5.0

N_GROUPS  = 5   # groupes de 20 frames a tirer aleatoirement
N_SAMPLES = 20  # trajectoires generees par pieton (Best-of-N)
SEED      = None  # None = aleatoire a chaque run | entier = reproductible (ex: 42)

SCENE_MAP = {
    "biwi_eth":      "eth",
    "biwi_hotel":    "hotel",
    "students001":   "university",
    "students003":   "university",
    "uni_examples":  "university",
    "crowds_zara01": "zara_01",
    "crowds_zara02": "zara_02",
    "crowds_zara03": "zara_02",
}

dataset_base = os.getenv("DATASET_BASE")
val_dir      = os.path.join(dataset_base, "raw", "val")

# ══════════════════════════════════════════════════════
#  DEVICE
# ══════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")


# ══════════════════════════════════════════════════════
#  ARCHITECTURE  (copie des classes de TransformerInterraction)
# ══════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, nb_head=8):
        super().__init__()
        self.nb_head  = nb_head
        self.head_dim = d_model // nb_head
        self.d_model  = d_model
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        Q = self.q_proj(query).view(B, Tq, self.nb_head, self.head_dim).transpose(1, 2)
        K = self.k_proj(key  ).view(B, Tk, self.nb_head, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Tk, self.nb_head, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + (1.0 - mask.unsqueeze(1)) * -1e9
        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.attention     = MultiHeadAttention(d_model)
        self.norm1         = nn.LayerNorm(d_model)
        self.feed_forward  = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm2         = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attention(x, x, x))
        x = self.norm2(x + self.feed_forward(x))
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, nb_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model) for _ in range(nb_layers)])

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SocialAttention(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.attention = MultiHeadAttention(d_model)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, ego, neighbors, mask):
        all_masked = (mask == 0).all(dim=-1, keepdim=True)
        mask = mask.clone()
        mask[all_masked.squeeze(-1)] = 1.0
        ego_query = ego.unsqueeze(1)
        ctx = self.attention(ego_query, neighbors, neighbors, mask=mask.unsqueeze(1))
        ctx = ctx.squeeze(1)
        ctx = ctx * (~all_masked.squeeze(-1)).float().unsqueeze(-1)
        return self.norm(ego + ctx)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_projection = nn.Linear(2048, D_MODEL)
        self.ego_encoder      = Encoder(input_dim=2, nb_layers=2)
        self.neighbor_encoder = Encoder(input_dim=2, nb_layers=1)
        self.social_attention = SocialAttention()
        self.prediction_head  = nn.Sequential(
            nn.Linear(D_MODEL + NOISE_DIM, 256), nn.ReLU(),
            nn.Linear(256, 128),                 nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, traj, neighbors, mask, scene_emb, z):
        scene_token = self.scene_projection(scene_emb).unsqueeze(1)
        ego_encoded = self.ego_encoder(traj)
        ego_h       = torch.cat([scene_token, ego_encoded], dim=1)[:, -1, :]
        B, K, T, _  = neighbors.shape
        neigh_flat  = neighbors.view(B * K, T, 2)
        neigh_h     = self.neighbor_encoder(neigh_flat)[:, -1, :].view(B, K, D_MODEL)
        neigh_h     = neigh_h * mask.unsqueeze(-1)
        social_h    = self.social_attention(ego_h, neigh_h, mask)
        h           = torch.cat([social_h, z], dim=-1)
        return self.prediction_head(h)


# ══════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════

def normalize(arr, mean, std):
    return (arr - mean) / std

def compute_ade_fde(y_true, y_pred):
    dist = np.linalg.norm(y_pred - y_true, axis=-1)  # (N, T)
    return dist.mean(), dist[:, -1].mean()

def best_of_n(all_samples, y_true):
    """all_samples : (n_samples, N, T, 2) — retourne (N, T, 2)"""
    ades     = np.linalg.norm(all_samples - y_true[None], axis=-1).mean(axis=2)
    best_idx = ades.argmin(axis=0)
    return all_samples[best_idx, np.arange(len(y_true))]

@torch.no_grad()
def rollout(G, traj, neighbors, mask, scene_emb, z):
    """Rollout autorégressif sur SEQ_FUT pas → (B, SEQ_FUT, 2)."""
    current = traj.clone()
    preds   = []
    for _ in range(SEQ_FUT):
        nxt = G(current, neighbors, mask, scene_emb, z)   # (B, 2)
        preds.append(nxt.unsqueeze(1))
        current = torch.cat([current[:, 1:, :], nxt.unsqueeze(1)], dim=1)
    return torch.cat(preds, dim=1)                        # (B, SEQ_FUT, 2)

@torch.no_grad()
def generate_predictions(G, X_n, N_n, M, EMB, n_samples=20, batch_size=256):
    """Retourne (n_samples, N, SEQ_FUT, 2) normalisé."""
    G.eval()
    all_samples = []
    for _ in range(n_samples):
        parts = []
        for s in range(0, len(X_n), batch_size):
            xb   = torch.tensor(X_n[s:s+batch_size], dtype=torch.float32).to(device)
            nb   = torch.tensor(N_n[s:s+batch_size], dtype=torch.float32).to(device)
            mb   = torch.tensor(M[s:s+batch_size],   dtype=torch.float32).to(device)
            eb   = torch.tensor(EMB[s:s+batch_size],  dtype=torch.float32).to(device)
            z    = torch.randn(len(xb), NOISE_DIM, device=device)
            pred = rollout(G, xb, nb, mb, eb, z).cpu().numpy()
            parts.append(pred)
        all_samples.append(np.concatenate(parts, axis=0))
    return np.stack(all_samples)   # (n_samples, N, SEQ_FUT, 2)


# ══════════════════════════════════════════════════════
#  RESNET50 — embeddings de scène
# ══════════════════════════════════════════════════════
print("\n=== Chargement ResNet50 ===")
_resnet = torchvision_models.resnet50(
    weights=torchvision_models.ResNet50_Weights.IMAGENET1K_V1)
_resnet = nn.Sequential(*list(_resnet.children())[:-1]).to(device)
_resnet.eval()

_preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_scene_embedding(frame_id: int, scene_dir: str) -> np.ndarray:
    path = os.path.join(scene_dir, f"frame{int(frame_id):06d}.jpg")
    if not os.path.exists(path):
        existing = [
            int(f[5:11]) for f in os.listdir(scene_dir)
            if f.startswith("frame") and f.endswith(".jpg")
        ]
        if not existing:
            return np.zeros(2048, dtype=np.float32)
        path = os.path.join(
            scene_dir,
            f"frame{min(existing, key=lambda x: abs(x - frame_id)):06d}.jpg")
    img    = Image.open(path).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = _resnet(tensor).squeeze().cpu().numpy()
    return emb.astype(np.float32)


# ══════════════════════════════════════════════════════
#  CHARGEMENT DU GÉNÉRATEUR ET DES NORMALISEURS
# ══════════════════════════════════════════════════════
print("\n=== Chargement du modèle ===")
G = Generator().to(device)
G.load_state_dict(torch.load("generator.pt", map_location=device))
G.eval()

mean_norm = np.load("normalizer_mean.npy")
std_norm  = np.load("normalizer_std.npy")
print(f"mean={mean_norm}  std={std_norm}")


# ══════════════════════════════════════════════════════
#  ÉTAPE 1 — Lister toutes les fenêtres valides
#            dans raw/val/
# ══════════════════════════════════════════════════════
def get_scene_key(fname: str) -> str:
    for suffix in ["_val.txt", "_train.txt"]:
        if fname.endswith(suffix):
            return fname[:-len(suffix)]
    return fname

def find_valid_windows(data: np.ndarray):
    """
    Retourne une liste de tableaux de SEQ_PAST+SEQ_FUT frame IDs consécutifs,
    chacun contenant au moins 1 piéton présent du début à la fin.
    """
    seq_len = SEQ_PAST + SEQ_FUT

    tracks = defaultdict(set)
    for frame, pid, x, y in data:
        tracks[int(pid)].add(int(frame))

    all_frames = sorted({int(r[0]) for r in data})
    if len(all_frames) < seq_len:
        return []

    dt = int(np.median(np.diff(all_frames)))

    windows = []
    for i in range(len(all_frames) - seq_len + 1):
        wf = all_frames[i : i + seq_len]
        # Fenêtre strictement consécutive ?
        if not np.all(np.abs(np.diff(wf) - dt) < 1e-3):
            continue
        # Au moins 1 piéton présent dans toutes les frames ?
        if any(all(f in tracks[pid] for f in wf) for pid in tracks):
            windows.append(np.array(wf, dtype=np.int32))
    return windows


print(f"\n=== Scan de {val_dir} ===")
all_windows = []

for fname in sorted(os.listdir(val_dir)):
    if not fname.endswith(".txt"):
        continue
    scene_key = get_scene_key(fname)
    if scene_key not in SCENE_MAP:
        print(f"  Ignoré : {fname}")
        continue

    scene_name = SCENE_MAP[scene_key]
    scene_dir  = os.path.join(dataset_base, scene_name, "visual_data")
    data       = np.loadtxt(os.path.join(val_dir, fname))
    windows    = find_valid_windows(data)
    print(f"  {fname:35s}  →  {len(windows):5d} fenêtres")

    # Toutes les frames d annotation de ce fichier (pour mapping proportionnel)
    annot_frames_all = np.unique(data[:, 0].astype(int))
    for wf in windows:
        all_windows.append({
            "scene_key":       scene_key,
            "scene_name":      scene_name,
            "scene_dir":       scene_dir,
            "fname":           fname,
            "data":            data,
            "window_frames":   wf,
            "annot_frames_all": annot_frames_all,
        })

print(f"\nTotal fenêtres disponibles : {len(all_windows)}")
assert len(all_windows) >= N_GROUPS, \
    f"Pas assez de fenêtres ({len(all_windows)}) pour {N_GROUPS} groupes."


# ══════════════════════════════════════════════════════
#  ÉTAPE 2 — Tirage aléatoire de N_GROUPS fenêtres
# ══════════════════════════════════════════════════════
rng    = np.random.default_rng(SEED)  # None => seed aleatoire
print(f"Seed utilise : {rng.bit_generator.state["state"]["state"]}")
chosen = rng.choice(len(all_windows), size=N_GROUPS, replace=False)
groups = [all_windows[i] for i in chosen]

print(f"\n=== {N_GROUPS} groupes sélectionnés ===")
for g_i, g in enumerate(groups):
    wf = g["window_frames"]
    print(f"  Groupe {g_i+1} | {g['fname']:35s} | frames {wf[0]:6d}–{wf[-1]:6d}")


# ══════════════════════════════════════════════════════
#  ÉTAPE 3 — Construire X, Y, N, M, EMB par groupe
# ══════════════════════════════════════════════════════
def build_group(group: dict) -> dict | None:
    data   = group["data"]
    wf     = group["window_frames"]
    obs_f  = wf[:SEQ_PAST]   # 8 frames d'observation
    fut_f  = wf[SEQ_PAST:]   # 12 frames futures

    tracks = defaultdict(dict)
    for frame, pid, x, y in data:
        tracks[int(pid)][int(frame)] = (float(x), float(y))

    valid_pids = [pid for pid in tracks
                  if all(f in tracks[pid] for f in wf)]
    if not valid_pids:
        return None

    X_l, Y_l, N_l, M_l = [], [], [], []

    for pid in valid_pids:
        obs_xy  = np.array([tracks[pid][f] for f in obs_f], dtype=np.float32)
        fut_xy  = np.array([tracks[pid][f] for f in fut_f], dtype=np.float32)
        ego_pos = obs_xy[-1]

        # Voisins dans NEIGH_RADIUS à la dernière frame d'obs
        candidates = []
        for other in valid_pids:
            if other == pid:
                continue
            other_last = np.array(tracks[other][obs_f[-1]])
            dist = np.linalg.norm(other_last - ego_pos)
            if dist <= NEIGH_RADIUS:
                traj = np.array([tracks[other][f] for f in obs_f], dtype=np.float32)
                candidates.append((dist, traj))

        candidates.sort(key=lambda c: c[0])
        candidates = candidates[:MAX_NEIGH]

        neigh = np.zeros((MAX_NEIGH, SEQ_PAST, 2), dtype=np.float32)
        mask  = np.zeros(MAX_NEIGH,               dtype=np.float32)
        for k, (_, traj) in enumerate(candidates):
            neigh[k] = traj
            mask[k]  = 1.0

        X_l.append(obs_xy)
        Y_l.append(fut_xy)
        N_l.append(neigh)
        M_l.append(mask)

    n_ped      = len(valid_pids)
    emb_single = get_scene_embedding(int(obs_f[-1]), group["scene_dir"])

    return {
        "X":          np.stack(X_l),
        "Y":          np.stack(Y_l),
        "N":          np.stack(N_l),
        "M":          np.stack(M_l),
        "EMB":        np.tile(emb_single, (n_ped, 1)),
        "pids":       valid_pids,
        "obs_frames": obs_f,
        "fut_frames": fut_f,
    }


# ══════════════════════════════════════════════════════
#  ÉTAPE 4 — Inférence par groupe
# ══════════════════════════════════════════════════════
print(f"\n=== Inférence (Best-of-{N_SAMPLES}) ===\n")
group_results = []

for g_i, group in enumerate(groups):
    wf = group["window_frames"]
    print(f"--- Groupe {g_i+1}/{N_GROUPS} | {group['fname']}  frames {wf[0]}–{wf[-1]} ---")

    gdata = build_group(group)
    if gdata is None:
        print("  ⚠ Aucun piéton valide, groupe ignoré.\n")
        continue

    n_ped = len(gdata["X"])
    print(f"  Piétons : {n_ped}")

    # Normalisation — X ET N !
    X_n = normalize(gdata["X"], mean_norm, std_norm)
    N_n = normalize(gdata["N"], mean_norm, std_norm)

    # Génération — (N_SAMPLES, n_ped, SEQ_FUT, 2)
    samples_n    = generate_predictions(G, X_n, N_n, gdata["M"], gdata["EMB"],
                                        n_samples=N_SAMPLES)
    samples_real = samples_n * std_norm + mean_norm   # dénormaliser

    Y_pred_bon  = best_of_n(samples_real, gdata["Y"])   # (n_ped, 12, 2)
    Y_pred_mean = samples_real.mean(axis=0)              # (n_ped, 12, 2)

    ade_bon,  fde_bon  = compute_ade_fde(gdata["Y"], Y_pred_bon)
    ade_mean, fde_mean = compute_ade_fde(gdata["Y"], Y_pred_mean)

    print(f"  Best-of-{N_SAMPLES} → ADE={ade_bon:.4f} m   FDE={fde_bon:.4f} m")
    print(f"  Mean-of-{N_SAMPLES} → ADE={ade_mean:.4f} m   FDE={fde_mean:.4f} m\n")

    group_results.append({
        "g_idx":              g_i + 1,
        "scene_name":         group["scene_name"],
        "scene_dir":          group["scene_dir"],
        "annot_frames_all":   group.get("annot_frames_all"),  # pour mapping image
        "fname":         group["fname"],
        "obs_frames":    gdata["obs_frames"],
        "fut_frames":    gdata["fut_frames"],
        "n_ped":         n_ped,
        "X":             gdata["X"],
        "Y":             gdata["Y"],
        "Y_pred_bon":    Y_pred_bon,
        "Y_pred_mean":   Y_pred_mean,
        "samples_real":  samples_real,
        "ade_bon":       ade_bon,  "fde_bon":      fde_bon,
        "ade_mean":      ade_mean, "fde_mean":     fde_mean,
        "pids":          gdata["pids"],
    })


# ══════════════════════════════════════════════════════
#  FIGURE 0 — Pour chaque groupe :
#    Ligne du haut  : frame 1  + trajectoires dessinées dessus
#    Ligne du bas   : frame 20 (dernière) sans aucun dessin
#                     → permet de voir où les gens sont arrivés
# ══════════════════════════════════════════════════════

def load_homography(scene_dir: str):
    """
    ETH/UCY : Homography.txt contient H qui mappe pixel->monde.
    On retourne H_inv = inv(H) pour aller monde->pixel.
    """
    parent = os.path.dirname(scene_dir)
    h_path = os.path.join(parent, "Homography.txt")
    if not os.path.exists(h_path):
        print(f"  Homography.txt introuvable : {h_path}")
        return None
    H = np.loadtxt(h_path)
    if H.shape != (3, 3):
        print(f"  Format inattendu ({H.shape}) pour {h_path}")
        return None
    H_inv = np.linalg.inv(H.astype(np.float64))
    print(f"  Homography chargee : {h_path}")
    return H_inv

def world_to_pixel(xy: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Coordonnees monde (x,y) -> pixel (col, row) via H_inv."""
    n   = len(xy)
    pts = np.hstack([xy, np.ones((n, 1))])
    p   = (H_inv @ pts.T).T
    return p[:, :2] / p[:, 2:3]

def get_frame_path(scene_dir: str, frame_id: int,
                   annot_frames_all: np.ndarray | None = None) -> str | None:
    """
    Retourne le chemin de la frame jpg correspondant a frame_id.

    Dans ETH/UCY, les fichiers image sont souvent numérotes
    sequentiellement (000001, 000002, ...) alors que les annotations
    utilisent les numeros de frames originaux du video (ex: 3790, 3800...).
    Si frame_id ne correspond a aucun fichier, on fait un mapping
    proportionnel : on cherche la position relative de frame_id dans
    la plage des annotations, puis on l applique aux images disponibles.

    annot_frames_all : tous les frame IDs presents dans le fichier .txt
                       (pour calculer la plage annotation)
    """
    # 1) Essai exact
    path = os.path.join(scene_dir, f"frame{int(frame_id):06d}.jpg")
    if os.path.exists(path):
        return path

    # 2) Lister les images disponibles
    existing = sorted([
        int(f[5:11]) for f in os.listdir(scene_dir)
        if f.startswith("frame") and f.endswith(".jpg")
    ])
    if not existing:
        return None

    # 3) Essai : frame_id proche d un numero image existant ?
    closest_abs = min(existing, key=lambda x: abs(x - int(frame_id)))
    gap = abs(closest_abs - int(frame_id))

    # Si l ecart est petit (<= 2x l intervalle images), c est probablement bon
    img_dt = existing[1] - existing[0] if len(existing) > 1 else 1
    if gap <= 2 * img_dt:
        return os.path.join(scene_dir, f"frame{closest_abs:06d}.jpg")

    # 4) Mapping proportionnel annot -> image
    #    Cas typique : annotations frame 3790..5000, images 000001..000612
    if annot_frames_all is not None and len(annot_frames_all) > 1:
        a_min = int(annot_frames_all.min())
        a_max = int(annot_frames_all.max())
        if a_max > a_min:
            ratio  = (int(frame_id) - a_min) / (a_max - a_min)
            ratio  = max(0.0, min(1.0, ratio))
            target = existing[0] + ratio * (existing[-1] - existing[0])
            mapped = min(existing, key=lambda x: abs(x - target))
            return os.path.join(scene_dir, f"frame{mapped:06d}.jpg")

    # 5) Fallback : le plus proche en absolu
    return os.path.join(scene_dir, f"frame{closest_abs:06d}.jpg")

def load_img(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    return np.array(Image.open(path).convert("RGB"))


# Couleurs distinctes par piéton
PED_COLORS = ["cyan", "yellow", "magenta", "lime",
              "orange", "white", "red", "deepskyblue"]

from matplotlib.lines import Line2D

n_groups = len(group_results)
# 2 lignes par groupe : haut=frame1+trajets, bas=frame20 seule
fig0, axes0 = plt.subplots(
    2, n_groups,
    figsize=(6 * n_groups, 11),
    gridspec_kw={"hspace": 0.04, "wspace": 0.05}
)
if n_groups == 1:
    axes0 = axes0[:, np.newaxis]   # (2, 1)

for col_i, res in enumerate(group_results):

    ax_top = axes0[0, col_i]   # frame 1  + trajectoires
    ax_bot = axes0[1, col_i]   # frame 20 seule

    scene_dir = res["scene_dir"]
    fid_first = int(res["obs_frames"][0])     # frame 1  du groupe
    fid_last  = int(res["fut_frames"][-1])    # frame 20 du groupe
    H_inv     = load_homography(scene_dir)
    af        = res.get("annot_frames_all")   # plage annotations pour mapping

    path_first = get_frame_path(scene_dir, fid_first, af)
    path_last  = get_frame_path(scene_dir, fid_last,  af)
    print(f"  Gr.{res['g_idx']} frame {fid_first} -> {os.path.basename(str(path_first))}"
          f"  |  frame {fid_last} -> {os.path.basename(str(path_last))}")

    img_first = load_img(path_first)
    img_last  = load_img(path_last)

    # ── Ligne du bas : frame 20 brute ───────────────────
    if img_last is not None:
        ax_bot.imshow(img_last)
        ax_bot.set_xlim(0, img_last.shape[1])
        ax_bot.set_ylim(img_last.shape[0], 0)
    else:
        ax_bot.set_facecolor("#1a1a2e")
    ax_bot.axis("off")
    ax_bot.set_title(
        f"Frame {fid_last}  (position finale réelle)",
        fontsize=8, color="white", pad=3)

    # ── Ligne du haut : frame 1 + trajectoires ──────────
    if img_first is not None:
        img_h, img_w = img_first.shape[:2]
        ax_top.imshow(img_first, zorder=0)
        ax_top.set_xlim(0, img_w)
        ax_top.set_ylim(img_h, 0)
    else:
        img_h, img_w = 576, 720
        ax_top.set_facecolor("#1a1a2e")

    n_show = min(res["n_ped"], len(PED_COLORS))

    for p_i in range(n_show):
        color   = PED_COLORS[p_i % len(PED_COLORS)]
        obs     = res["X"][p_i]            # (8,  2) monde
        gt      = res["Y"][p_i]            # (12, 2) monde
        pred    = res["Y_pred_bon"][p_i]   # (12, 2) monde
        full_gt = np.vstack([obs, gt])     # (20, 2)

        if H_inv is not None:
            full_gt_px = world_to_pixel(full_gt, H_inv)
            obs_px     = world_to_pixel(obs,     H_inv)
            pred_px    = world_to_pixel(pred,    H_inv)

            # Debug coordonnees du 1er pieton
            if p_i == 0:
                print(f"  [debug] Gr.{res['g_idx']} ped#{res['pids'][0]} "
                      f"obs_px[0]={obs_px[0].round(1)}  img=({img_w}x{img_h})")

            # Detecter si les axes sont echanges (row,col) vs (col,row)
            in_img = lambda px: ((-img_w < px[:,0]) & (px[:,0] < 2*img_w) &
                                 (-img_h < px[:,1]) & (px[:,1] < 2*img_h))
            if not in_img(full_gt_px).any():
                full_gt_px = world_to_pixel(full_gt[:, ::-1], H_inv)
                obs_px     = world_to_pixel(obs[:,  ::-1],    H_inv)
                pred_px    = world_to_pixel(pred[:, ::-1],    H_inv)
                if p_i == 0:
                    print(f"  [debug] axes echanges -> obs_px[0]={obs_px[0].round(1)}")

            # Trajectoire reelle — ligne pleine épaisse
            ax_top.plot(full_gt_px[:, 0], full_gt_px[:, 1],
                        "-", color=color, lw=2.5, zorder=3)
            ax_top.scatter(full_gt_px[0, 0], full_gt_px[0, 1],
                           color=color, s=70, marker="o", zorder=5)   # debut
            ax_top.scatter(obs_px[-1, 0], obs_px[-1, 1],
                           color=color, s=90, marker="x",
                           linewidths=2.5, zorder=5)                  # fin obs

            # Prediction — pointilles
            link = np.array([obs_px[-1], pred_px[0]])
            ax_top.plot(link[:, 0], link[:, 1],
                        "--", color=color, lw=1.5, alpha=0.5, zorder=4)
            ax_top.plot(pred_px[:, 0], pred_px[:, 1],
                        "--", color=color, lw=2.5, zorder=4)
            ax_top.scatter(pred_px[-1, 0], pred_px[-1, 1],
                           color=color, s=90, marker="^", zorder=5)   # fin predit
        else:
            # Pas d'homographie : coordonnees monde brutes
            ax_top.plot(full_gt[:, 0], full_gt[:, 1],
                        "-",  color=color, lw=2, zorder=3)
            ax_top.plot(pred[:, 0], pred[:, 1],
                        "--", color=color, lw=2, zorder=4)

    # Légende compacte
    legend_handles = [
        Line2D([0], [0], color="w", lw=2,  ls="-",  label="Réel (obs+futur)"),
        Line2D([0], [0], color="w", lw=2,  ls="--", label=f"Prédit Best-of-{N_SAMPLES}"),
        Line2D([0], [0], color="w", lw=0,  marker="o", ms=5,
               markerfacecolor="w", label="Début"),
        Line2D([0], [0], color="w", lw=0,  marker="x", ms=6,
               markerfacecolor="w", label="Fin obs."),
        Line2D([0], [0], color="w", lw=0,  marker="^", ms=6,
               markerfacecolor="w", label="Fin prédit"),
    ]
    ax_top.legend(handles=legend_handles, fontsize=6.5,
                  loc="lower right", framealpha=0.65,
                  labelcolor="white", facecolor="#00000099")

    ax_top.set_title(
        f"Groupe {res['g_idx']} — {res['scene_name']}  "
        f"({res['n_ped']} piétons)  ADE={res['ade_bon']:.2f} m\nFrame {fid_first}  (debut - trajectoires superposees)",
        fontsize=8, color="white", pad=4)
    ax_top.set_facecolor("black")
    ax_top.axis("off")

    ax_bot.set_facecolor("black")

# Étiquettes latérales communes
fig0.text(0.01, 0.73, "Frame 1\n+ trajectoires", va="center", ha="left",
          fontsize=9, color="white", rotation=90,
          fontweight="bold")
fig0.text(0.01, 0.27, "Frame 20\n(sans dessin)", va="center", ha="left",
          fontsize=9, color="white", rotation=90,
          fontweight="bold")

fig0.patch.set_facecolor("#111111")
fig0.suptitle(
    f"Vue scene par groupe  |  Haut: frame 1 + trajets  |  Bas: frame 20 (arrivee reelle)\n"
    f"Trajectoire reelle (-)  vs  Prediction Best-of-{N_SAMPLES} (--)",
    fontsize=11, fontweight="bold", color="white", y=1.01)
plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig("inference_scene_overlay.png", dpi=150, bbox_inches="tight",
            facecolor=fig0.get_facecolor())
print("Figure 0 sauvegardee : inference_scene_overlay.png")
plt.show()

# ══════════════════════════════════════════════════════
#  FIGURE 1 — Trajectoires (lignes=groupes, col=piétons)
# ══════════════════════════════════════════════════════
MAX_PED = 4
n_rows  = len(group_results)
fig, axes = plt.subplots(n_rows, MAX_PED, figsize=(5 * MAX_PED, 4.5 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

for row, res in enumerate(group_results):
    n_show = min(res["n_ped"], MAX_PED)
    for col in range(MAX_PED):
        ax = axes[row, col]
        if col >= n_show:
            ax.axis("off")
            continue

        obs   = res["X"][col]
        gt    = res["Y"][col]
        pred  = res["Y_pred_bon"][col]
        trajs = res["samples_real"][:, col]   # (N_SAMPLES, 12, 2)

        # Toutes les trajectoires générées en fond gris
        for s in range(N_SAMPLES):
            ax.plot(trajs[s, :, 0], trajs[s, :, 1],
                    color="gray", alpha=0.12, linewidth=0.7, zorder=1)

        ax.plot(obs[:, 0],  obs[:, 1],  "o-",  color="steelblue", lw=2, ms=4,
                label="Observé",            zorder=3)
        ax.plot(gt[:, 0],   gt[:, 1],   "s--", color="green",     lw=2, ms=4,
                label="Ground truth",       zorder=4)
        ax.plot(pred[:, 0], pred[:, 1], "^-",  color="red",       lw=2, ms=4,
                label=f"Best-of-{N_SAMPLES}", zorder=5)

        # Jonction obs → prédiction
        ax.plot([obs[-1, 0], pred[0, 0]], [obs[-1, 1], pred[0, 1]],
                "--", color="red", alpha=0.35, lw=1, zorder=2)
        ax.scatter(obs[-1, 0], obs[-1, 1], color="black", s=55, marker="x", zorder=6)

        dist    = np.linalg.norm(pred - gt, axis=-1)
        ade_loc = dist.mean()
        fde_loc = dist[-1]

        ax.set_title(
            f"Ped #{res['pids'][col]}\nADE={ade_loc:.2f} m  FDE={fde_loc:.2f} m",
            fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.set_aspect("equal")
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_xlabel("x (m)", fontsize=7)

        if col == 0:
            ax.set_ylabel(
                f"Gr.{res['g_idx']} – {res['scene_name']}\n"
                f"frames {res['obs_frames'][0]}–{res['fut_frames'][-1]}\ny (m)",
                fontsize=8)

plt.suptitle(
    f"Inférence — {N_GROUPS} groupes de {SEQ_PAST+SEQ_FUT} frames "
    f"(Best-of-{N_SAMPLES})",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("inference_groups.png", dpi=150)
print("Figure 1 sauvegardée : inference_groups.png")
plt.show()


# ══════════════════════════════════════════════════════
#  FIGURE 2 — Comparaison ADE/FDE par groupe
# ══════════════════════════════════════════════════════
labels    = [f"Grp {r['g_idx']}\n{r['scene_name']}" for r in group_results]
ade_bons  = [r["ade_bon"]  for r in group_results]
fde_bons  = [r["fde_bon"]  for r in group_results]
ade_means = [r["ade_mean"] for r in group_results]
fde_means = [r["fde_mean"] for r in group_results]
x         = np.arange(len(group_results))
w         = 0.2

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.bar(x - w, ade_bons,  w*2, label=f"Best-of-{N_SAMPLES}", color="tomato",    alpha=0.85)
ax1.bar(x + w, ade_means, w*2, label=f"Mean-of-{N_SAMPLES}", color="steelblue",  alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("ADE (m)"); ax1.set_title("ADE par groupe")
ax1.axhline(np.mean(ade_bons),  color="tomato",    ls="--", lw=1.2,
            label=f"Moy. Best = {np.mean(ade_bons):.3f} m")
ax1.axhline(np.mean(ade_means), color="steelblue", ls=":",  lw=1.2,
            label=f"Moy. Mean = {np.mean(ade_means):.3f} m")
ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.4)

ax2.bar(x - w, fde_bons,  w*2, label=f"Best-of-{N_SAMPLES}", color="tomato",    alpha=0.85)
ax2.bar(x + w, fde_means, w*2, label=f"Mean-of-{N_SAMPLES}", color="steelblue",  alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("FDE (m)"); ax2.set_title("FDE par groupe")
ax2.axhline(np.mean(fde_bons),  color="tomato",    ls="--", lw=1.2,
            label=f"Moy. Best = {np.mean(fde_bons):.3f} m")
ax2.axhline(np.mean(fde_means), color="steelblue", ls=":",  lw=1.2,
            label=f"Moy. Mean = {np.mean(fde_means):.3f} m")
ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.4)

plt.suptitle(f"Comparaison ADE/FDE — {N_GROUPS} groupes × {N_SAMPLES} samples",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("inference_comparison.png", dpi=150)
print("Figure 2 sauvegardée : inference_comparison.png")
plt.show()


# ══════════════════════════════════════════════════════
#  RÉSUMÉ CONSOLE
# ══════════════════════════════════════════════════════
print("\n" + "="*70)
print("                      RÉSUMÉ FINAL")
print("="*70)
print(f"{'Gr.':<4} {'Scène':<13} {'Frames':<14} {'Pied.':<6}"
      f"{'ADE Best':>9} {'FDE Best':>9} {'ADE Mean':>9} {'FDE Mean':>9}")
print("-"*70)
for r in group_results:
    fr = f"{r['obs_frames'][0]}–{r['fut_frames'][-1]}"
    print(f"{r['g_idx']:<4} {r['scene_name']:<13} {fr:<14} {r['n_ped']:<6}"
          f"{r['ade_bon']:>9.4f} {r['fde_bon']:>9.4f}"
          f"{r['ade_mean']:>9.4f} {r['fde_mean']:>9.4f}")
if group_results:
    print("-"*70)
    print(f"{'Moy.':<4} {'':<13} {'':<14} {'':<6}"
          f"{np.mean(ade_bons):>9.4f} {np.mean(fde_bons):>9.4f}"
          f"{np.mean(ade_means):>9.4f} {np.mean(fde_means):>9.4f}")
print("="*70)