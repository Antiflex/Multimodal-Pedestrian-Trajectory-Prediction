import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as torchvision_models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict

SEQ_PAST  = 8
SEQ_FUT   = 12
NOISE_DIM = 64
D_MODEL   = 128

BASE_DIR      = os.getenv("DATASET_BASE")
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "raw", "train")
RAW_VAL_DIR   = os.path.join(BASE_DIR, "raw", "val")

SCENE_MAP = {
    "biwi_eth":      "eth",
    "biwi_hotel":    "hotel",
    "students001":   "university",
    "uni_examples":  "university",
    "crowds_zara01": "zara_01",
    "crowds_zara02": "zara_02",
}

IMG_H, IMG_W = 224, 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU disponible :", torch.cuda.get_device_name(0))
else:
    print("CPU uniquement")


# ─────────────────────────────────────────────
# ResNet50 PyTorch — précalcul embeddings
# ─────────────────────────────────────────────
resnet_pt = torchvision_models.resnet50(weights=torchvision_models.ResNet50_Weights.IMAGENET1K_V1)
resnet_pt = nn.Sequential(*list(resnet_pt.children())[:-1])
resnet_pt = resnet_pt.to(device)
resnet_pt.eval()

preprocess_pt = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_img_pytorch(frame_id: int, scene_dir: str) -> torch.Tensor:
    path = os.path.join(scene_dir, f"frame{frame_id:06d}.jpg")
    img  = Image.open(path).convert("RGB")
    return preprocess_pt(img).unsqueeze(0)

def precompute_embeddings_scene(frame_ids, scene_dir, batch_size=64):
    unique_ids   = np.unique(frame_ids)
    existing_ids = [f for f in unique_ids
                    if os.path.exists(os.path.join(scene_dir, f"frame{int(f):06d}.jpg"))]
    missing = len(unique_ids) - len(existing_ids)
    if missing > 0:
        print(f"    ⚠ {missing} frames manquantes ignorées")
    frame_to_emb = {}
    with torch.no_grad():
        for start in range(0, len(existing_ids), batch_size):
            batch_ids = existing_ids[start:start+batch_size]
            imgs = torch.cat([load_img_pytorch(int(f), scene_dir)
                              for f in batch_ids], dim=0).to(device)
            embs = resnet_pt(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
            for fid, emb in zip(batch_ids, embs):
                frame_to_emb[int(fid)] = emb
    return frame_to_emb

def precompute_all_embeddings(F, S):
    global_emb = {}
    scene_available = {}
    for scene_dir in np.unique(S):
        mask      = S == scene_dir
        frame_ids = F[mask]
        scene_name = os.path.basename(os.path.dirname(scene_dir))
        print(f"  {scene_name} : {len(np.unique(frame_ids))} frames uniques")
        emb_dict = precompute_embeddings_scene(frame_ids, scene_dir)
        for fid, emb in emb_dict.items():
            global_emb[(scene_dir, fid)] = emb
        scene_available[scene_dir] = sorted(emb_dict.keys())

    def get_emb(sd, fid):
        key = (sd, fid)
        if key in global_emb:
            return global_emb[key]
        closest = min(scene_available[sd], key=lambda x: abs(x - fid))
        return global_emb[(sd, closest)]

    return np.stack([get_emb(S[i], int(F[i])) for i in range(len(F))]).astype(np.float32)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def get_scene_key(txt_filename):
    base = os.path.basename(txt_filename)
    for suffix in ["_train.txt", "_val.txt"]:
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base

def build_multistep_gt_with_frame(data, seq_past=8, seq_fut=12, stride=1):
    """Retourne X (8 obs), Y (12 futurs), F (frame id)."""
    tracks = defaultdict(list)
    for frame, pid, x, y in data:
        tracks[int(pid)].append((int(frame), float(x), float(y)))
    X_list, Y_list, F_list = [], [], []
    for pid, pts in tracks.items():
        pts.sort(key=lambda t: t[0])
        frames = np.array([p[0] for p in pts], dtype=np.int32)
        xy     = np.array([[p[1], p[2]] for p in pts], dtype=np.float32)
        dt     = np.median(np.diff(frames)) if len(frames) > 1 else None
        for i in range(0, len(xy) - (seq_past + seq_fut) + 1, stride):
            f_window = frames[i:i+seq_past+seq_fut]
            if dt is not None:
                diffs = np.diff(f_window.astype(np.float32))
                if np.any(np.abs(diffs - dt) > 1e-3):
                    continue
            X_list.append(xy[i:i+seq_past])
            Y_list.append(xy[i+seq_past:i+seq_past+seq_fut])
            F_list.append(frames[i+seq_past-1])
    return (np.stack(X_list).astype(np.float32),
            np.stack(Y_list).astype(np.float32),
            np.array(F_list, dtype=np.int32))

def load_all_scenes(raw_dir):
    X_all, Y_all, F_all, S_all = [], [], [], []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".txt"):
            continue
        scene_key = get_scene_key(fname)
        if scene_key not in SCENE_MAP:
            print(f"  Ignoré : {fname}")
            continue
        scene_dir = os.path.join(BASE_DIR, SCENE_MAP[scene_key], "visual_data")
        print(f"  Chargement {fname} → {SCENE_MAP[scene_key]}")
        data = np.loadtxt(os.path.join(raw_dir, fname))
        # On charge directement les 12 pas futurs pour le discriminateur
        X, Y, F = build_multistep_gt_with_frame(data, SEQ_PAST, SEQ_FUT, stride=1)
        X_all.append(X); Y_all.append(Y); F_all.append(F)
        S_all.extend([scene_dir] * len(X))
    return (np.concatenate(X_all), np.concatenate(Y_all),
            np.concatenate(F_all), np.array(S_all))

def fit_normalizer(X, Y):
    all_xy = np.concatenate([X.reshape(-1,2), Y.reshape(-1,2)], axis=0)
    mean   = all_xy.mean(axis=0).astype(np.float32)
    std    = (all_xy.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std

def apply_normalizer(X, Y, mean, std):
    return (X-mean)/std, (Y-mean)/std


# ─────────────────────────────────────────────
# Dataset PyTorch
# Y est maintenant (B, 12, 2) — 12 pas futurs
# ─────────────────────────────────────────────
class TrajGANDataset(Dataset):
    def __init__(self, X, Y, EMB):
        self.X   = torch.tensor(X,   dtype=torch.float32)
        self.Y   = torch.tensor(Y,   dtype=torch.float32)
        self.EMB = torch.tensor(EMB, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.EMB[idx]


# ─────────────────────────────────────────────
# Transformer blocks
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=128, nb_head=8):
        super().__init__()
        self.nb_head  = nb_head
        self.head_dim = dim // nb_head
        self.dim      = dim
        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, mask=None):
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        Q = self.q_proj(query).view(B, Tq, self.nb_head, self.head_dim).transpose(1,2)
        K = self.k_proj(key).view(B, Tk, self.nb_head, self.head_dim).transpose(1,2)
        V = self.v_proj(value).view(B, Tk, self.nb_head, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1,2).contiguous().view(B, Tq, self.dim)
        return self.out_proj(out)

class EncoderLayer(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.attn  = MultiHeadAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ff(x))
        return x

class Encoder(nn.Module):
    def __init__(self, nb_layers=2, dim=128):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim) for _ in range(nb_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ─────────────────────────────────────────────
# Générateur — next-step autorégressif
# Identique à avant : prédit 1 pas à la fois
# ─────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, seq_past=8, d_model=128, noise_dim=64):
        super().__init__()
        self.scene_proj = nn.Linear(2048, d_model)
        self.traj_proj  = nn.Linear(2, d_model)
        self.encoder    = Encoder(nb_layers=2, dim=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, traj, emb, z):
        """Prédit le prochain pas (B,2)."""
        scene_tok = self.scene_proj(emb).unsqueeze(1)        # (B,1,128)
        traj_emb  = self.traj_proj(traj)                     # (B,8,128)
        enc_in    = torch.cat([scene_tok, traj_emb], dim=1)  # (B,9,128)
        enc_out   = self.encoder(enc_in)
        h         = enc_out[:, -1, :]                        # (B,128)
        h         = torch.cat([h, z], dim=-1)                # (B,192)
        return self.head(h)                                  # (B,2)

def rollout_generator(generator, traj_obs, emb, z, seq_fut=12):
    """
    Rollout autorégressif sur seq_fut pas.
    Le bruit z est fixé pour toute la séquence (cohérence de la trajectoire).
    traj_obs : (B, 8, 2)
    emb      : (B, 2048)
    z        : (B, noise_dim)
    Retourne : (B, seq_fut, 2)
    """
    X_cur = traj_obs.clone()
    preds = []
    for _ in range(seq_fut):
        y_next = generator(X_cur, emb, z)           # (B,2)
        preds.append(y_next.unsqueeze(1))            # (B,1,2)
        X_cur = torch.cat([X_cur[:, 1:, :],
                           y_next.unsqueeze(1)], dim=1)  # (B,8,2) sliding window
    return torch.cat(preds, dim=1)                   # (B,12,2)


# ─────────────────────────────────────────────
# Critique WGAN-GP — juge la trajectoire complète (12 pas)
# ─────────────────────────────────────────────
class Critique(nn.Module):
    def __init__(self, seq_past=8, seq_fut=12, d_model=128):
        super().__init__()
        self.scene_proj = nn.Linear(2048, d_model)
        self.traj_proj  = nn.Linear(2, d_model)
        # Encode trajectoire complète : 1 token scène + 8 obs + 12 futurs = 21 tokens
        self.encoder    = Encoder(nb_layers=3, dim=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
            # Pas de sigmoid — WGAN utilise des scores non bornés
        )

    def forward(self, traj_obs, traj_fut, emb):
        """
        traj_obs : (B, 8,  2)
        traj_fut : (B, 12, 2)
        emb      : (B, 2048)
        """
        traj_full = torch.cat([traj_obs, traj_fut], dim=1)  # (B, 20, 2)
        scene_tok = self.scene_proj(emb).unsqueeze(1)        # (B, 1,  128)
        traj_emb  = self.traj_proj(traj_full)                # (B, 20, 128)
        enc_in    = torch.cat([scene_tok, traj_emb], dim=1)  # (B, 21, 128)
        enc_out   = self.encoder(enc_in)
        h         = enc_out.mean(dim=1)                      # (B, 128)
        return self.head(h)                                  # (B, 1) — score non borné


# ─────────────────────────────────────────────
# Gradient Penalty (WGAN-GP)
# ─────────────────────────────────────────────
def gradient_penalty(critique, traj_obs, y_real, y_fake, emb, lambda_gp=10.0):
    """
    Calcule la gradient penalty sur des trajectoires interpolées.
    y_real, y_fake : (B, 12, 2)
    """
    B = y_real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)               # (B,1,1) broadcast
    y_interp = (alpha * y_real + (1 - alpha) * y_fake).requires_grad_(True)

    score_interp = critique(traj_obs, y_interp, emb)

    grads = torch.autograd.grad(
        outputs=score_interp,
        inputs=y_interp,
        grad_outputs=torch.ones_like(score_interp),
        create_graph=True,
        retain_graph=True,
    )[0]  # (B, 12, 2)

    grads_norm = grads.reshape(B, -1).norm(2, dim=1)          # (B,)
    gp = lambda_gp * ((grads_norm - 1) ** 2).mean()
    return gp


# ─────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────
def compute_ade_fde(y_true, y_pred):
    dist = np.linalg.norm(y_pred - y_true, axis=-1)
    return dist.mean(), dist[:, -1].mean()


# ─────────────────────────────────────────────
# Rollout pour évaluation (numpy) — Best-of-N
# ─────────────────────────────────────────────
@torch.no_grad()
def rollout_eval(generator, X0, EMB0, seq_fut=12, noise_dim=64, n_samples=1, batch_size=256):
    """
    Génère n_samples trajectoires et retourne la meilleure par sample
    selon l'ADE (Best-of-N). Retourne (N, seq_fut, 2).
    Si n_samples=1, retourne directement la trajectoire.
    """
    generator.eval()
    all_preds = []
    for _ in range(n_samples):
        results = []
        for s in range(0, len(X0), batch_size):
            xb = torch.tensor(X0[s:s+batch_size],  dtype=torch.float32).to(device)
            eb = torch.tensor(EMB0[s:s+batch_size], dtype=torch.float32).to(device)
            zb = torch.randn(len(xb), noise_dim, device=device)
            yb = rollout_generator(generator, xb, eb, zb, seq_fut).cpu().numpy()
            results.append(yb)
        all_preds.append(np.concatenate(results, axis=0))  # (N, 12, 2)
    return np.stack(all_preds, axis=0)  # (n_samples, N, 12, 2)

def best_of_n(all_preds, y_true):
    """
    all_preds : (n_samples, N, 12, 2)
    y_true    : (N, 12, 2)
    Retourne  : (N, 12, 2) — la trajectoire la plus proche du ground truth
    """
    # ADE par sample et par trajectoire : (n_samples, N)
    ades = np.linalg.norm(all_preds - y_true[None], axis=-1).mean(axis=2)
    best_idx = ades.argmin(axis=0)                              # (N,)
    return all_preds[best_idx, np.arange(len(y_true))]         # (N, 12, 2)

def mean_of_n(all_preds):
    """Moyenne des n_samples trajectoires — (N, 12, 2)."""
    return all_preds.mean(axis=0)


# ─────────────────────────────────────────────
# Training steps WGAN-GP
# ─────────────────────────────────────────────
N_CRITIC = 5  # steps critique pour 1 step générateur

def train_step_critique(generator, critique, opt_c, traj, y_real_12, emb, noise_dim):
    """
    y_real_12 : (B, 12, 2) — trajectoire future réelle normalisée
    Entraîne le critique N_CRITIC fois.
    """
    critique.train(); generator.eval()
    total_loss = 0.0

    for _ in range(N_CRITIC):
        z = torch.randn(traj.size(0), noise_dim, device=device)

        with torch.no_grad():
            y_fake_12 = rollout_generator(generator, traj, emb, z, SEQ_FUT)  # (B,12,2)

        score_real = critique(traj, y_real_12, emb)   # (B,1)
        score_fake = critique(traj, y_fake_12, emb)   # (B,1)

        # Wasserstein loss
        loss_w = -score_real.mean() + score_fake.mean()

        # Gradient penalty
        gp = gradient_penalty(critique, traj, y_real_12, y_fake_12, emb)

        loss_c = loss_w + gp

        opt_c.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(critique.parameters(), 1.0)
        opt_c.step()
        total_loss += loss_c.item()

    return total_loss / N_CRITIC

def train_step_generator(generator, critique, opt_g, traj, y_real_12, emb, noise_dim,
                         lambda_l2=1.0):
    """Un step générateur après N_CRITIC steps critique."""
    generator.train(); critique.eval()

    z = torch.randn(traj.size(0), noise_dim, device=device)
    y_fake_12 = rollout_generator(generator, traj, emb, z, SEQ_FUT)  # (B,12,2)

    score_fake = critique(traj, y_fake_12, emb)

    # WGAN : maximiser le score → minimiser -score
    loss_adv = -score_fake.mean()

    # Régularisation L2 sur toute la trajectoire
    loss_l2 = torch.mean((y_fake_12 - y_real_12) ** 2)

    loss_g = loss_adv + lambda_l2 * loss_l2

    opt_g.zero_grad()
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
    opt_g.step()

    return loss_g.item(), loss_adv.item(), loss_l2.item()


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════
print("\n=== Chargement train ===")
X_train, Y_train, F_train, S_train = load_all_scenes(RAW_TRAIN_DIR)
print("\n=== Chargement val ===")
X_val, Y_val, F_val, S_val = load_all_scenes(RAW_VAL_DIR)
print(f"\nTrain: {X_train.shape} | Y_train: {Y_train.shape}")
print(f"Val:   {X_val.shape}   | Y_val:   {Y_val.shape}")

print("\n=== Précalcul embeddings train ===")
EMB_train = precompute_all_embeddings(F_train, S_train)
print("\n=== Précalcul embeddings val ===")
EMB_val   = precompute_all_embeddings(F_val, S_val)

# Normalisation
mean, std = fit_normalizer(X_train, Y_train)
X_train_n, Y_train_n = apply_normalizer(X_train, Y_train, mean, std)
X_val_n,   Y_val_n   = apply_normalizer(X_val,   Y_val,   mean, std)

# Dataset / DataLoader
train_ds     = TrajGANDataset(X_train_n, Y_train_n, EMB_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

# Modèles
generator = Generator(SEQ_PAST, D_MODEL, NOISE_DIM).to(device)
critique  = Critique(SEQ_PAST, SEQ_FUT, D_MODEL).to(device)

print(f"\nGénérateur : {sum(p.numel() for p in generator.parameters()):,} params")
print(f"Critique   : {sum(p.numel() for p in critique.parameters()):,} params")

opt_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_c = optim.Adam(critique.parameters(),  lr=1e-4, betas=(0.0, 0.9))

# ── Boucle d'entraînement ─────────────────────
EPOCHS       = 200
PATIENCE     = 20
best_ade     = float('inf')
patience_cnt = 0
best_g_state = None

history = {"loss_g": [], "loss_c": [], "loss_adv": [], "loss_l2": [],
           "val_ADE": [], "val_FDE": []}

print("\n=== Entraînement WGAN-GP ===")
for epoch in range(1, EPOCHS + 1):
    epoch_lc, epoch_lg, epoch_ladv, epoch_ll2 = [], [], [], []

    for traj_b, y_b, emb_b in train_loader:
        traj_b = traj_b.to(device)   # (B, 8, 2)
        y_b    = y_b.to(device)      # (B, 12, 2)
        emb_b  = emb_b.to(device)    # (B, 2048)

        # N_CRITIC steps critique, puis 1 step générateur
        lc = train_step_critique(generator, critique, opt_c,
                                 traj_b, y_b, emb_b, NOISE_DIM)
        lg, ladv, ll2 = train_step_generator(generator, critique, opt_g,
                                             traj_b, y_b, emb_b, NOISE_DIM)

        epoch_lc.append(lc)
        epoch_lg.append(lg)
        epoch_ladv.append(ladv)
        epoch_ll2.append(ll2)

    mean_lc   = np.mean(epoch_lc)
    mean_lg   = np.mean(epoch_lg)
    mean_ladv = np.mean(epoch_ladv)
    mean_ll2  = np.mean(epoch_ll2)

    # ADE / FDE sur val — Best-of-3 pendant le training
    all_preds_n    = rollout_eval(generator, X_val_n, EMB_val, SEQ_FUT, NOISE_DIM, n_samples=3)
    all_preds_real = all_preds_n * std + mean   # (3, N, 12, 2)
    Y_pred_real    = best_of_n(all_preds_real, Y_val)
    ade, fde       = compute_ade_fde(Y_val, Y_pred_real)

    history["loss_c"].append(mean_lc)
    history["loss_g"].append(mean_lg)
    history["loss_adv"].append(mean_ladv)
    history["loss_l2"].append(mean_ll2)
    history["val_ADE"].append(ade)
    history["val_FDE"].append(fde)

    print(f"Epoch {epoch:3d} | loss_C={mean_lc:.4f} | loss_G={mean_lg:.4f} "
          f"| adv={mean_ladv:.4f} | L2={mean_ll2:.4f} "
          f"| ADE={ade:.4f} | FDE={fde:.4f}")

    if ade < best_ade:
        best_ade     = ade
        best_g_state = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping à l'époque {epoch} (best ADE={best_ade:.4f})")
            break

# Restaurer meilleurs poids
generator.load_state_dict(best_g_state)
generator.to(device)

# Évaluation finale — Best-of-20
print("\n=== Évaluation finale (Best-of-20) ===")
all_preds_n    = rollout_eval(generator, X_val_n, EMB_val, SEQ_FUT, NOISE_DIM, n_samples=20)
all_preds_real = all_preds_n * std + mean  # (20, N, 12, 2)

# Best-of-20 : la meilleure trajectoire parmi 20
Y_pred_bon  = best_of_n(all_preds_real, Y_val)
ade_bon, fde_bon = compute_ade_fde(Y_val, Y_pred_bon)
print(f"[WGAN-GP - Best-of-20]  ADE={ade_bon:.4f} FDE={fde_bon:.4f}")

# Moyenne-de-20 : pour comparaison
Y_pred_mean = mean_of_n(all_preds_real)
ade_mean, fde_mean = compute_ade_fde(Y_val, Y_pred_mean)
print(f"[WGAN-GP - Mean-of-20]  ADE={ade_mean:.4f} FDE={fde_mean:.4f}")

torch.save(generator.state_dict(), "trajectory_wgan_generator.pt")
torch.save(critique.state_dict(),  "trajectory_wgan_critique.pt")
print("Modèles sauvegardés.")

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["loss_c"],   label="loss_C (critique)")
ax1.plot(history["loss_g"],   label="loss_G (générateur)")
ax1.plot(history["loss_adv"], label="loss_adv")
ax1.plot(history["loss_l2"],  label="loss_L2")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("WGAN-GP - Losses"); ax1.legend()

ax2.plot(history["val_ADE"], label="val_ADE", color="orange")
ax2.plot(history["val_FDE"], label="val_FDE", color="green")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mètres")
ax2.set_title("WGAN-GP - ADE / FDE"); ax2.legend()

plt.tight_layout()
plt.savefig("wgan_training.png", dpi=150)
plt.show()