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

SEQ_PAST     = 8
SEQ_FUT      = 12
NOISE_DIM    = 64
D_MODEL      = 128
MAX_NEIGH    = 10
NEIGH_RADIUS = 5.0

BASE_DIR = os.getenv("DATASET_BASE")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU :", torch.cuda.get_device_name(0))


# RESNET50 pour les embeddings de scène

# On charge ResNet50 pré-entrainé et on enlève la dernière couche
# pour récupérer un vecteur de 2048 dimensions par image
resnet = torchvision_models.resnet50(weights=torchvision_models.ResNet50_Weights.IMAGENET1K_V1)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

# Preprocessing standard ImageNet
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_embedding(frame_id, scene_dir):
    path = os.path.join(scene_dir, f"frame{frame_id:06d}.jpg")
    img  = Image.open(path).convert("RGB")
    img  = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(img).squeeze()
    return emb.cpu().numpy()

def precompute_embeddings(frame_ids, scene_dir):
    # On calcule les embeddings pour toutes les frames d'une scène
    unique_frames = np.unique(frame_ids)
    embeddings = {}
    batch = []
    batch_ids = []

    for fid in unique_frames:
        path = os.path.join(scene_dir, f"frame{int(fid):06d}.jpg")
        if not os.path.exists(path):
            continue
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        batch.append(img)
        batch_ids.append(int(fid))

        # On traite par batch de 64
        if len(batch) == 64:
            imgs = torch.cat(batch).to(device)
            with torch.no_grad():
                embs = resnet(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
            for fid2, emb in zip(batch_ids, embs):
                embeddings[fid2] = emb
            batch = []
            batch_ids = []

    # Dernier batch
    if batch:
        imgs = torch.cat(batch).to(device)
        with torch.no_grad():
            embs = resnet(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
        for fid2, emb in zip(batch_ids, embs):
            embeddings[fid2] = emb

    return embeddings

def get_all_embeddings(frame_ids, scene_dirs):
    # Pour chaque scène, on calcule les embeddings et on les stocke
    cache = {}
    available_frames = {}

    for scene_dir in np.unique(scene_dirs):
        mask = scene_dirs == scene_dir
        print(f"  Calcul embeddings pour {os.path.basename(os.path.dirname(scene_dir))}...")
        emb_dict = precompute_embeddings(frame_ids[mask], scene_dir)
        for fid, emb in emb_dict.items():
            cache[(scene_dir, fid)] = emb
        available_frames[scene_dir] = sorted(emb_dict.keys())

    # Pour chaque sample, on récupère l'embedding correspondant
    result = []
    for i in range(len(frame_ids)):
        sd  = scene_dirs[i]
        fid = int(frame_ids[i])
        if (sd, fid) in cache:
            result.append(cache[(sd, fid)])
        else:
            # Si la frame n'existe pas, on prend la plus proche
            closest = min(available_frames[sd], key=lambda x: abs(x - fid))
            result.append(cache[(sd, closest)])

    return np.stack(result).astype(np.float32)


# CHARGEMENT DES DONNÉES

def load_all_scenes(raw_dir):
    X_list, Y_list, N_list, M_list, F_list, S_list = [], [], [], [], [], []

    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".txt"):
            continue

        # Récupérer le nom de la scène
        scene_key = filename.replace("_train.txt", "").replace("_val.txt", "")
        if scene_key not in SCENE_MAP:
            print(f"  Ignoré : {filename}")
            continue

        scene_dir = os.path.join(BASE_DIR, SCENE_MAP[scene_key], "visual_data")
        print(f"  Chargement {filename}...")

        data = np.loadtxt(os.path.join(raw_dir, filename))
        X, Y, N, M, F = build_sequences(data)

        X_list.append(X)
        Y_list.append(Y)
        N_list.append(N)
        M_list.append(M)
        F_list.append(F)
        S_list.extend([scene_dir] * len(X))

    return (np.concatenate(X_list), np.concatenate(Y_list),
            np.concatenate(N_list), np.concatenate(M_list),
            np.concatenate(F_list), np.array(S_list))

def build_sequences(data):
    # On regroupe les positions par piéton
    tracks = defaultdict(list)
    for frame, pid, x, y in data:
        tracks[int(pid)].append((int(frame), float(x), float(y)))

    X_list, Y_list, N_list, M_list, F_list = [], [], [], [], []

    for pid in tracks:
        pts = sorted(tracks[pid])  # trier par frame
        frames = np.array([p[0] for p in pts], dtype=np.int32)
        xy     = np.array([[p[1], p[2]] for p in pts], dtype=np.float32)

        # Intervalle de temps médian (pour détecter les sauts)
        dt = np.median(np.diff(frames)) if len(frames) > 1 else None

        for i in range(len(xy) - SEQ_PAST - SEQ_FUT + 1):
            # Vérifier qu'il n'y a pas de saut temporel
            window = frames[i : i + SEQ_PAST + SEQ_FUT]
            if dt is not None:
                if np.any(np.abs(np.diff(window.astype(float)) - dt) > 1e-3):
                    continue

            obs    = xy[i : i + SEQ_PAST]        # 8 positions observées
            future = xy[i + SEQ_PAST : i + SEQ_PAST + SEQ_FUT]  # 12 positions futures
            obs_frames = frames[i : i + SEQ_PAST]

            # Trouver les voisins proches
            neighbors, mask = find_neighbors(pid, tracks, obs_frames, obs[-1])

            X_list.append(obs)
            Y_list.append(future)
            N_list.append(neighbors)
            M_list.append(mask)
            F_list.append(obs_frames[-1])

    return (np.stack(X_list).astype(np.float32),
            np.stack(Y_list).astype(np.float32),
            np.stack(N_list).astype(np.float32),
            np.stack(M_list).astype(np.float32),
            np.array(F_list, dtype=np.int32))

def find_neighbors(ego_pid, tracks, obs_frames, ego_pos):
    # On cherche les piétons dans le rayon de 5m
    candidates = []

    for other_pid in tracks:
        if other_pid == ego_pid:
            continue

        # Construire un dictionnaire frame -> position pour ce piéton
        pos_dict = {p[0]: (p[1], p[2]) for p in tracks[other_pid]}

        # Le voisin doit être présent dans toutes les frames observées
        if not all(f in pos_dict for f in obs_frames):
            continue

        # Calculer la distance à la dernière frame
        ox, oy = pos_dict[obs_frames[-1]]
        distance = np.sqrt((ox - ego_pos[0])**2 + (oy - ego_pos[1])**2)
        if distance > NEIGH_RADIUS:
            continue

        # Extraire la trajectoire du voisin
        traj = np.array([[pos_dict[f][0], pos_dict[f][1]] for f in obs_frames],
                        dtype=np.float32)
        candidates.append((distance, traj))

    # Trier par distance et garder les MAX_NEIGH plus proches
    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:MAX_NEIGH]

    # Créer les tableaux avec padding
    neighbors = np.zeros((MAX_NEIGH, SEQ_PAST, 2), dtype=np.float32)
    mask      = np.zeros(MAX_NEIGH, dtype=np.float32)

    for k, (dist, traj) in enumerate(candidates):
        neighbors[k] = traj
        mask[k]      = 1.0

    return neighbors, mask


# NORMALISATION

def fit_normalizer(X, Y):
    all_coords = np.concatenate([X.reshape(-1, 2), Y.reshape(-1, 2)], axis=0)
    mean = all_coords.mean(axis=0).astype(np.float32)
    std  = (all_coords.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std

def normalize(arr, mean, std):
    return (arr - mean) / std


# DATASET PYTORCH

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y, N, M, EMB):
        self.X   = torch.tensor(X,   dtype=torch.float32)
        self.Y   = torch.tensor(Y,   dtype=torch.float32)
        self.N   = torch.tensor(N,   dtype=torch.float32)
        self.M   = torch.tensor(M,   dtype=torch.float32)
        self.EMB = torch.tensor(EMB, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.N[idx], self.M[idx], self.EMB[idx]


# ARCHITECTURE TRANSFORMER

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

        # Projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape en têtes d'attention
        Q = Q.view(B, Tq, self.nb_head, self.head_dim).transpose(1, 2)
        K = K.view(B, Tk, self.nb_head, self.head_dim).transpose(1, 2)
        V = V.view(B, Tk, self.nb_head, self.head_dim).transpose(1, 2)

        # Scores d'attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Appliquer le masque si besoin
        if mask is not None:
            scores = scores + (1.0 - mask.unsqueeze(1)) * -1e9

        attn = torch.softmax(scores, dim=-1)

        # Agréger les valeurs
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.attention  = MultiHeadAttention(d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention + résiduel
        x = self.norm1(x + self.attention(x, x, x))
        # Feed-forward + résiduel
        x = self.norm2(x + self.feed_forward(x))
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, nb_layers=2):
        super().__init__()
        # Projeter l'entrée vers d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        # Empiler les couches encoder
        self.layers = nn.ModuleList([EncoderLayer(d_model) for _ in range(nb_layers)])

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        return x  # (B, T, d_model)


# ATTENTION SOCIALE

class SocialAttention(nn.Module):
    """
    Le piéton courant (ego) fait attention à ses voisins.
    Cela permet au modèle d'apprendre à éviter les collisions,
    suivre un groupe, etc.
    """
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.attention = MultiHeadAttention(d_model)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, ego, neighbors, mask):
        # ego       : (B, d_model)
        # neighbors : (B, K, d_model)
        # mask      : (B, K) — 1 = voisin valide, 0 = padding

        # Si tous les voisins sont masqués, on évite les NaN
        all_masked = (mask == 0).all(dim=-1, keepdim=True)
        mask = mask.clone()
        mask[all_masked.squeeze(-1)] = 1.0  # on "dé-masque" temporairement

        # Le piéton ego (query) fait attention aux voisins (key, value)
        ego_query = ego.unsqueeze(1)  # (B, 1, d_model)
        ctx = self.attention(ego_query, neighbors, neighbors, mask=mask.unsqueeze(1))
        ctx = ctx.squeeze(1)  # (B, d_model)

        # On annule le contexte si le piéton n'avait pas de voisins
        ctx = ctx * (~all_masked.squeeze(-1)).float().unsqueeze(-1)

        return self.norm(ego + ctx)


# GÉNÉRATEUR

class Generator(nn.Module):
    """
    Prédit le prochain pas (x, y) à partir de :
    - la trajectoire observée (8 pas)
    - les trajectoires des voisins
    - l'embedding de la scène (ResNet)
    - un vecteur de bruit z (pour la multimodalité)
    """
    def __init__(self):
        super().__init__()
        # Encoder la scène
        self.scene_projection = nn.Linear(2048, D_MODEL)

        # Encoder la trajectoire du piéton courant
        self.ego_encoder = Encoder(input_dim=2, nb_layers=2)

        # Encoder les trajectoires des voisins (même encodeur partagé)
        self.neighbor_encoder = Encoder(input_dim=2, nb_layers=1)

        # Attention sociale
        self.social_attention = SocialAttention()

        # Prédire le prochain pas
        self.prediction_head = nn.Sequential(
            nn.Linear(D_MODEL + NOISE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, traj, neighbors, mask, scene_emb, z):
        # 1. Encoder la trajectoire ego + scène
        scene_token = self.scene_projection(scene_emb).unsqueeze(1)  # (B, 1, d)
        ego_encoded = self.ego_encoder(traj)                          # (B, 8, d)
        ego_h       = torch.cat([scene_token, ego_encoded], dim=1)[:, -1, :]  # (B, d)

        # 2. Encoder les voisins en parallèle
        B, K, T, _ = neighbors.shape
        neigh_flat  = neighbors.view(B * K, T, 2)
        neigh_h     = self.neighbor_encoder(neigh_flat)[:, -1, :]  # (B*K, d)
        neigh_h     = neigh_h.view(B, K, D_MODEL)
        neigh_h     = neigh_h * mask.unsqueeze(-1)  # mettre à zéro les voisins paddés

        # 3. Attention sociale
        social_h = self.social_attention(ego_h, neigh_h, mask)  # (B, d)

        # 4. Prédire le prochain pas avec le bruit z
        h = torch.cat([social_h, z], dim=-1)  # (B, d + noise_dim)
        return self.prediction_head(h)         # (B, 2)


# CRITIQUE (WGAN)

class Critic(nn.Module):
    """
    Évalue si une trajectoire complète (8 obs + 12 futurs) est réaliste.
    Retourne un score non borné (pas de sigmoid — c'est WGAN).
    """
    def __init__(self):
        super().__init__()
        self.scene_projection = nn.Linear(2048, D_MODEL)
        self.trajectory_encoder = Encoder(input_dim=2, nb_layers=3)
        self.score_head = nn.Sequential(
            nn.Linear(D_MODEL, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, traj_obs, traj_fut, scene_emb):
        # Concatener trajectoire observée + future
        full_traj = torch.cat([traj_obs, traj_fut], dim=1)  # (B, 20, 2)

        # Encoder + token scène
        scene_token = self.scene_projection(scene_emb).unsqueeze(1)
        traj_encoded = self.trajectory_encoder(full_traj)
        h = torch.cat([scene_token, traj_encoded], dim=1).mean(dim=1)

        return self.score_head(h)  # (B, 1) — score non borné


# WGAN-GP

def compute_gradient_penalty(critic, traj_obs, y_real, y_fake, scene_emb):
    """
    Pénalité de gradient pour stabiliser l'entraînement WGAN.
    On interpole entre une trajectoire réelle et une fausse,
    et on punit si le gradient de la critique s'éloigne de 1.
    """
    B     = y_real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)

    # Trajectoire interpolée
    y_interp = (alpha * y_real + (1 - alpha) * y_fake).requires_grad_(True)

    score = critic(traj_obs, y_interp, scene_emb)

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=y_interp,
        grad_outputs=torch.ones(B, 1, device=device),
        create_graph=True
    )[0]

    gradient_norm = gradients.reshape(B, -1).norm(2, dim=1)
    penalty = 10.0 * ((gradient_norm - 1) ** 2).mean()
    return penalty

def rollout(generator, traj, neighbors, mask, scene_emb, z):
    """
    Génère une trajectoire de SEQ_FUT pas en mode autorégressif.
    À chaque pas, la prédiction devient la nouvelle entrée.
    """
    current_traj = traj.clone()
    predictions  = []

    for step in range(SEQ_FUT):
        next_pos = generator(current_traj, neighbors, mask, scene_emb, z)  # (B, 2)
        predictions.append(next_pos.unsqueeze(1))

        # Sliding window : on enlève le premier pas et on ajoute la prédiction
        current_traj = torch.cat([current_traj[:, 1:, :], next_pos.unsqueeze(1)], dim=1)

    return torch.cat(predictions, dim=1)  # (B, 12, 2)


# ÉVALUATION

def compute_ade_fde(y_true, y_pred):
    distances = np.linalg.norm(y_pred - y_true, axis=-1)  # (N, T)
    ade = distances.mean()
    fde = distances[:, -1].mean()
    return ade, fde

@torch.no_grad()
def generate_predictions(generator, X, N, M, EMB, n_samples=3, batch_size=256):
    """Génère n_samples trajectoires pour chaque piéton."""
    generator.eval()
    all_samples = []

    for sample_idx in range(n_samples):
        sample_preds = []
        for start in range(0, len(X), batch_size):
            xb   = torch.tensor(X[start:start+batch_size]).to(device)
            nb   = torch.tensor(N[start:start+batch_size]).to(device)
            mb   = torch.tensor(M[start:start+batch_size]).to(device)
            emb  = torch.tensor(EMB[start:start+batch_size]).to(device)
            z    = torch.randn(len(xb), NOISE_DIM, device=device)
            pred = rollout(generator, xb, nb, mb, emb, z).cpu().numpy()
            sample_preds.append(pred)
        all_samples.append(np.concatenate(sample_preds, axis=0))

    return np.stack(all_samples)  # (n_samples, N, 12, 2)

def best_of_n(all_samples, y_true):
    """Pour chaque piéton, garder la trajectoire la plus proche du ground truth."""
    ades     = np.linalg.norm(all_samples - y_true[None], axis=-1).mean(axis=2)
    best_idx = ades.argmin(axis=0)
    return all_samples[best_idx, np.arange(len(y_true))]


# MAIN

print("\n=== Chargement train ===")
X_train, Y_train, N_train, M_train, F_train, S_train = load_all_scenes(RAW_TRAIN_DIR)

print("\n=== Chargement val ===")
X_val, Y_val, N_val, M_val, F_val, S_val = load_all_scenes(RAW_VAL_DIR)

print(f"\nTrain : {X_train.shape} | Voisins moyens : {M_train.sum(1).mean():.1f}")

print("\n=== Calcul embeddings scène ===")
EMB_train = get_all_embeddings(F_train, S_train)
EMB_val   = get_all_embeddings(F_val,   S_val)

mean, std = fit_normalizer(X_train, Y_train)

X_train_n = normalize(X_train, mean, std)
Y_train_n = normalize(Y_train, mean, std)
N_train_n = normalize(N_train, mean, std)
X_val_n   = normalize(X_val,   mean, std)
Y_val_n   = normalize(Y_val,   mean, std)
N_val_n   = normalize(N_val,   mean, std)

train_dataset = TrajectoryDataset(X_train_n, Y_train_n, N_train_n, M_train, EMB_train)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

G = Generator().to(device)
C = Critic().to(device)

print(f"\nGénérateur : {sum(p.numel() for p in G.parameters()):,} paramètres")
print(f"Critique   : {sum(p.numel() for p in C.parameters()):,} paramètres")

optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
optimizer_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

EPOCHS    = 200
PATIENCE  = 20
N_CRITIC  = 5
LAMBDA_L2 = 2.0

best_ade   = float("inf")
patience   = 0
best_state = None
history    = {"loss_C": [], "loss_G": [], "ADE": [], "FDE": []}

print("\n=== Entraînement ===")
for epoch in range(1, EPOCHS + 1):

    losses_C, losses_G = [], []

    for traj_b, y_b, neigh_b, mask_b, emb_b in train_loader:
        traj_b  = traj_b.to(device)
        y_b     = y_b.to(device)
        neigh_b = neigh_b.to(device)
        mask_b  = mask_b.to(device)
        emb_b   = emb_b.to(device)

        # Entraîner le Critique N_CRITIC fois
        C.train(); G.eval()
        for _ in range(N_CRITIC):
            z = torch.randn(len(traj_b), NOISE_DIM, device=device)

            with torch.no_grad():
                y_fake = rollout(G, traj_b, neigh_b, mask_b, emb_b, z)

            # Wasserstein loss + gradient penalty
            loss_real = C(traj_b, y_b, emb_b).mean()
            loss_fake = C(traj_b, y_fake, emb_b).mean()
            gp        = compute_gradient_penalty(C, traj_b, y_b, y_fake, emb_b)
            loss_C    = -loss_real + loss_fake + gp

            optimizer_C.zero_grad()
            loss_C.backward()
            nn.utils.clip_grad_norm_(C.parameters(), 1.0)
            optimizer_C.step()

        # Entraîner le Générateur
        G.train(); C.eval()
        z = torch.randn(len(traj_b), NOISE_DIM, device=device)

        y_fake   = rollout(G, traj_b, neigh_b, mask_b, emb_b, z)
        loss_adv = -C(traj_b, y_fake, emb_b).mean()         # tromper la critique
        loss_l2  = torch.mean((y_fake - y_b) ** 2)          # rester proche du réel
        loss_G   = loss_adv + LAMBDA_L2 * loss_l2

        optimizer_G.zero_grad()
        loss_G.backward()
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        optimizer_G.step()

        losses_C.append(loss_C.item())
        losses_G.append(loss_G.item())

    # Évaluation sur la validation
    samples    = generate_predictions(G, X_val_n, N_val_n, M_val, EMB_val, n_samples=3)
    preds_real = best_of_n(samples * std + mean, Y_val)
    ade, fde   = compute_ade_fde(Y_val, preds_real)

    history["loss_C"].append(np.mean(losses_C))
    history["loss_G"].append(np.mean(losses_G))
    history["ADE"].append(ade)
    history["FDE"].append(fde)

    print(f"Epoch {epoch:3d} | loss_C={np.mean(losses_C):.4f} | loss_G={np.mean(losses_G):.4f}"
          f" | ADE={ade:.4f} | FDE={fde:.4f}")

    # Early stopping
    if ade < best_ade:
        best_ade   = ade
        best_state = {k: v.cpu().clone() for k, v in G.state_dict().items()}
        patience   = 0
    else:
        patience += 1
        if patience >= PATIENCE:
            print(f"Early stopping à l'époque {epoch} (meilleur ADE = {best_ade:.4f})")
            break

G.load_state_dict(best_state)
G.to(device)

samples    = generate_predictions(G, X_val_n, N_val_n, M_val, EMB_val, n_samples=20)
real_preds = samples * std + mean

ade_best, fde_best = compute_ade_fde(Y_val, best_of_n(real_preds, Y_val))
ade_mean, fde_mean = compute_ade_fde(Y_val, real_preds.mean(axis=0))

print(f"\n[Best-of-20]  ADE = {ade_best:.4f}  FDE = {fde_best:.4f}")
print(f"[Mean-of-20]  ADE = {ade_mean:.4f}  FDE = {fde_mean:.4f}")

torch.save(G.state_dict(), "generator.pt")
np.save("normalizer_mean.npy", mean)
np.save("normalizer_std.npy", std)
torch.save(C.state_dict(), "critic.pt")
print("Modèles sauvegardés.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["loss_C"], label="Critique")
ax1.plot(history["loss_G"], label="Générateur")
ax1.set_title("Losses")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(history["ADE"], label="ADE", color="orange")
ax2.plot(history["FDE"], label="FDE", color="green")
ax2.set_title("ADE / FDE sur la validation")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mètres")
ax2.legend()

plt.tight_layout()
plt.savefig("training.png", dpi=150)
plt.show()