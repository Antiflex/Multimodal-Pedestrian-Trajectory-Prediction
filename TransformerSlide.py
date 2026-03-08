import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
import torchvision.transforms as T
from PIL import Image
import os

from collections import defaultdict

SEQ_PAST = 8
SEQ_FUT  = 12

BASE_DIR      = r"C:\Users\xordp\Desktop\EFREI\IA\ING3\TP_Attention\data_trajpred-20260304T083725Z-3-001\data_trajpred"
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "raw", "train")
RAW_VAL_DIR   = os.path.join(BASE_DIR, "raw", "val")

# Mapping nom de fichier txt (sans _train/_val.txt) → dossier scène
SCENE_MAP = {
    "biwi_eth":      "eth",
    "biwi_hotel":    "hotel",
    "students001":   "university",
    "crowds_zara01": "zara_01",
    "crowds_zara02": "zara_02",
    # crowds_zara03 ignoré
}

IMG_H, IMG_W = 224, 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU disponible :", torch.cuda.get_device_name(0))
else:
    print("Attention : GPU non disponible, CPU utilisé")


# ─────────────────────────────────────────────
# ResNet50 PyTorch
# ─────────────────────────────────────────────
resnet_pt = torchvision_models.resnet50(weights=torchvision_models.ResNet50_Weights.IMAGENET1K_V1)
resnet_pt = nn.Sequential(*list(resnet_pt.children())[:-1])
resnet_pt = resnet_pt.to(device)
resnet_pt.eval()

preprocess_pt = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def load_img_pytorch(frame_id: int, scene_dir: str) -> torch.Tensor:
    fname = f"frame{frame_id:06d}.jpg"
    path  = os.path.join(scene_dir, fname)
    img   = Image.open(path).convert("RGB")
    return preprocess_pt(img).unsqueeze(0)

def precompute_embeddings_scene(frame_ids: np.ndarray, scene_dir: str, batch_size: int = 64) -> dict:
    unique_ids   = np.unique(frame_ids)
    frame_to_emb = {}
    with torch.no_grad():
        for start in range(0, len(unique_ids), batch_size):
            batch_ids = unique_ids[start:start+batch_size]
            imgs = torch.cat([load_img_pytorch(int(f), scene_dir) for f in batch_ids], dim=0).to(device)
            embs = resnet_pt(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
            for fid, emb in zip(batch_ids, embs):
                frame_to_emb[int(fid)] = emb
    return frame_to_emb


# ─────────────────────────────────────────────
# Chargement multi-scènes
# ─────────────────────────────────────────────
def get_scene_key(txt_filename: str) -> str:
    base = os.path.basename(txt_filename)
    for suffix in ["_train.txt", "_val.txt"]:
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base

def load_all_scenes(raw_dir: str):
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
        X, Y, F = build_nextstep_sequences_with_frame(data, seq_past=SEQ_PAST, stride=1)
        X_all.append(X)
        Y_all.append(Y)
        F_all.append(F)
        S_all.extend([scene_dir] * len(X))
    return (np.concatenate(X_all), np.concatenate(Y_all),
            np.concatenate(F_all), np.array(S_all))

def load_all_scenes_multistep(raw_dir: str):
    X_all, Y_all, F_all, S_all = [], [], [], []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".txt"):
            continue
        scene_key = get_scene_key(fname)
        if scene_key not in SCENE_MAP:
            continue
        scene_dir = os.path.join(BASE_DIR, SCENE_MAP[scene_key], "visual_data")
        data = np.loadtxt(os.path.join(raw_dir, fname))
        X, Y, F = build_multistep_gt_with_frame(data, SEQ_PAST, SEQ_FUT, stride=1)
        X_all.append(X)
        Y_all.append(Y)
        F_all.append(F)
        S_all.extend([scene_dir] * len(X))
    return (np.concatenate(X_all), np.concatenate(Y_all),
            np.concatenate(F_all), np.array(S_all))

def precompute_all_embeddings(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    global_emb = {}
    for scene_dir in np.unique(S):
        mask      = S == scene_dir
        frame_ids = F[mask]
        print(f"  Scène: '{scene_dir}'")
        print(f"  nb samples: {mask.sum()} | frames max: {frame_ids.max()}")
        emb_dict  = precompute_embeddings_scene(frame_ids, scene_dir)
        for fid, emb in emb_dict.items():
            global_emb[(scene_dir, fid)] = emb
    return np.stack([global_emb[(S[i], int(F[i]))] for i in range(len(F))]).astype(np.float32)

# ─────────────────────────────────────────────
# Utilitaires trajectoires
# ─────────────────────────────────────────────
def compute_ade_fde(y_true, y_pred):
    dist = np.linalg.norm(y_pred - y_true, axis=-1)
    return dist.mean(), dist[:, -1].mean()

def build_nextstep_sequences_with_frame(data, seq_past=8, stride=1):
    tracks = defaultdict(list)
    for frame, pid, x, y in data:
        tracks[int(pid)].append((int(frame), float(x), float(y)))
    X_list, Y_list, F_list = [], [], []
    for pid, pts in tracks.items():
        pts.sort(key=lambda t: t[0])
        frames = np.array([p[0] for p in pts], dtype=np.int32)
        xy     = np.array([[p[1], p[2]] for p in pts], dtype=np.float32)
        dt     = np.median(np.diff(frames)) if len(frames) > 1 else None
        for i in range(0, len(xy) - (seq_past + 1) + 1, stride):
            f_window = frames[i:i+seq_past+1]
            if dt is not None:
                diffs = np.diff(f_window.astype(np.float32))
                if np.any(np.abs(diffs - dt) > 1e-3):
                    continue
            X_list.append(xy[i:i+seq_past])
            Y_list.append(xy[i+seq_past])
            F_list.append(frames[i+seq_past-1])
    return (np.stack(X_list).astype(np.float32),
            np.stack(Y_list).astype(np.float32),
            np.array(F_list, dtype=np.int32))

def build_multistep_gt_with_frame(data, seq_past=8, seq_fut=12, stride=1):
    tracks = defaultdict(list)
    for frame, pid, x, y in data:
        tracks[int(pid)].append((int(frame), float(x), float(y)))
    X_list, Y_list, F_list = [], [], []
    for pid, pts in tracks.items():
        pts.sort(key=lambda t: t[0])
        frames = np.array([p[0] for p in pts], dtype=np.int32)
        xy     = np.array([[p[1], p[2]] for p in pts], dtype=np.float32)
        for i in range(0, len(xy) - (seq_past + seq_fut) + 1, stride):
            X_list.append(xy[i:i+seq_past])
            Y_list.append(xy[i+seq_past:i+seq_past+seq_fut])
            F_list.append(frames[i+seq_past-1])
    return (np.stack(X_list).astype(np.float32),
            np.stack(Y_list).astype(np.float32),
            np.array(F_list, dtype=np.int32))

def fit_normalizer(X, Y):
    all_xy = np.concatenate([X.reshape(-1,2), Y.reshape(-1,2)], axis=0)
    mean   = all_xy.mean(axis=0)
    std    = all_xy.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def apply_normalizer(X, Y, mean, std):
    return (X-mean)/std, (Y-mean)/std


# ─────────────────────────────────────────────
# Dataset TF
# ─────────────────────────────────────────────
def make_scene_nextstep_dataset_emb(X, Y, EMB, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y, EMB))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y, e: ({"traj": x, "emb": e}, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def rollout_nextstep_scene(model, X0, EMB0, seq_fut=12):
    X_cur = X0.copy()
    preds = []
    for _ in range(seq_fut):
        y_next = model.predict({"traj": X_cur, "emb": EMB0}, verbose=0)
        preds.append(y_next[:, None, :])
        X_cur = np.concatenate([X_cur[:, 1:, :], y_next[:, None, :]], axis=1)
    return np.concatenate(preds, axis=1)


# ─────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim=128, nb_head=8, **kwargs):
        self.dim      = dim
        self.head_dim = dim // nb_head
        self.nb_head  = nb_head
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.query_layer = tf.keras.layers.Dense(128)
        self.value_layer = tf.keras.layers.Dense(128)
        self.key_layer   = tf.keras.layers.Dense(128)
        self.out_proj    = tf.keras.layers.Dense(128)
        super().build(input_shape)

    def mask_softmax(self, x, mask):
        x_expe        = tf.math.exp(x)
        x_expe_masked = x_expe * mask
        x_expe_sum    = tf.reduce_sum(x_expe_masked, axis=-1, keepdims=True)
        return x_expe_masked / x_expe_sum

    def call(self, x, mask=None):
        in_query, in_key, in_value = x
        if isinstance(mask, (tuple, list)):
            mask = None
        Q = self.query_layer(in_query)
        K = self.key_layer(in_key)
        V = self.value_layer(in_value)
        batch_size = tf.shape(Q)[0]
        Q_seq_len  = tf.shape(Q)[1]
        K_seq_len  = tf.shape(K)[1]
        V_seq_len  = tf.shape(V)[1]
        Q = tf.reshape(Q, [batch_size, Q_seq_len, self.nb_head, self.head_dim])
        K = tf.reshape(K, [batch_size, K_seq_len, self.nb_head, self.head_dim])
        V = tf.reshape(V, [batch_size, V_seq_len, self.nb_head, self.head_dim])
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])
        Q = tf.reshape(Q, [batch_size * self.nb_head, Q_seq_len, self.head_dim])
        K = tf.reshape(K, [batch_size * self.nb_head, K_seq_len, self.head_dim])
        V = tf.reshape(V, [batch_size * self.nb_head, V_seq_len, self.head_dim])
        QK = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        if mask is not None:
            softmax_QK = self.mask_softmax(QK * mask, mask)
        else:
            softmax_QK = tf.nn.softmax(QK, axis=-1)
        attention = tf.matmul(softmax_QK, V)
        attention = tf.reshape(attention, [batch_size, self.nb_head, Q_seq_len, self.head_dim])
        attention = tf.transpose(attention, [0, 2, 1, 3])
        attention = tf.reshape(attention, [batch_size, Q_seq_len, self.nb_head * self.head_dim])
        return self.out_proj(attention)


# ─────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.multi_head_attention = MultiHeadAttention()
        self.norm      = tf.keras.layers.LayerNormalization()
        self.dense_out = tf.keras.layers.Dense(128)
        super().build(input_shape)

    def call(self, x, mask=None):
        attention = self.multi_head_attention((x, x, x), mask=None)
        post_att  = self.norm(attention + x)
        ff        = self.dense_out(post_att)
        return self.norm(ff + post_att)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, nb_encoder, **kwargs):
        self.nb_encoder = nb_encoder
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.encoder_layers = [EncoderLayer() for _ in range(self.nb_encoder)]
        super().build(input_shape)

    def call(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.multi_head_self_attention = MultiHeadAttention()
        self.multi_head_enc_attention  = MultiHeadAttention()
        self.norm        = tf.keras.layers.LayerNormalization()
        self.proj_output = tf.keras.layers.Dense(128)
        super().build(input_shape)

    def call(self, x):
        enc_output, output_emb, mask = x
        self_att      = self.multi_head_self_attention((output_emb, output_emb, output_emb), mask=mask)
        post_self_att = self.norm(output_emb + self_att)
        enc_att       = self.multi_head_enc_attention((post_self_att, enc_output, enc_output), mask=None)
        post_enc_att  = self.norm(enc_att + post_self_att)
        proj          = self.proj_output(post_enc_att)
        return self.norm(proj + post_enc_att)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, nb_decoder, **kwargs):
        self.nb_decoder = nb_decoder
        super(**kwargs).__init__()

    def build(self, input_shape):
        self.decoder_layers = [DecoderLayer() for _ in range(self.nb_decoder)]
        super().build(input_shape)

    def call(self, x):
        enc_output, output_emb, mask = x
        dec_output = output_emb
        for layer in self.decoder_layers:
            dec_output = layer((enc_output, dec_output, mask))
        return dec_output


# ─────────────────────────────────────────────
# Callback ADE/FDE
# ─────────────────────────────────────────────
class ADE_FDE_Scene_Callback(tf.keras.callbacks.Callback):
    def __init__(self, Xv8_real, Yv12_real, EMB_val, mean, std, seq_fut=12, max_samples=512):
        super().__init__()
        self.Xv8_real    = Xv8_real
        self.Yv12_real   = Yv12_real
        self.EMB_val     = EMB_val
        self.mean, self.std = mean, std
        self.seq_fut     = seq_fut
        self.max_samples = max_samples

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        n    = min(len(self.Xv8_real), self.max_samples)
        Xn   = (self.Xv8_real[:n] - self.mean) / self.std
        emb0 = self.EMB_val[:n]
        Y_pred_n = rollout_nextstep_scene(self.model, Xn, emb0, self.seq_fut)
        Y_pred   = Y_pred_n * self.std + self.mean
        ade, fde = compute_ade_fde(self.Yv12_real[:n], Y_pred)
        logs["val_ADE"] = float(ade)
        logs["val_FDE"] = float(fde)
        print(f" — val_ADE: {ade:.4f} — val_FDE: {fde:.4f}", end="")


# ─────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────
def get_scene_aware_nextstep_model(seq_past=8, d_model=128):
    traj_in = tf.keras.Input(shape=(seq_past, 2), name="traj")
    emb_in  = tf.keras.Input(shape=(2048,), name="emb")

    scene_emb = tf.keras.layers.Dense(d_model)(emb_in)
    scene_tok = tf.keras.layers.Reshape((1, d_model))(scene_emb)
    traj_emb  = tf.keras.layers.Dense(d_model)(traj_in)

    enc_in  = tf.keras.layers.Concatenate(axis=1)([scene_tok, traj_emb])
    enc_out = Encoder(nb_encoder=2)(enc_in)

    h      = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(enc_out)
    h      = tf.keras.layers.Dense(128, activation="relu")(h)
    y_next = tf.keras.layers.Dense(2)(h)

    return tf.keras.Model({"traj": traj_in, "emb": emb_in}, y_next)


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════
print("\n=== Chargement train ===")
X_train, Y_train, F_train, S_train = load_all_scenes(RAW_TRAIN_DIR)
print("\n=== Chargement val ===")
X_val, Y_val, F_val, S_val = load_all_scenes(RAW_VAL_DIR)
print(f"\nTrain: {X_train.shape} | Val: {X_val.shape}")

print("\n=== Précalcul embeddings train ===")
EMB_train = precompute_all_embeddings(F_train, S_train)
print("\n=== Précalcul embeddings val ===")
EMB_val   = precompute_all_embeddings(F_val, S_val)

# Normalisation
mean, std = fit_normalizer(X_train, Y_train)
X_train, Y_train = apply_normalizer(X_train, Y_train, mean, std)
X_val,   Y_val   = apply_normalizer(X_val,   Y_val,   mean, std)

# Datasets
train_ds = make_scene_nextstep_dataset_emb(X_train, Y_train, EMB_train, batch_size=64, shuffle=True)
val_ds   = make_scene_nextstep_dataset_emb(X_val,   Y_val,   EMB_val,   batch_size=64, shuffle=False)

# Modèle
model = get_scene_aware_nextstep_model(seq_past=SEQ_PAST, d_model=128)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
              loss="mse", metrics=["mae"])

# Callback ADE/FDE sur val multistep
print("\n=== Chargement val multistep ===")
Xv8, Yv12, Fv, Sv = load_all_scenes_multistep(RAW_VAL_DIR)
EMB_v = precompute_all_embeddings(Fv, Sv)

ade_fde_cb = ADE_FDE_Scene_Callback(
    Xv8_real=Xv8, Yv12_real=Yv12, EMB_val=EMB_v,
    mean=mean, std=std, seq_fut=SEQ_FUT, max_samples=512
)
early     = tf.keras.callbacks.EarlyStopping(monitor="val_ADE", mode="min", patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_ADE", mode="min", factor=0.5, patience=8, min_lr=1e-6, verbose=1)

# Entraînement
print("\n=== Entraînement ===")
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=200, callbacks=[ade_fde_cb, reduce_lr, early]
)

# Évaluation finale
Xv8_n       = (Xv8 - mean) / std
Y_pred_n    = rollout_nextstep_scene(model, Xv8_n, EMB_v, SEQ_FUT)
Y_pred_real = Y_pred_n * std + mean
ade, fde    = compute_ade_fde(Yv12, Y_pred_real)
print(f"\n[ROLLOUT scene-aware] ADE={ade:.4f} FDE={fde:.4f}")

model.save("trajectory_transformer.h5")

# Visualisation
plt.plot(history.history['loss'],    label='MSE Loss')
plt.plot(history.history["val_ADE"], label="val_ADE")
plt.plot(history.history["val_FDE"], label="val_FDE")
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Trajectory Transformer - Multi-scènes')
plt.legend(); plt.tight_layout(); plt.show()