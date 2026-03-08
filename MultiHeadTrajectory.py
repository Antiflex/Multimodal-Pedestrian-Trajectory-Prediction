import os
from dotenv import load_dotenv
load_dotenv()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch

from collections import defaultdict

SEQ_PAST   = 8   # nombre de positions passées (entrée)
SEQ_FUT    = 12   # nombre de positions futures  (sortie)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU disponible :", torch.cuda.get_device_name(0))

def compute_ade_fde(y_true, y_pred):
    """
    y_true, y_pred: (N, T, 2)
    ADE = moyenne des distances sur tous les pas
    FDE = distance au dernier pas
    """
    dist = np.linalg.norm(y_pred - y_true, axis=-1)  # (N, T)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde



def load_txt(path):
    return np.loadtxt(path)  # frame, person, x, y

def build_sequences(data, seq_past=8, seq_future=12, stride=1):
    tracks = defaultdict(list)
    for frame, pid, x, y in data:
        tracks[int(pid)].append((float(frame), float(x), float(y)))

    X_list, Y_list = [], []
    for pid, pts in tracks.items():
        pts.sort(key=lambda t: t[0])
        frames = np.array([p[0] for p in pts], dtype=np.float32)
        xy     = np.array([[p[1], p[2]] for p in pts], dtype=np.float32)

        dt = np.median(np.diff(frames)) if len(frames) > 1 else None

        for i in range(0, len(xy) - (seq_past + seq_future) + 1, stride):
            f_window = frames[i:i+seq_past+seq_future]
            if dt is not None:
                diffs = np.diff(f_window)
                if np.any(np.abs(diffs - dt) > 1e-3):
                    continue

            X_list.append(xy[i:i+seq_past])
            Y_list.append(xy[i+seq_past:i+seq_past+seq_future])

    X = np.stack(X_list).astype(np.float32) if X_list else np.empty((0,seq_past,2), np.float32)
    Y = np.stack(Y_list).astype(np.float32) if Y_list else np.empty((0,seq_future,2), np.float32)
    return X, Y

def fit_normalizer(X, Y):
    all_xy = np.concatenate([X.reshape(-1,2), Y.reshape(-1,2)], axis=0)
    mean = all_xy.mean(axis=0)
    std  = all_xy.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def apply_normalizer(X, Y, mean, std):
    return (X-mean)/std, (Y-mean)/std

def make_tf_dataset_no_teacher_forcing(X, Y, seq_fut, batch_size=64, shuffle=True):
    # on initialise l'entrée décodeur avec la dernière position observée (répétée)
    last = X[:, -1:, :]                     # (N,1,2)
    Y_in = np.repeat(last, seq_fut, axis=1) # (N,seq_fut,2)

    ds = tf.data.Dataset.from_tensor_slices(((X, Y_in), Y))  # input=(X, Y_in), target=Y
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds





# ─────────────────────────────────────────────
# Multi-Head Attention (inchangée)
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

        QK = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(256.)

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
# Encoder (inchangé)
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
        attention   = self.multi_head_attention((x, x, x), mask=None)
        post_att    = self.norm(attention + x)
        ff          = self.dense_out(post_att)
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
# Decoder (inchangé)
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
# Modèle Transformer pour trajectoires
# ─────────────────────────────────────────────
def get_trajectory_model(seq_past, seq_future):
    input_token  = tf.keras.Input(shape=(seq_past,   2))   # (x,y) passés
    output_token = tf.keras.Input(shape=(seq_future, 2))   # (x,y) futurs

    # ✅ Dense(256) au lieu de EmbeddingLayer — les coordonnées sont continues
    input_embedding  = tf.keras.layers.Dense(128)(input_token)
    output_embedding = tf.keras.layers.Dense(128)(output_token)

    # Encodeur
    enc_output = Encoder(nb_encoder=2)(input_embedding)

    # Masque causal pour le décodeur
    mask = tf.sequence_mask(tf.range(seq_future) + 1, seq_future)
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=0)

    # Décodeur
    dec_output = Decoder(nb_decoder=2)((enc_output, output_embedding, mask))

    # ✅ Dense(2) au lieu de Dense(vocab_size) + Softmax — régression sur (x, y)
    predictions = tf.keras.layers.Dense(2)(dec_output)

    model = tf.keras.Model([input_token, output_token], predictions)
    return model




dataset_base = os.getenv("DATASET_BASE")
trainning_set = "raw/train/students001_train.txt"
val_set = "raw/val/students001_val.txt"
print("Dataset base path:", dataset_base)
train_raw = load_txt(os.path.join(dataset_base, trainning_set))
val_raw = load_txt(os.path.join(dataset_base, val_set))


# 2) Fenêtres 8->12
X_train, Y_train = build_sequences(train_raw, SEQ_PAST, SEQ_FUT)
X_val,   Y_val   = build_sequences(val_raw,   SEQ_PAST, SEQ_FUT)

print("Train:", X_train.shape, Y_train.shape)
print("Val  :", X_val.shape, Y_val.shape)

# 3) Normalisation (fit sur train uniquement)
mean, std = fit_normalizer(X_train, Y_train)
X_train, Y_train = apply_normalizer(X_train, Y_train, mean, std)
X_val,   Y_val   = apply_normalizer(X_val,   Y_val,   mean, std)

# 4) tf.data
train_ds = make_tf_dataset_no_teacher_forcing(X_train, Y_train, SEQ_FUT, batch_size=64, shuffle=True)
val_ds   = make_tf_dataset_no_teacher_forcing(X_val,   Y_val,   SEQ_FUT, batch_size=64, shuffle=False)

# 5) Entraîner
model = get_trajectory_model(SEQ_PAST, SEQ_FUT)
model.summary()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    clipnorm=1.0
)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

def compute_ade_fde(y_true, y_pred):
    dist = np.linalg.norm(y_pred - y_true, axis=-1)  # (N, T)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde


class ADE_FDE_Multistep_Callback(tf.keras.callbacks.Callback):
    """
    Calcule ADE/FDE sur la validation (multi-step) à chaque époque
    et ajoute val_ADE / val_FDE dans history + affichage console.
    """
    def __init__(self, X_val_n, Y_val_real, mean, std, max_samples=2048):
        super().__init__()
        self.X_val_n = X_val_n          # X_val normalisé (N,8,2)
        self.Y_val_real = Y_val_real    # Y_val en réel (N,12,2)
        self.mean = mean
        self.std = std
        self.max_samples = max_samples

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        n = min(len(self.X_val_n), self.max_samples)
        Xn = self.X_val_n[:n]
        Y_true = self.Y_val_real[:n]

        # input décodeur: répéter dernière pos observée (normalisée)
        last = Xn[:, -1:, :]                      # (n,1,2)
        Y_infer = np.repeat(last, Y_true.shape[1], axis=1)  # (n,12,2)

        # prédiction normalisée
        Y_pred_n = self.model.predict([Xn, Y_infer], verbose=0)

        # dé-normaliser
        Y_pred = Y_pred_n * self.std + self.mean

        ade, fde = compute_ade_fde(Y_true, Y_pred)

        logs["val_ADE"] = float(ade)
        logs["val_FDE"] = float(fde)

        print(f" — val_ADE: {ade:.4f} — val_FDE: {fde:.4f}", end="")

# Y_val en réel (dénormalisé) pour ADE/FDE
Y_val_real = Y_val * std + mean

ade_fde_cb = ADE_FDE_Multistep_Callback(
    X_val_n=X_val,            # X_val est déjà normalisé
    Y_val_real=Y_val_real,    # Y_val en réel
    mean=mean,
    std=std,
    max_samples=2048
)

early = tf.keras.callbacks.EarlyStopping(
    monitor="val_ADE",
    mode="min",
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_ADE",
    mode="min",
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=200,
    callbacks=[ade_fde_cb, reduce_lr, early]
)

last = X_val[:, -1:, :]
Y_infer = np.repeat(last, SEQ_FUT, axis=1)
Y_pred_n = model.predict([X_val, Y_infer], verbose=0)
Y_pred_real = Y_pred_n * std + mean
ade, fde = compute_ade_fde(Y_val_real, Y_pred_real)
print(f"[ONE-SHOT] ADE={ade:.4f}  FDE={fde:.4f}")


model.save("trajectory_transformer.h5")

# ─────────────────────────────────────────────
# Visualisation de la loss
# ─────────────────────────────────────────────
plt.plot(history.history["loss"], label="train_MSE")
plt.plot(history.history["val_loss"], label="val_MSE")

plt.plot(history.history["val_ADE"], label="val_ADE")
plt.plot(history.history["val_FDE"], label="val_FDE")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Trajectory Transformer - Loss / ADE / FDE")
plt.legend()
plt.tight_layout()
plt.show()