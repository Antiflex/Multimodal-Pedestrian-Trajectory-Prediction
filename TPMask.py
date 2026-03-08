from xml.parsers.expat import model
from sklearn.discriminant_analysis import softmax
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



input_embedding = [["Salut", "comment", "ca", "va", "?"]]

output_embedding = [["<START>","Hi", "how", "are", "you", "?"]]

def get_vocabulary(sequences):
    token_to_info = {}
    for sequence in sequences:
        for word in sequence:
            if word not in token_to_info:
                token_to_info[word] = len(token_to_info)
    return token_to_info

input_voc = get_vocabulary(input_embedding)
output_voc = get_vocabulary(output_embedding)

input_voc["<START>"] = len(input_voc)
input_voc["<END>"] = len(input_voc)
input_voc["<PAD>"] = len(input_voc)

output_voc["<END>"] = len(output_voc)
output_voc["<PAD>"] = len(output_voc)


print (input_voc)
print (output_voc)


def sequence_to_int(sequences,voc):
    for sequence in sequences:
        for s, word in enumerate(sequence):
            sequence[s] = voc[word]
    return np.array(sequences)

input_seq = sequence_to_int(input_embedding,input_voc)
output_seq = sequence_to_int(output_embedding,output_voc)

print (input_seq)
print (output_seq)


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__ (self, nb_token, **kwargs):
        self.nb_token = nb_token
        super(**kwargs).__init__()
        
    
    def build(self,input_shape):
        self.word_embedding = tf.keras.layers.Embedding(self.nb_token,256)
        super().build(input_shape)

    def call(self,x):
        embed = self.word_embedding(x)
        return embed


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__ (self, **kwargs):
        super(**kwargs).__init__()
        
    def build(self, input_shape):
        self.query_layer = tf.keras.layers.Dense(256)
        self.value_layer = tf.keras.layers.Dense(256)
        self.key_layer = tf.keras.layers.Dense(256)
        super().build(input_shape)

    def call(self, x):

        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.math.sqrt(256.0)

        softmax_QK = tf.nn.softmax(QK, axis=-1)

        attention = tf.matmul(softmax_QK, V)

        self.attention_weights = softmax_QK
        return attention
    
def test():
    layer_input = tf.keras.Input(shape=(5,))

    embedding_layer = EmbeddingLayer(nb_token=len(input_voc))
    attn_layer = ScaledDotProductAttention()

    x = embedding_layer(layer_input)
    y = attn_layer(x)

    model = tf.keras.Model(layer_input, y)
    model.summary()
    return model, attn_layer


m_test, attn_layer = test()
out = m_test(input_seq)

weights = attn_layer.attention_weights.numpy()  # (batch, seq_len, seq_len)
print("weights shape:", weights.shape)

tokens = ["Salut", "comment", "ca", "va", "?"]
A = weights[0]  # on enlève la dimension batch → (5,5)

plt.figure(figsize=(6,5))
plt.imshow(A)
plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
plt.yticks(range(len(tokens)), tokens)
plt.colorbar()
plt.title("Self-attention weights")
plt.tight_layout()
plt.show()



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__ (self, dim = 256, nb_head = 8, **kwargs):
        self.dim = dim
        self.head_dim = dim // nb_head
        self.nb_head = nb_head

        super(**kwargs).__init__()

    def build(self, input_shape):

        self.query_layer = tf.keras.layers.Dense(256)
        self.value_layer = tf.keras.layers.Dense(256)
        self.key_layer = tf.keras.layers.Dense(256)
        self.out_proj = tf.keras.layers.Dense(256)
        



        super().build(input_shape)

    def mask_softmax(self, x, mask):
        x_expe = tf.math.exp(x)
        x_expe_masked = x_expe * mask
        x_expe_sum = tf.reduce_sum(x_expe_masked, axis=-1)    
        x_expe_sum = tf.expand_dims(x_expe_sum, axis=-1)
        softmax = x_expe_masked / x_expe_sum
        return softmax

    def call(self, x, mask = None):
        in_query, in_key, in_value = x

        if isinstance(mask, (tuple, list)):
            mask = None
        Q = self.query_layer(in_query)
        K = self.key_layer(in_key)
        V = self.value_layer(in_value)

        batch_size = tf.shape(Q)[0]
        Q_seq_len = tf.shape(Q)[1]
        K_seq_len = tf.shape(K)[1]
        V_seq_len = tf.shape(V)[1]
        Q = tf.reshape(Q,[batch_size, Q_seq_len, self.nb_head, self.head_dim])
        K = tf.reshape(K,[batch_size, K_seq_len, self.nb_head, self.head_dim])
        V = tf.reshape(V,[batch_size, V_seq_len, self.nb_head, self.head_dim])

        Q = tf.transpose(Q, [0,2,1,3])
        K = tf.transpose(K, [0,2,1,3])
        V = tf.transpose(V, [0,2,1,3])


        Q = tf.reshape(Q,[batch_size * self.nb_head, Q_seq_len, self.head_dim])
        K = tf.reshape(K,[batch_size * self.nb_head, K_seq_len, self.head_dim])
        V = tf.reshape(V,[batch_size * self.nb_head, V_seq_len, self.head_dim])

        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.math.sqrt(256.)


        if mask is not None:
            QK = QK * mask
            #print("mask", mask.shape)
            #print("QK", QK.shape)
            softmax_QK = self.mask_softmax(QK, mask)
        else:
            softmax_QK = tf.nn.softmax(QK, axis=-1)

        attention = tf.matmul(softmax_QK, V)

        attention = tf.reshape(
            attention, [batch_size, self.nb_head, Q_seq_len, self.head_dim])
        
        attention = tf.transpose(attention, [0, 2, 1, 3])
        # Concat
        attention = tf.reshape(
            attention, [batch_size, Q_seq_len, self.nb_head*self.head_dim]
        )

        out_attention = self.out_proj(attention)
        return out_attention

def test():
    layer_input = tf.keras.Input(shape=(6,))

    # mask
    mask = tf.sequence_mask(tf.range(6) + 1, 6)
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=0)

    embedding = EmbeddingLayer(nb_token = 6)(layer_input)
    multi_attention = MultiHeadAttention()((embedding, embedding, embedding),mask=mask)
    model = tf.keras.Model(layer_input, multi_attention)
    model.summary()
    return model

m_test = test()
out = m_test(output_seq)
print(out.shape)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__ (self,  **kwargs):
        super(**kwargs).__init__()
        
    
    def build(self,input_shape):
        self.multi_head_attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dense_out = tf.keras.layers.Dense(256)
        super().build(input_shape)

    def call(self,x, mask = None):

        attention = self.multi_head_attention((x,x,x), mask=mask)

        post_attention = self.norm(attention + x)

        x = self.dense_out(post_attention)

        enc_output = self.norm(x + post_attention)

        print ("enc_output", enc_output.shape)
        return enc_output
    


    
class Encoder(tf.keras.layers.Layer):
    def __init__ (self, nb_encoder,**kwargs):
        self.nb_encoder = nb_encoder
        super(**kwargs).__init__()
        
    
    def build(self,input_shape):

        self.encoder_layers = []   

        for nb in range (self.nb_encoder):
            self.encoder_layers.append(
                EncoderLayer()
            )
        super().build(input_shape)

    def call(self,x):
        for encoder_layer in self.encoder_layers:
            x=encoder_layer(x)
        return x

def test():
    layer_input = tf.keras.Input(shape=(5,))

    embedding = EmbeddingLayer(nb_token = 5)(layer_input)
    enc_output = Encoder(nb_encoder=6)(embedding)
    model = tf.keras.Model(layer_input, enc_output)
    model.summary()
    return model

m_test = test()
out = m_test(input_seq)

print(out.shape)


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(**kwargs).__init__()
    
    def build(self, input_shape):
        self.multi_head_self_attention = MultiHeadAttention()
        self.multi_head_enc_attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()

        self.proj_output = tf.keras.layers.Dense(256)

        super().build(input_shape)
    
    def call(self, x):

        enc_output, output_embedding, mask = x

        self_attention = self.multi_head_self_attention(
            (output_embedding, output_embedding, output_embedding), mask=mask
        )
        post_self_att = self.norm(output_embedding + self_attention)

        enc_attention = self.multi_head_enc_attention(
            (post_self_att, enc_output, enc_output), mask=None
        )
        post_enc_attention = self.norm(enc_attention + post_self_att)
        proj_out = self.proj_output(post_enc_attention)

        dec_output = self.norm(proj_out + post_enc_attention)

        return dec_output


class Decoder(tf.keras.layers.Layer):

    def __init__(self, nb_decoder, **kwargs):
        self.nb_decoder = nb_decoder
        super(**kwargs).__init__()
    
    def build(self, input_shape):

        self.decoder_layers = []
        for nb in range(self.nb_decoder):
            self.decoder_layers.append(
                DecoderLayer()        
            )
        super().build(input_shape)
    
    def call(self, x):

        enc_output, output_embedding, mask = x

        dec_output = output_embedding
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer((enc_output, dec_output, mask))
        return dec_output


def get_transformer_model(output_voc):
    input_token = tf.keras.Input(shape=(5,))
    output_token = tf.keras.Input(shape=(6,))

    # Positional encoding
    input_pos_encoding = EmbeddingLayer(nb_token=5)(tf.range(5))
    output_pos_encoding = EmbeddingLayer(nb_token=6)(tf.range(6))

    # Retrieve embedding
    input_embedding = EmbeddingLayer(nb_token=5)(input_token)
    output_embedding = EmbeddingLayer(nb_token=6)(output_token)

    # Add the positional encoding
    input_embedding = input_embedding + input_pos_encoding
    output_embedding = output_embedding + output_pos_encoding

    # Encoder
    enc_output = Encoder(nb_encoder=6)(input_embedding)

    # mask + Decoder
    mask = tf.sequence_mask(tf.range(6) + 1, 6)
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=0)
    dec_output = Decoder(nb_decoder=6)((enc_output, output_embedding, mask))
    

    # Predictions
    out_pred = tf.keras.layers.Dense(len(output_voc))(dec_output)
    predictions = tf.keras.layers.Softmax(axis=-1)(out_pred)

    model = tf.keras.Model([input_token, output_token], predictions)
    model.summary()
    return model

transformer = get_transformer_model(output_voc)
out = transformer((input_seq, output_seq))
print(out.shape)
