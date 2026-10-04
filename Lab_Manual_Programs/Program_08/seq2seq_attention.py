import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# 1) Tiny toy parallel corpus
# -------------------------
pairs = [
    ("i love you", "je t'aime"),
    ("how are you", "comment ça va"),
    ("good morning", "bonjour"),
    ("thank you", "merci"),
    ("see you soon", "à bientôt"),
    ("where are you", "où es-tu"),
    ("i am fine", "je vais bien"),
    ("what is your name", "comment tu t'appelles"),
    ("nice to meet you", "ravi de te rencontrer"),
    ("good night", "bonne nuit"),
]

# -------------------------
# 2) Tokenization
# -------------------------
def build_tokenizer(texts, num_words=None, oov="<unk>"):
    tok = Tokenizer(num_words=num_words, oov_token=oov, filters="") # keep accents/punct
    tok.fit_on_texts(texts)
    return tok

en_texts = [en for en, fr in pairs]
fr_texts = [fr for en, fr in pairs]

# Add <sos> and <eos> to targets
fr_in_texts = ["<sos> " + t for t in fr_texts]
fr_out_texts = [t + " <eos>" for t in fr_texts]

tok_en = build_tokenizer(en_texts)
tok_fr = build_tokenizer(fr_in_texts + fr_out_texts)

en_seqs = tok_en.texts_to_sequences(en_texts)
fr_in_seqs = tok_fr.texts_to_sequences(fr_in_texts)
fr_out_seqs = tok_fr.texts_to_sequences(fr_out_texts)

max_len_en = max(len(s) for s in en_seqs)
max_len_fr = max(len(s) for s in fr_out_seqs)

en_seqs = pad_sequences(en_seqs, maxlen=max_len_en, padding="post")
fr_in_seqs = pad_sequences(fr_in_seqs, maxlen=max_len_fr, padding="post")
fr_out_seqs = pad_sequences(fr_out_seqs, maxlen=max_len_fr, padding="post")

en_vocab = len(tok_en.word_index) + 1
fr_vocab = len(tok_fr.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((en_seqs, fr_in_seqs, fr_out_seqs)).shuffle(100).batch(16)

# -------------------------
# 3) Model components
# -------------------------
EMB_EN = 64
EMB_FR = 64
RNN_UNITS = 128

class Encoder(tf.keras.Model):
    def __init__(self, vocab, emb_dim, rnn_units):
        super().__init__()
        self.emb = layers.Embedding(vocab, emb_dim, mask_zero=True)
        self.rnn = layers.GRU(rnn_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.emb(x)
        seq, h = self.rnn(x)
        return seq, h

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.v = layers.Dense(1)

    def call(self, enc_seq, dec_state, mask):
        dec_state_exp = tf.expand_dims(dec_state, 1)
        score = self.v(tf.nn.tanh(self.W1(enc_seq) + self.W2(dec_state_exp)))
        score = tf.squeeze(score, axis=-1)
        if mask is not None:
            score = tf.where(mask, score, tf.fill(tf.shape(score), -1e9))
        attn = tf.nn.softmax(score, axis=1)
        context = tf.matmul(tf.expand_dims(attn, 1), enc_seq)
        context = tf.squeeze(context, axis=1)
        return context, attn

class Decoder(tf.keras.Model):
    def __init__(self, vocab, emb_dim, rnn_units, attention):
        super().__init__()
        self.emb = layers.Embedding(vocab, emb_dim, mask_zero=True)
        self.rnn = layers.GRU(rnn_units, return_state=True)
        self.fc = layers.Dense(vocab)
        self.attn = attention

    def call(self, y_prev, state, enc_seq, enc_mask):
        y_emb = self.emb(y_prev)
        context, attn = self.attn(enc_seq, state, enc_mask)
        rnn_input = tf.concat([y_emb, context], axis=-1)
        out, state = self.rnn(tf.expand_dims(rnn_input, 1), initial_state=state)
        logits = self.fc(out)
        return logits, state, attn

encoder = Encoder(en_vocab, EMB_EN, RNN_UNITS)
attention = BahdanauAttention(RNN_UNITS)
decoder = Decoder(fr_vocab, EMB_FR, RNN_UNITS, attention)

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)

@tf.function
def train_step(en_batch, fr_in_batch, fr_out_batch):
    with tf.GradientTape() as tape:
        enc_seq, enc_state = encoder(en_batch)
        enc_mask = tf.cast(tf.not_equal(en_batch, 0), tf.bool)
        dec_state = enc_state
        loss = 0.0
        y_prev = fr_in_batch[:, 0]
        for t in tf.range(1, tf.shape(fr_out_batch)[1]):
            logits, dec_state, _ = decoder(y_prev, dec_state, enc_seq, enc_mask)
            y_true = fr_out_batch[:, t-1]
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            step_loss = loss_obj(y_true, logits) * mask
            loss += tf.reduce_sum(step_loss) / (tf.reduce_sum(mask) + 1e-9)
            if t < tf.shape(fr_in_batch)[1]:
                y_prev = fr_in_batch[:, t]
        batch_loss = loss / tf.cast(tf.shape(en_batch)[0], tf.float32)
    vars_ = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(batch_loss, vars_)
    optimizer.apply_gradients(zip(grads, vars_))
    return batch_loss

EPOCHS = 50
for epoch in range(1, EPOCHS + 1):
    for en_b, fr_in_b, fr_out_b in dataset:
        loss = train_step(en_b, fr_in_b, fr_out_b)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} - loss: {loss.numpy():.4f}")

idx2fr = {v: k for k, v in tok_fr.word_index.items()}
sos_id = tok_fr.word_index.get("<sos>")
eos_id = tok_fr.word_index.get("<eos>")

def translate(sentence_en, max_len=20):
    seq = pad_sequences(tok_en.texts_to_sequences([sentence_en]), maxlen=max_len_en, padding="post")
    enc_seq, enc_state = encoder(seq)
    enc_mask = tf.cast(tf.not_equal(seq, 0), tf.bool)
    dec_state = enc_state
    y_prev = tf.constant([sos_id], dtype=tf.int32)
    out_tokens = []
    for _ in range(max_len):
        logits, dec_state, _ = decoder(y_prev, dec_state, enc_seq, enc_mask)
        next_id = int(tf.argmax(logits[0]).numpy())
        if next_id == eos_id:
            break
        out_tokens.append(idx2fr.get(next_id, "<unk>"))
        y_prev = tf.constant([next_id], dtype=tf.int32)
    return " ".join(out_tokens)

print("\nDemo translations:")
tests = ["i love you", "thank you", "good night", "how are you", "where are you"]
for s in tests:
    print(f"EN: {s:>20} --> FR: {translate(s)}")
