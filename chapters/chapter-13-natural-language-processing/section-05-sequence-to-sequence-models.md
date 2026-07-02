# Section 13.5: Sequence-to-Sequence Models

> **Source:** Prosise, Ch. 13 - encoder-decoder and neural machine translation  
> **Prerequisites:** [Section 13.4](./section-04-recurrent-neural-networks.md)  
> **Glossary:** [encoder-decoder](../../GLOSSARY.md#encoder-decoder) | [teacher forcing](../../GLOSSARY.md#teacher-forcing) | [BLEU score](../../GLOSSARY.md#bleu-score) | [neural machine translation](../../GLOSSARY.md#neural-machine-translation)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## One Sequence In, Another Out

**Sequence-to-sequence (seq2seq)** models map variable-length input to variable-length output - neural machine translation (NMT), summarization, chatbots.

```
English:  "How are you?"     →  French: "Comment allez-vous ?"
```

Architecture: **encoder** compresses input; **decoder** generates output token by token.

> **In plain English:** The encoder reads the whole English sentence into a thought vector; the decoder speaks French one word at a time from that thought.

---

## Encoder-Decoder Architecture

```
Source:  w1  w2  w3  ...  wT
           ↓   ↓   ↓        ↓
Encoder LSTM → → → → → context vector c (= final h_T)
                              ↓
Decoder LSTM ← ← ← ← ← target: <start> y1 y2 ... yU
                              ↓
                         softmax over vocab
```

$$
P(y_1, \ldots, y_U \mid x_1, \ldots, x_T) = \prod_{u=1}^{U} P(y_u \mid y_{<u}, \mathbf{c})
$$
> **Readable form:** probability of target sequence = product of each next-token probabilities given previous targets and context c

See [encoder-decoder](../../GLOSSARY.md#encoder-decoder).

---

## Keras Encoder-Decoder Sketch

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Hyperparameters
vocab_src = 15000
vocab_tgt = 15000
embed_dim = 256
latent_dim = 512

# Encoder
encoder_inputs = layers.Input(shape=(None,), name='encoder_inputs')
enc_emb = layers.Embedding(vocab_src, embed_dim, mask_zero=True)(encoder_inputs)
_, state_h, state_c = layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = layers.Input(shape=(None,), name='decoder_inputs')
dec_emb = layers.Embedding(vocab_tgt, embed_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = layers.Dense(vocab_tgt, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Training requires **teacher forcing** - feed ground-truth previous tokens as decoder input.

---

## Teacher Forcing

During training, decoder input at step $u$ is **true** token $y_{u-1}$, not model prediction:

$$
\text{input}_u = y_{u-1} \quad \text{(training)}
$$
> **Readable form:** at training step u, feed the correct previous word - not what the model guessed

**Exposure bias:** at inference, errors compound because model feeds its own (possibly wrong) predictions. Mitigations: scheduled sampling, beam search - beyond Prosise scope but good to name.

See [teacher forcing](../../GLOSSARY.md#teacher-forcing).

---

## Data Preparation for NMT

English-French sentence pairs (Prosise uses parallel corpora):

```python
# Pseudo: source/target integer sequences, padded
# decoder_input:  [<start>, y1, y2, ..., y_{U-1}]
# decoder_target: [y1, y2, ..., yU, <end>]

def prepare_decoder_data(target_seqs, start_token=1, end_token=2):
    decoder_input = np.array([[start_token] + seq[:-1] for seq in target_seqs])
    decoder_target = np.array([seq + [end_token] for seq in target_seqs])
    return decoder_input, decoder_target
```

Separate **TextVectorization** layers for source and target languages.

---

## Inference: Greedy Decoding

At test time, generate one token at a time:

```python
def translate_greedy(encoder_model, decoder_model, source_seq, max_len=50, start=1, end=2):
    states = encoder_model.predict(source_seq)
    target_seq = [start]
    for _ in range(max_len):
        output_tokens = decoder_model.predict([np.array([target_seq]), states])
        sampled = int(np.argmax(output_tokens[0, -1, :]))
        if sampled == end:
            break
        target_seq.append(sampled)
    return target_seq
```

**Greedy** picks argmax each step - fast but suboptimal. **Beam search** keeps top-k partial sequences for better translations.

---

## BLEU Score Evaluation

**BLEU** (Bilingual Evaluation Understudy) measures n-gram overlap between hypothesis and reference translations:

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$
> **Readable form:** BLEU = brevity penalty times geometric mean of n-gram precision scores

Range 0-1 (often reported 0-100). Imperfect but standard for NMT benchmarks.

```python
# pip install sacrebleu
import sacrebleu

refs = [["Comment allez-vous ?"]]
hyp = ["Comment allez vous"]
bleu = sacrebleu.sentence_bleu(hyp, refs)
print(bleu.score)
```

See [BLEU in glossary](../../GLOSSARY.md#bleu-score).

---

## Attention Mechanism (Bridge to Transformers)

Classic seq2seq bottleneck: single context vector $\mathbf{c}$ must encode entire source sentence. **Attention** lets decoder look at **all encoder hidden states** each step:

$$
\alpha_{tu} = \text{softmax}(\mathbf{h}_u^{dec\top} \mathbf{W} \mathbf{h}_t^{enc})
$$
> **Readable form:** attention weight = how much decoder step u focuses on encoder step t

Attention improves long sentences dramatically - precursor to full transformers ([Section 13.6](./section-06-the-transformer-architecture.md)).

---

## Prosise NMT Mini-Project Expectations

[Lab 13](./section-lab-13-nlp-progression-sentiment-to-translation.md) uses a **small** parallel corpus (thousands of pairs, not millions):

| Expectation | Realistic target |
|-------------|------------------|
| Training epochs | 10-20 with early stopping |
| BLEU | Modest (15-30 on tiny data) |
| Qualitative | Short phrases translate; long sentences fail |

Goal is **architectural literacy**, not Google Translate parity.

---

## Seq2seq Beyond Translation

| Task | Input seq | Output seq |
|------|-----------|------------|
| NMT | Source language | Target language |
| Summarization | Article | Headline |
| Chatbot | User utterance | Response |
| Code generation | Docstring | Code |

Same encoder-decoder template; data and evaluation metrics differ.

---

## Key Takeaways

1. **Seq2seq** = encoder compresses input; decoder generates output autoregressively
2. **Teacher forcing** trains decoder with ground-truth previous tokens
3. **Greedy decoding** at inference; beam search improves quality
4. **BLEU** evaluates translation against reference n-grams
5. **Attention** fixes single-vector bottleneck - leads to transformers

---

## Check Your Understanding

1. What is the encoder's output used to initialize?
2. Why is teacher forcing used during training but not at inference?
3. What does BLEU measure?
4. Why does a single context vector struggle on long sentences?
5. How does attention differ from fixed context vector seq2seq?

---

## References

- Prosise, Ch. 13 - NMT sections
- Sutskever et al., seq2seq - [https://arxiv.org/abs/1409.3215](https://arxiv.org/abs/1409.3215)
- Bahdanau et al., attention - [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)

---

**Next:** [Section 13.6 - The Transformer Architecture](./section-06-the-transformer-architecture.md)
