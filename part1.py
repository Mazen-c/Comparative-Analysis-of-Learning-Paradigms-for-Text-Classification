# %% [Cell 1] Imports
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset


# %% [Cell 2] Load dataset and inspect structure
dataset = load_dataset("dair-ai/emotion")

# Print high-level dataset info (splits, features, number of rows)
print(dataset)

# Show one raw example from each split to understand the data format
for split in ["train", "validation", "test"]:
    print(f"\n--- {split.upper()} SAMPLE ---")
    print(dataset[split][0])


# %% [Cell 3] Class distribution — counts per emotion per split
label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Print total examples per split
for split in ["train", "validation", "test"]:
    print(f"{split}: {len(dataset[split])} examples")

# Count how many examples belong to each emotion label
for split in ["train", "validation", "test"]:
    labels = dataset[split]["label"]
    counts = Counter(labels)
    print(f"\n{split.upper()}:")
    for label_id, count in sorted(counts.items()):
        print(f"  {label_names[label_id]:<10} : {count}")


# %% [Cell 4] Text length statistics — character-level summary per split
for split in ["train", "validation", "test"]:
    lengths = [len(text) for text in dataset[split]["text"]]
    avg = sum(lengths) / len(lengths)
    print(f"  {split} — min: {min(lengths)}, max: {max(lengths)}, avg: {avg:.1f}")


# %% [Cell 5] Plot class distribution across all splits
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, split in enumerate(["train", "validation", "test"]):
    labels = dataset[split]["label"]
    counts = Counter(labels)
    # Map integer label IDs to human-readable emotion names
    names  = [label_names[k] for k in sorted(counts)]
    values = [counts[k]       for k in sorted(counts)]

    axes[i].bar(names, values, color="steelblue")
    axes[i].set_title(f"{split.capitalize()} — Class Distribution")
    axes[i].set_xlabel("Emotion")
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()


# %% [Cell 6] Plot text length distribution across all splits
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, split in enumerate(["train", "validation", "test"]):
    # Compute character-level length for every text in this split
    lengths = [len(text) for text in dataset[split]["text"]]
    axes[i].hist(lengths, bins=40, color="coral", edgecolor="white")
    axes[i].set_title(f"{split.capitalize()} — Text Length")
    axes[i].set_xlabel("Characters")
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("text_length_distribution.png", dpi=150)
plt.show()


# %% [Cell 7] Build custom vocabulary for the RNN model
def simple_tokenize(text):
    # Lowercase and whitespace-split; no stemming or punctuation removal
    # (punctuation carries emotional signal)
    return text.lower().split()

# Count every token across the entire training set
word_freq = Counter()
for text in dataset["train"]["text"]:
    word_freq.update(simple_tokenize(text))

PAD_TOKEN = "<PAD>"  # used to pad sequences to a fixed length
UNK_TOKEN = "<UNK>"  # replaces tokens not seen during training

# Reserve indices 0 and 1 for special tokens, then add words with freq >= 2
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, freq in word_freq.most_common():
    if freq >= 2:  # hapax legomena are likely noise; skip them
        vocab[word] = len(vocab)

print(f"Vocabulary size : {len(vocab)}")


# %% [Cell 8] RNN encoding — convert text to a fixed-length integer sequence
def encode_rnn(text, vocab, max_len=50):
    # Truncate to max_len tokens (99%+ of tweets fit within 50 whitespace tokens)
    tokens = simple_tokenize(text)[:max_len]
    # Map each token to its vocab index, falling back to UNK for OOV words
    ids    = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    # Right-pad with PAD index so every sequence has the same length
    ids   += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids

# Demonstrate encoding on the first training example
sample_text = dataset["train"][0]["text"]
print("Text   :", sample_text)
print("Encoded:", encode_rnn(sample_text, vocab))


# %% [Cell 9] BERT/DistilBERT tokenizer — WordPiece tokenization demo
# Load the pre-trained DistilBERT tokenizer (shares vocab with BERT-base-uncased)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

sample_text = dataset["train"][0]["text"]
encoded = tokenizer(
    sample_text,
    truncation=True,       # clip sequences longer than max_length
    padding="max_length",  # pad shorter sequences to max_length
    max_length=64,         # 64 tokens covers all examples; 512 is overkill here
    return_tensors="pt"    # return PyTorch tensors
)

print("Text          :", sample_text)
print("Input IDs     :", encoded["input_ids"])
print("Attention Mask:", encoded["attention_mask"])  # 1 = real token, 0 = padding
print("Shape         :", encoded["input_ids"].shape)


# %% [Cell 10] Preprocessing decisions — rationale summary
justification = """
## Preprocessing Decisions

| Choice                          | Rationale                                                                                       |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| Lowercasing (RNN only)          | Reduces vocab size. BERT handles casing internally.                                             |
| Frequency threshold >= 2 (RNN) | Words appearing once are likely noise; dropping them shrinks the embedding table.               |
| max_length = 50 tokens (RNN)   | 99%+ of tweets fit within 50 whitespace-split tokens.                                           |
| max_length = 64 (BERT/LLM)     | Covers all examples. BERT's default 512 is overkill here.                                       |
| No punctuation stripping        | Punctuation carries emotional signal ("fine. whatever." vs "fine whatever").                    |
| No stop-word removal            | Emotion often lives in function words ("I do love", "I don't care").                            |
"""

print(justification)
