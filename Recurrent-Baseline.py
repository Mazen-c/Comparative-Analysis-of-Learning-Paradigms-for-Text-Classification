# %% [Cell 1] Imports and reproducibility setup
import os
import random
import re
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset


# A fixed seed makes the random embedding initialization and training order reproducible.
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# %% [Cell 2] Dataset loading and label setup
# Required bullet: Train only from the labeled data.
# We use the same labeled Hugging Face emotion dataset as the rest of the project.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NUM_LABELS = len(label_names)


def load_emotion_dataset():
    """Load cached Arrow splits first, then fall back to Hugging Face if needed."""
    cache_root = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "datasets",
        "dair-ai___emotion",
        "split",
        "0.0.0",
    )
    if os.path.isdir(cache_root):
        for revision in os.listdir(cache_root):
            revision_dir = os.path.join(cache_root, revision)
            split_paths = {
                "train": os.path.join(revision_dir, "emotion-train.arrow"),
                "validation": os.path.join(revision_dir, "emotion-validation.arrow"),
                "test": os.path.join(revision_dir, "emotion-test.arrow"),
            }
            if all(os.path.exists(path) for path in split_paths.values()):
                print(f"Loading cached dair-ai/emotion Arrow files from: {revision_dir}")
                return DatasetDict(
                    {
                        split: HFDataset.from_file(path)
                        for split, path in split_paths.items()
                    }
                )

    print("Cached Arrow files not found; loading dair-ai/emotion through Hugging Face.")
    return load_dataset("dair-ai/emotion")


dataset = load_emotion_dataset()


def env_int(name, default):
    """Read integer hyperparameters from environment variables when provided."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Ignoring invalid {name}={value!r}; using {default}.")
        return default


# FAST_DEV_RUN is only for quick debugging. The default uses the full dataset.
FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "").lower() in {"1", "true", "yes"}
BATCH_SIZE = env_int("BATCH_SIZE", 16 if FAST_DEV_RUN else 64)
EPOCHS = env_int("EPOCHS", 2 if FAST_DEV_RUN else 10)
MAX_LEN = env_int("MAX_LEN", 50)
TRAIN_SAMPLE_LIMIT = env_int("TRAIN_SAMPLE_LIMIT", 256 if FAST_DEV_RUN else 0)
VAL_SAMPLE_LIMIT = env_int("VAL_SAMPLE_LIMIT", 128 if FAST_DEV_RUN else 0)
TEST_SAMPLE_LIMIT = env_int("TEST_SAMPLE_LIMIT", 128 if FAST_DEV_RUN else 0)


# %% [Cell 3] Vocabulary building from the training split only
# Required bullet: Randomly initialized embedding layer.
# The embedding matrix is random, so this vocabulary is built from labeled train text only.
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def simple_tokenize(text):
    """Lowercase and tokenize words/punctuation without using pretrained tokenizers."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text.lower())


def limit_split(split_name, limit):
    """Optionally shorten a split for FAST_DEV_RUN while keeping full-data defaults."""
    split_dataset = dataset[split_name]
    if limit > 0:
        split_dataset = split_dataset.select(range(min(limit, len(split_dataset))))
    return split_dataset


train_split = limit_split("train", TRAIN_SAMPLE_LIMIT)
val_split = limit_split("validation", VAL_SAMPLE_LIMIT)
test_split = limit_split("test", TEST_SAMPLE_LIMIT)

word_freq = Counter()
for text in train_split["text"]:
    word_freq.update(simple_tokenize(text))

vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
for token, freq in word_freq.most_common():
    # Keeping words that appear at least twice reduces noise and keeps the model compact.
    if freq >= 2:
        vocab[token] = len(vocab)

print(f"Vocabulary size: {len(vocab):,}")
print(f"Train/Val/Test examples: {len(train_split):,}/{len(val_split):,}/{len(test_split):,}")


# %% [Cell 4] Encoding text and creating PyTorch datasets
# Required bullet: Train on the full training set with an appropriate batch size.
# Each text becomes a fixed-length integer sequence plus its real length for the RNN.
def encode_text(text, max_len=MAX_LEN):
    tokens = simple_tokenize(text)[:max_len]
    token_ids = [vocab.get(token, UNK_IDX) for token in tokens]
    length = max(1, len(token_ids))
    token_ids += [PAD_IDX] * (max_len - len(token_ids))
    return token_ids, length


class EmotionRnnDataset(TorchDataset):
    def __init__(self, split_dataset):
        self.texts = list(split_dataset["text"])
        self.labels = torch.tensor(list(split_dataset["label"]), dtype=torch.long)
        encoded = [encode_text(text) for text in self.texts]
        self.input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "lengths": self.lengths[index],
            "labels": self.labels[index],
        }


train_loader = DataLoader(EmotionRnnDataset(train_split), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(EmotionRnnDataset(val_split), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(EmotionRnnDataset(test_split), batch_size=BATCH_SIZE, shuffle=False)


# %% [Cell 5] LSTM/GRU classifier architecture
# Required bullet: Build an LSTM or GRU classifier with random embeddings, recurrent layers,
# and a final dense classification head with softmax.
class RecurrentEmotionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        rnn_type="GRU",
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()
        if rnn_type not in {"GRU", "LSTM"}:
            raise ValueError("rnn_type must be 'GRU' or 'LSTM'")

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Random embedding layer: no pretrained vectors are loaded.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)

        rnn_class = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Final dense classification head. Softmax is exposed for inference probabilities.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, lengths, return_probs=False):
        embedded = self.embedding(input_ids)

        # Packing tells the RNN to ignore padding tokens when building the sentence vector.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)

        # LSTM returns (hidden_state, cell_state); GRU returns hidden_state only.
        if self.rnn_type == "LSTM":
            hidden = hidden[0]

        if self.bidirectional:
            sentence_vector = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            sentence_vector = hidden[-1]

        logits = self.classifier(self.dropout(sentence_vector))
        if return_probs:
            return self.softmax(logits)
        return logits


# %% [Cell 6] Training and validation helpers
# Required bullet: Train with an appropriate optimizer, learning rate, and batch size.
# CrossEntropyLoss expects raw logits, so softmax is not applied during training.
def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(is_training):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=range(NUM_LABELS),
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, macro_f1


def train_model(rnn_type):
    """Train one recurrent model and keep the best validation checkpoint."""
    print(f"\nTraining {rnn_type} baseline...\n")

    model = RecurrentEmotionClassifier(
        vocab_size=len(vocab),
        num_labels=NUM_LABELS,
        rnn_type=rnn_type,
    ).to(device)

    print(f"{rnn_type} total parameters    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"{rnn_type} trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }
    best_val_macro_f1 = -1.0
    best_state = None
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_macro_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_macro_f1 = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_macro_f1"].append(train_macro_f1)
        history["val_macro_f1"].append(val_macro_f1)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"{rnn_type} Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Macro F1: {train_macro_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_macro_f1:.4f}"
        )

    training_time = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "name": rnn_type,
        "model": model,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "training_time": training_time,
    }


# %% [Cell 7] Plot training and validation loss curves across epochs
# Required bullet: Plot training and validation loss curves across epochs.
def plot_loss_curves(results):
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(10, 6))

    for result in results:
        name = result["name"]
        history = result["history"]
        plt.plot(epochs_range, history["train_loss"], marker="o", label=f"{name} Train Loss")
        plt.plot(epochs_range, history["val_loss"], marker="s", linestyle="--", label=f"{name} Val Loss")

    plt.title("Training and Validation Loss - Recurrent Baselines")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rnn_training_curves.png", dpi=150)
    plt.close()


# %% [Cell 8] Test evaluation and inference timing
# Required bullet: Evaluate on the test set and report accuracy, macro F1, and per-class F1.
# Required bullet: Measure and report inference time on the test set.
def evaluate_on_test(model):
    model.eval()
    all_preds = []
    all_labels = []

    infer_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            # The model exposes softmax probabilities for inference, then argmax picks a class.
            probs = model(input_ids, lengths, return_probs=True)
            preds = probs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    inference_time = time.time() - infer_start
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=range(NUM_LABELS),
        average="macro",
        zero_division=0,
    )
    per_class_f1 = f1_score(
        all_labels,
        all_preds,
        labels=range(NUM_LABELS),
        average=None,
        zero_division=0,
    )

    return {
        "labels": all_labels,
        "preds": all_preds,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "inference_time": inference_time,
    }


# %% [Cell 9] Confusion matrix display for the test set
# Required bullet: Display a confusion matrix on the test set.
def plot_confusion_matrix(labels, preds, model_name):
    cm = confusion_matrix(labels, preds, labels=range(NUM_LABELS))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title(f"Confusion Matrix - {model_name} Recurrent Baseline")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("rnn_confusion_matrix.png", dpi=150)
    plt.close()


# %% [Cell 10] Main experiment runner
# Required bullet: Measure and report total training time.
if __name__ == "__main__":
    experiment_start = time.time()

    # Train both candidates under identical settings, then select by validation macro F1.
    results = [train_model("GRU"), train_model("LSTM")]
    plot_loss_curves(results)

    best_result = max(results, key=lambda item: item["best_val_macro_f1"])
    best_model = best_result["model"]
    test_metrics = evaluate_on_test(best_model)
    plot_confusion_matrix(test_metrics["labels"], test_metrics["preds"], best_result["name"])

    total_training_time = sum(result["training_time"] for result in results)
    total_experiment_time = time.time() - experiment_start

    print("\nBest recurrent model")
    print("--------------------")
    for result in results:
        print(
            f"{result['name']}: best validation macro F1 = {result['best_val_macro_f1']:.4f}, "
            f"training time = {result['training_time']:.1f}s"
        )
    print(f"Selected model: {best_result['name']}")

    print("\nTest set metrics")
    print("----------------")
    print(f"Accuracy       : {test_metrics['accuracy']:.4f}")
    print(f"Macro F1       : {test_metrics['macro_f1']:.4f}")
    print(f"Inference time : {test_metrics['inference_time']:.2f}s")
    print(f"Training time  : {total_training_time:.1f}s")
    print(f"Experiment time: {total_experiment_time:.1f}s")

    print("\nPer-class F1")
    print("------------")
    for name, score in zip(label_names, test_metrics["per_class_f1"]):
        print(f"{name:<10}: {score:.4f}")

    print("\nDetailed classification report")
    print("------------------------------")
    print(
        classification_report(
            test_metrics["labels"],
            test_metrics["preds"],
            labels=range(NUM_LABELS),
            target_names=label_names,
            zero_division=0,
        )
    )

    print("Saved plots: rnn_training_curves.png, rnn_confusion_matrix.png")
