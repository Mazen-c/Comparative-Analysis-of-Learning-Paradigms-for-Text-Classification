# %% [Cell 1] Imports
import os
import time
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# %% [Cell 2] Device setup and dataset loading
# Use GPU if available; BERT fine-tuning is very slow on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NUM_LABELS  = len(label_names)


def env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Ignoring invalid {name}={value!r}; using {default}.")
        return default


FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "").lower() in {"1", "true", "yes"}
BATCH_SIZE = env_int("BATCH_SIZE", 8 if FAST_DEV_RUN else 32)
TRAIN_SAMPLE_LIMIT = env_int("TRAIN_SAMPLE_LIMIT", 64 if FAST_DEV_RUN else 0)
VAL_SAMPLE_LIMIT = env_int("VAL_SAMPLE_LIMIT", 32 if FAST_DEV_RUN else 0)
TEST_SAMPLE_LIMIT = env_int("TEST_SAMPLE_LIMIT", 32 if FAST_DEV_RUN else 0)

# Load the same emotion dataset used in Part 1
dataset = load_dataset("dair-ai/emotion")


# %% [Cell 3] Tokenizer and PyTorch Dataset class
# bert-base-uncased uses WordPiece tokenization with a 30k-token vocabulary
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class EmotionDataset(Dataset):
    def __init__(self, split, sample_limit=0):
        split_dataset = dataset[split]
        if sample_limit > 0:
            split_dataset = split_dataset.select(range(min(sample_limit, len(split_dataset))))

        self.texts = list(split_dataset["text"])
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(split_dataset["label"]), dtype=torch.long)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }


# %% [Cell 4] DataLoaders — batch and shuffle the three splits
# Shuffle training data each epoch to prevent ordering bias
train_loader = DataLoader(EmotionDataset("train", TRAIN_SAMPLE_LIMIT), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(EmotionDataset("validation", VAL_SAMPLE_LIMIT), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(EmotionDataset("test", TEST_SAMPLE_LIMIT), batch_size=BATCH_SIZE, shuffle=False)


# %% [Cell 5] Load pre-trained BERT and attach a classification head
# AutoModelForSequenceClassification replaces BERT's MLM head with a linear
# classifier that outputs one logit per emotion class
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS
).to(device)

# Sanity-check model size: all ~110M parameters are trainable (full fine-tuning)
print(f"Total parameters    : {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# %% [Cell 6] Optimizer and learning-rate scheduler
EPOCHS = env_int("EPOCHS", 1 if FAST_DEV_RUN else 3)
LR     = 2e-5  # standard BERT fine-tuning range: 1e-5 to 5e-5

# AdamW (Adam + decoupled weight decay) is the standard choice for BERT
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS

# Linear warmup for the first 10% of steps, then linear decay to 0
# Warmup prevents large gradient updates that could destroy pre-trained weights
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# %% [Cell 7] Training / evaluation loop function
def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    # Disable gradient computation during validation to save memory and time
    with torch.set_grad_enabled(training):
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Forward pass; HuggingFace models compute cross-entropy loss internally
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients in deep transformers
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            # Predicted class = index of the highest logit
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# %% [Cell 8] Run training loop and record metrics per epoch
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

if __name__ == "__main__":
    print("\nTraining...\n")
    train_start = time.time()

    for epoch in range(EPOCHS):
        # Full pass over training data (with gradient updates)
        t_loss, t_acc = run_epoch(train_loader, training=True)
        # Full pass over validation data (no gradient updates)
        v_loss, v_acc = run_epoch(val_loader,   training=False)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} — "
              f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}")

    train_time = time.time() - train_start
    print(f"\nTotal training time: {train_time:.1f}s")


# %% [Cell 9] Plot loss and accuracy curves
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 4))

# Loss subplot — convergence check; val loss rising signals overfitting
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses,   label="Val Loss")
plt.title("Loss Curves — BERT Fine-Tuning")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy subplot — visual confirmation that val acc tracks train acc
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label="Train Acc")
plt.plot(epochs_range, val_accs,   label="Val Acc")
plt.title("Accuracy Curves — BERT Fine-Tuning")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("bert_training_curves.png", dpi=150)
plt.close()


# %% [Cell 10] Evaluate on the held-out test set
model.eval()
all_preds, all_labels = [], []

infer_start = time.time()

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # No labels passed here — we only need logits for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = outputs.logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

infer_time = time.time() - infer_start

# Macro F1 treats each class equally regardless of support — important for
# imbalanced classes like "surprise" and "love"
accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, labels=range(NUM_LABELS), average="macro", zero_division=0)

print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"Macro F1      : {macro_f1:.4f}")
print(f"Inference Time: {infer_time:.2f}s\n")
print(classification_report(
    all_labels,
    all_preds,
    labels=range(NUM_LABELS),
    target_names=label_names,
    zero_division=0
))


# %% [Cell 11] Confusion matrix — per-class error analysis
cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_LABELS))

plt.figure(figsize=(8, 6))
# annot=True prints raw counts in each cell; fmt="d" keeps them as integers
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix — BERT Fine-Tuned")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("bert_confusion_matrix.png", dpi=150)
plt.close()


# %% [Cell 12] Model summary and reflection
summary = f"""
## Part 3 — Model Summary

| Item                  | Detail                                                         |
|-----------------------|----------------------------------------------------------------|
| Model                 | bert-base-uncased                                              |
| Total parameters      | ~110 million                                                   |
| Trainable parameters  | ~110 million (all layers unfrozen)                             |
| Pretraining objective | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)|
| Fine-tuning epochs    | {EPOCHS}                                                       |
| Learning rate         | {LR}                                                           |
| Test Accuracy         | {accuracy:.4f}                                                 |
| Macro F1              | {macro_f1:.4f}                                                 |
| Training Time         | {train_time:.1f}s                                              |
| Inference Time        | {infer_time:.2f}s                                              |

### Why does pretraining help?
BERT was pretrained on BookCorpus and English Wikipedia using MLM, which forced it to learn
deep bidirectional context — understanding that a word's meaning depends on both what comes
before and after it. When we fine-tune, we are not teaching BERT what language is; we are
only redirecting its existing knowledge toward our six emotion labels. The from-scratch RNN
in Part 2 must learn language structure and the classification task simultaneously from only
16,000 examples, which is why BERT almost always wins on accuracy and especially on minority
classes like surprise and love.
"""
print(summary)
