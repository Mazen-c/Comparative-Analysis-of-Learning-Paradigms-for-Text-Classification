# %% [Cell 1] Imports and experiment configuration
import json
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


# This notebook-style script evaluates an instruction-tuned LLM with no gradient updates.
# Default model for the assignment: microsoft/Phi-3-mini-4k-instruct.
# On a CPU-only machine, use FAST_DEV_RUN=1 and/or set MODEL_ID to a smaller local model.
MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "").lower() in {"1", "true", "yes"}
TEST_SAMPLE_LIMIT = int(os.getenv("TEST_SAMPLE_LIMIT", "24" if FAST_DEV_RUN else "0"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "96"))
COT_MAX_NEW_TOKENS = int(os.getenv("COT_MAX_NEW_TOKENS", "160"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_4BIT = (
    DEVICE.type == "cuda"
    and BitsAndBytesConfig is not None
    and os.getenv("DISABLE_4BIT", "").lower() not in {"1", "true", "yes"}
)

OUTPUT_DIR = Path("llm_prompting_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
label_to_id = {label: index for index, label in enumerate(label_names)}

print(f"Using device: {DEVICE}")
print(f"Model id: {MODEL_ID}")
print(f"4-bit quantization enabled: {USE_4BIT}")


# %% [Cell 2] Load the labeled emotion dataset without train-time updates
def load_emotion_dataset():
    """Load cached Arrow splits first, then fall back to Hugging Face if needed."""
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "dair-ai___emotion" / "split" / "0.0.0"
    if cache_root.exists():
        for revision_dir in cache_root.iterdir():
            split_paths = {
                "train": revision_dir / "emotion-train.arrow",
                "validation": revision_dir / "emotion-validation.arrow",
                "test": revision_dir / "emotion-test.arrow",
            }
            if all(path.exists() for path in split_paths.values()):
                print(f"Loading cached dair-ai/emotion Arrow files from: {revision_dir}")
                return DatasetDict(
                    {split: HFDataset.from_file(str(path)) for split, path in split_paths.items()}
                )

    print("Cached Arrow files not found; loading dair-ai/emotion through Hugging Face.")
    return load_dataset("dair-ai/emotion")


dataset = load_emotion_dataset()
train_split = dataset["train"]
test_split = dataset["test"]
if TEST_SAMPLE_LIMIT > 0:
    test_split = test_split.select(range(min(TEST_SAMPLE_LIMIT, len(test_split))))

print(f"Train examples available for few-shot prompts: {len(train_split):,}")
print(f"Test examples evaluated: {len(test_split):,}")


# %% [Cell 3] Model architecture and pretraining explanation for the notebook
# Assignment note: clearly explain architecture and pretraining objective.
# Source: Hugging Face model card for microsoft/Phi-3-mini-4k-instruct.
model_explanation = f"""
## Pretrained Instruction-Tuned LLM Used

Model: `{MODEL_ID}`

For the main assignment run, this script is configured for `microsoft/Phi-3-mini-4k-instruct`.
Phi-3 Mini-4K-Instruct is a 3.8B-parameter dense decoder-only Transformer language model
with a 4K-token context window. It is pretrained as an autoregressive causal language model,
meaning it learns to predict the next token from previous tokens. Its training data combines
filtered public web/code/educational text, synthetic textbook-like data, and chat-format data.
The instruction-tuned version is further aligned with supervised fine-tuning and Direct
Preference Optimization, which improves instruction following and safer assistant behavior.

In this experiment, the model is used only for inference. No gradients are computed, no model
weights are updated, and the classifier behavior comes entirely from prompting.
"""
print(model_explanation)


# %% [Cell 4] Load tokenizer and model with optional 4-bit quantization
def load_instruction_model(model_id):
    """Load an instruction-tuned causal LM for inference only."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if USE_4BIT:
        # Recommended for GPU memory efficiency when bitsandbytes is installed.
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif DEVICE.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not USE_4BIT:
        model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


tokenizer, model = load_instruction_model(MODEL_ID)


# %% [Cell 5] Prompt templates for zero-shot, few-shot, and chain-of-thought
def format_examples(num_examples):
    """Use labeled training examples only inside the few-shot prompt."""
    lines = []
    for row in train_split.select(range(num_examples)):
        lines.append(f"Text: {row['text']}\nLabel: {label_names[row['label']]}")
    return "\n\n".join(lines)


def build_prompt(text, setting):
    label_list = ", ".join(label_names)
    base_instruction = (
        "Classify the emotion expressed in the text. "
        f"Choose exactly one label from this list: {label_list}. "
        "Return the answer in the format: Final label: <label>."
    )

    if setting == "zero_shot":
        return f"{base_instruction}\n\nText: {text}\nFinal label:"

    if setting == "few_shot_3":
        return (
            f"{base_instruction}\n\n"
            "Here are labeled examples:\n"
            f"{format_examples(3)}\n\n"
            f"Text: {text}\nFinal label:"
        )

    if setting == "few_shot_8":
        return (
            f"{base_instruction}\n\n"
            "Here are labeled examples:\n"
            f"{format_examples(8)}\n\n"
            f"Text: {text}\nFinal label:"
        )

    if setting == "chain_of_thought":
        return (
            "Classify the emotion expressed in the text. "
            f"Choose exactly one label from this list: {label_list}.\n"
            "Reason step by step about the emotional cues, then end with exactly one line "
            "in this format: Final label: <label>.\n\n"
            f"Text: {text}\nReasoning:"
        )

    raise ValueError(f"Unknown setting: {setting}")


settings = ["zero_shot", "few_shot_3", "few_shot_8", "chain_of_thought"]


# %% [Cell 6] Generation and label parsing
def generate_raw_output(prompt, setting):
    """Generate model text without computing gradients."""
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    max_tokens = COT_MAX_NEW_TOKENS if setting == "chain_of_thought" else MAX_NEW_TOKENS

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_label(raw_output):
    """Extract the final emotion label from raw generated text."""
    normalized = raw_output.lower()
    final_match = re.search(r"final\s+label\s*:\s*(sadness|joy|love|anger|fear|surprise)", normalized)
    if final_match:
        return final_match.group(1)

    for label in label_names:
        if re.search(rf"\b{label}\b", normalized):
            return label

    # If parsing fails, use a deterministic fallback so metrics still run.
    return "sadness"


# %% [Cell 7] Evaluate one prompting setting
def evaluate_setting(setting):
    """Evaluate accuracy, macro F1, confusion matrix, raw examples, and inference time."""
    true_labels = []
    pred_labels = []
    examples = []

    start_time = time.time()
    for index, row in enumerate(test_split):
        prompt = build_prompt(row["text"], setting)
        raw_output = generate_raw_output(prompt, setting)
        predicted_label = parse_label(raw_output)

        true_label_id = row["label"]
        pred_label_id = label_to_id[predicted_label]
        true_labels.append(true_label_id)
        pred_labels.append(pred_label_id)

        # Show 3-5 prompt/output examples in the notebook for each setting.
        if len(examples) < 5:
            examples.append(
                {
                    "index": index,
                    "true_label": label_names[true_label_id],
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed_label": predicted_label,
                }
            )

        if (index + 1) % 25 == 0:
            print(f"{setting}: evaluated {index + 1}/{len(test_split)} examples")

    inference_time = time.time() - start_time
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(
        true_labels,
        pred_labels,
        labels=range(len(label_names)),
        average="macro",
        zero_division=0,
    )

    return {
        "setting": setting,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "inference_time": inference_time,
        "examples": examples,
    }


# %% [Cell 8] Confusion matrix plotting
def plot_confusion_matrix(result):
    cm = confusion_matrix(result["true_labels"], result["pred_labels"], labels=range(len(label_names)))
    setting = result["setting"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title(f"LLM Prompting Confusion Matrix - {setting}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"llm_confusion_matrix_{setting}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# %% [Cell 9] Run all prompting settings
all_results = []
for setting in settings:
    print(f"\nEvaluating setting: {setting}")
    result = evaluate_setting(setting)
    result["confusion_matrix_path"] = str(plot_confusion_matrix(result))
    all_results.append(result)


# %% [Cell 10] Report metrics and show raw prompts/outputs
summary_rows = []
examples_for_export = {}

for result in all_results:
    summary_rows.append(
        {
            "setting": result["setting"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "inference_time_seconds": result["inference_time"],
            "confusion_matrix": result["confusion_matrix_path"],
        }
    )
    examples_for_export[result["setting"]] = result["examples"]

print("\nPrompting results")
print("-----------------")
for row in summary_rows:
    print(
        f"{row['setting']:<16} | "
        f"Accuracy: {row['accuracy']:.4f} | "
        f"Macro F1: {row['macro_f1']:.4f} | "
        f"Inference time: {row['inference_time_seconds']:.2f}s | "
        f"CM: {row['confusion_matrix']}"
    )

print("\nExample prompts and raw outputs")
print("-------------------------------")
for setting, examples in examples_for_export.items():
    print(f"\n### {setting}")
    for example in examples[:5]:
        print(f"\nTrue label: {example['true_label']}")
        print("Prompt:")
        print(example["prompt"])
        print("Raw output:")
        print(example["raw_output"])
        print(f"Parsed label: {example['parsed_label']}")

with open(OUTPUT_DIR / "llm_prompting_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary_rows, file, indent=2)

with open(OUTPUT_DIR / "llm_prompt_examples.json", "w", encoding="utf-8") as file:
    json.dump(examples_for_export, file, indent=2)

print(f"\nSaved outputs in: {OUTPUT_DIR}")
