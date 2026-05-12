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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def env_int(name, default):
    return int(os.getenv(name, str(default)))


# This notebook-style script evaluates an instruction-tuned LLM with no gradient updates.
# Phi-3 is the intended assignment-scale model, but CPU notebooks need a small default.
ASSIGNMENT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
CPU_FALLBACK_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_ID = os.getenv(
    "MODEL_ID",
    ASSIGNMENT_MODEL_ID if DEVICE.type == "cuda" else CPU_FALLBACK_MODEL_ID,
)
FAST_DEV_RUN = env_flag("FAST_DEV_RUN")
HF_LOCAL_FILES_ONLY = env_flag("HF_LOCAL_FILES_ONLY", default=DEVICE.type == "cpu")
DEFAULT_TEST_SAMPLE_LIMIT = 24 if FAST_DEV_RUN or DEVICE.type == "cpu" else 0
TEST_SAMPLE_LIMIT = env_int("TEST_SAMPLE_LIMIT", DEFAULT_TEST_SAMPLE_LIMIT)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 8)
COT_MAX_NEW_TOKENS = env_int("COT_MAX_NEW_TOKENS", 64)

USE_4BIT = (
    DEVICE.type == "cuda"
    and BitsAndBytesConfig is not None
    and not env_flag("DISABLE_4BIT")
)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "llm_prompting_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
label_to_id = {label: index for index, label in enumerate(label_names)}

print(f"Using device: {DEVICE}")
print(f"Model id: {MODEL_ID}")
print(f"Local files only: {HF_LOCAL_FILES_ONLY}")
print(f"Test sample limit: {TEST_SAMPLE_LIMIT or 'full test split'}")
print(f"Output dir: {OUTPUT_DIR}")
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
MODEL_NOTES = {
    ASSIGNMENT_MODEL_ID: """
For the assignment-scale run, use `microsoft/Phi-3-mini-4k-instruct`.
Phi-3 Mini-4K-Instruct is a 3.8B-parameter dense decoder-only Transformer language model
with a 4K-token context window. It is pretrained as an autoregressive causal language model,
meaning it learns to predict the next token from previous tokens. Its training data combines
filtered public web/code/educational text, synthetic textbook-like data, and chat-format data.
The instruction-tuned version is further aligned with supervised fine-tuning and Direct
Preference Optimization, which improves instruction following and safer assistant behavior.
""",
    CPU_FALLBACK_MODEL_ID: """
For a CPU-safe notebook run, this section uses `HuggingFaceTB/SmolLM2-135M-Instruct`.
SmolLM2-135M-Instruct is a compact decoder-only causal language model from the SmolLM2
family. Like Phi-3, it is pretrained with a next-token prediction objective and then adapted
for instruction following. It is much smaller than Phi-3, so it is appropriate for validating
the prompting pipeline locally, while Phi-3 remains the stronger assignment-scale model when
GPU memory and the model weights are available.
""",
}

model_note = MODEL_NOTES.get(
    MODEL_ID,
    """
This experiment uses the instruction-tuned causal language model named by `MODEL_ID`.
It is used as a decoder-only language model for prompt-based classification: the prompt
constrains generation to one of the six emotion labels, and the generated text is parsed
into a class prediction.
""",
).strip()

model_explanation = f"""
## Pretrained Instruction-Tuned LLM Used

Model: `{MODEL_ID}`

{model_note}

In this experiment, the model is used only for inference. No gradients are computed, no model
weights are updated, and the classifier behavior comes entirely from prompting.
"""
print(model_explanation)


# %% [Cell 4] Load tokenizer and model with optional 4-bit quantization
def load_instruction_model(model_id):
    """Load an instruction-tuned causal LM for inference only."""
    print(f"Loading tokenizer/model for {model_id}...")
    common_kwargs = {
        "trust_remote_code": True,
        "local_files_only": HF_LOCAL_FILES_ONLY,
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **common_kwargs)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load tokenizer for {model_id!r}. "
            "The notebook is in local-files-only mode on CPU to avoid hanging on a large download. "
            "Use the cached CPU model, set MODEL_ID to an already downloaded model, or set "
            "HF_LOCAL_FILES_ONLY=0 if you intentionally want Hugging Face to download files."
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(common_kwargs)
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

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load model weights for {model_id!r}. "
            "If you see a stuck 'Fetching 2 files' progress bar, the selected model weights are "
            "not fully cached. On CPU, keep the default SmolLM2 model or set MODEL_ID to another "
            "small cached instruct model. For Phi-3, run on a machine with enough GPU/RAM and "
            "allow the full weights to download first."
        ) from exc

    if not USE_4BIT:
        model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


tokenizer, model = load_instruction_model(MODEL_ID)


# %% [Cell 5] Prompt templates for zero-shot, few-shot, and chain-of-thought
def select_balanced_examples(num_examples):
    """Select deterministic few-shot examples with broader label coverage."""
    chosen_rows = []
    seen_labels = set()

    for row in train_split:
        label_id = row["label"]
        if label_id not in seen_labels:
            chosen_rows.append(row)
            seen_labels.add(label_id)
        if len(chosen_rows) == min(num_examples, len(label_names)):
            break

    if len(chosen_rows) < num_examples:
        used_texts = {row["text"] for row in chosen_rows}
        for row in train_split:
            if row["text"] in used_texts:
                continue
            chosen_rows.append(row)
            used_texts.add(row["text"])
            if len(chosen_rows) == num_examples:
                break

    return chosen_rows[:num_examples]


def format_examples(num_examples):
    """Use labeled training examples only inside the few-shot prompt."""
    lines = []
    for row in select_balanced_examples(num_examples):
        lines.append(f"Text: {row['text']}\nFinal label: {label_names[row['label']]}")
    return "\n\n".join(lines)


def build_prompt(text, setting):
    label_list = ", ".join(label_names)
    base_instruction = (
        "Classify the emotion expressed in the text. "
        f"Choose exactly one label from this list: {label_list}. "
        "Return only one line in the format: Final label: <label>. "
        "Do not write new examples, explanations, or extra text."
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
            "Reason briefly about the emotional cues, then end with exactly one line "
            "in this format: Final label: <label>.\n\n"
            f"Text: {text}\nReasoning:"
        )

    raise ValueError(f"Unknown setting: {setting}")


settings = ["zero_shot", "few_shot_3", "few_shot_8", "chain_of_thought"]


# %% [Cell 6] Generation and label parsing
class StopAfterFinalLabel(StoppingCriteria):
    """Stop once the generated answer has completed the required final-label line."""

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.pattern = re.compile(
            r"final\s+label\s*:\s*(sadness|joy|love|anger|fear|surprise)\s*(?:\n|$)",
            re.IGNORECASE,
        )

    def __call__(self, input_ids, scores, **kwargs):
        new_tokens = input_ids[0][self.prompt_length :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return bool(self.pattern.search(text))


def encode_prompt_for_model(prompt):
    """Wrap the plain prompt in the model's chat format when available."""
    system_prompt = (
        "You are an emotion-classification model. Follow the user's requested output "
        "format exactly and classify using only the allowed labels."
    )

    if getattr(tokenizer, "chat_template", None):
        chat_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=3072)

    return tokenizer(
        f"{system_prompt}\n\n{prompt}",
        return_tensors="pt",
        truncation=True,
        max_length=3072,
    )


def generate_raw_output(prompt, setting):
    """Generate model text without computing gradients."""
    encoded = encode_prompt_for_model(prompt)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    max_tokens = COT_MAX_NEW_TOKENS if setting == "chain_of_thought" else MAX_NEW_TOKENS
    stopping_criteria = StoppingCriteriaList(
        [StopAfterFinalLabel(tokenizer, encoded["input_ids"].shape[1])]
    )

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_label(raw_output):
    """Extract the final emotion label from raw generated text."""
    normalized = raw_output.lower()
    final_match = re.search(r"final\s+label\s*:\s*(sadness|joy|love|anger|fear|surprise)", normalized)
    if final_match:
        return final_match.group(1)

    # If the model ignored the requested prefix but answered with a label first,
    # take that first generated label, not a later label from hallucinated examples.
    first_label_match = re.search(r"\b(sadness|joy|love|anger|fear|surprise)\b", normalized)
    if first_label_match:
        return first_label_match.group(1)

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
