#@title { vertical-output: true }
# ════════════════════════════════════════════════════════════════
# Text-to-SQL Training – DeepSeek Coder 1.3B (QLoRA fine-tuning)
# ════════════════════════════════════════════════════════════════

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU device: {gpu_name}")
    print(f"Memory: {gpu_memory:.2f} GB")

# Show GPU info
!nvidia-smi

# Install required packages
!pip uninstall -y -q transformers tokenizers peft > /dev/null 2>&1
!pip install --no-cache-dir -q \
        "transformers[torch]==4.51.3" \
        "tokenizers>=0.19.0" \
        "datasets>=2.18.0" \
        "accelerate>=0.28.0" \
        "tqdm>=4.66.0" \
        "bitsandbytes>=0.42.0" \
        "peft>=0.7.0"

import pathlib, sys, importlib, torch
if "transformers" in sys.modules: del sys.modules["transformers"]
import transformers, datasets, accelerate
from tqdm.auto import tqdm
print("transformers:", transformers.__version__)
print("datasets    :", datasets.__version__)
print("accelerate  :", accelerate.__version__)
print("file        :", pathlib.Path(transformers.__file__).as_posix())

# --- Verify Installation ---
import importlib, pkg_resources, sys, subprocess
import torch
import transformers, datasets, accelerate
import json
import os
import pandas as pd
import time
import numpy as np
import pathlib
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    default_data_collator,
    set_seed
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

set_seed(42)

# --- Configuration ---
SCHEMA_FORMAT = "sql"
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
EXPERIMENT_NAME = "deepseek_coder_1.3b_qlora"
MODEL_LABEL = MODEL_NAME.split("/")[-1]

# Special tokens for SQL
SQL_START_TOKEN = "<SQL_START>"
SQL_END_TOKEN = "<SQL_END>"

# Training hyper-parameters
EPOCHS = 4
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.10
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
MAX_GRAD_NORM = 1.0
RESUME_FROM_CHECKPOINT = False

# Mixed precision
LOAD_IN_4BIT = True
USE_BF16 = True

# LoRA Parameter
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODS = ["q_proj", "k_proj", "v_proj",
               "o_proj", "gate_proj",
               "up_proj", "down_proj"]

# Google Drive paths
DRIVE_BASE_DIR = "/content/drive/MyDrive/text2sql"
DRIVE_OUTPUT_DIR = f"{DRIVE_BASE_DIR}/{EXPERIMENT_NAME}"
DRIVE_DATASET_SOURCE_DIR = f"{DRIVE_BASE_DIR}/datasets/spider"
DRIVE_LOGS_DIR = f"{DRIVE_BASE_DIR}/logs/{EXPERIMENT_NAME}"

# Local paths
LOCAL_DATASET_DIR = "/content/datasets/spider"

# Create directories
os.makedirs(DRIVE_BASE_DIR, exist_ok=True)
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
os.makedirs(DRIVE_LOGS_DIR, exist_ok=True)
os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

print(f"--- Running Experiment: {EXPERIMENT_NAME} ---")
print(f"Model: {MODEL_NAME}")
print(f"Schema Format: {SCHEMA_FORMAT} (with Types)")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Per Device Batch Size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Gradient Clipping: {MAX_GRAD_NORM}")
print(f"Max Input Length: {MAX_INPUT_LENGTH}")
print(f"Max Target Length: {MAX_TARGET_LENGTH}")
print(f"Warmup Ratio: {WARMUP_RATIO}")
print(f"4-bit Quantization: {LOAD_IN_4BIT}")
print(f"BF16 Enabled: {USE_BF16}")
print(f"LoRA Rank: {LORA_R}")
print(f"LoRA Alpha: {LORA_ALPHA}")
print(f"LoRA Dropout: {LORA_DROPOUT}")
print(f"Target Modules: {TARGET_MODS}")

# Clear CUDA cache
torch.cuda.empty_cache()

# --- Schema Utilities ---
def load_tables_json(tables_path):
    """Load the tables.json file containing schema information."""
    full_path = os.path.join(LOCAL_DATASET_DIR, tables_path)
    print(f"Loading tables.json from: {full_path}")
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: tables.json not found at {full_path}")
        raise
    db_schemas = {db_info['db_id']: db_info for db_info in tables_data}
    return db_schemas

def get_sql_schema_string(db_id, db_schemas):
    """Create SQL schema string including types, PKs, FKs."""
    if db_id not in db_schemas: raise ValueError(f"DB ID '{db_id}' not found")
    schema_info = db_schemas[db_id]
    tables = schema_info['table_names_original']
    columns = schema_info['column_names_original']
    column_types = schema_info['column_types']
    primary_keys = set(schema_info.get('primary_keys', []))
    fk_dict = {}
    if isinstance(schema_info.get('foreign_keys'), list):
        for fk_pair in schema_info['foreign_keys']:
             if isinstance(fk_pair, (list, tuple)) and len(fk_pair) == 2:
                 col1_idx, col2_idx = fk_pair
                 if isinstance(col1_idx, int) and isinstance(col2_idx, int): fk_dict[col1_idx] = col2_idx
    table_defs = []
    for i, table in enumerate(tables):
        table_columns = []
        for col_idx, (tab_idx, col_name) in enumerate(columns):
            if tab_idx == i:
                col_type = column_types[col_idx].upper()
                col_info = f"{col_name} ({col_type})"
                if col_idx in primary_keys: col_info += " (PK)"
                if col_idx in fk_dict:
                    ref_col_idx = fk_dict[col_idx]
                    if 0 <= ref_col_idx < len(columns):
                         ref_tab_idx, ref_col_name = columns[ref_col_idx]
                         if 0 <= ref_tab_idx < len(tables):
                              ref_table = tables[ref_tab_idx]
                              col_info += f" (FK→{tables[ref_tab_idx]}.{ref_col_name})"
                table_columns.append(col_info)
        if table_columns:
            table_columns.sort()
            table_def = f"Table: {table}\nColumns: {', '.join(table_columns)}"
            table_defs.append(table_def)
    table_defs.sort()
    return "\n".join(table_defs)

def create_deepseek_prompt(example, db_schemas):
    """Create prompt with SQL_START token."""
    schema = get_sql_schema_string(example['db_id'], db_schemas)
    if len(schema.split()) > 800:
        print(f"Warning: Schema for {example['db_id']} exceeds 800 tokens!")
    return (
        f"### Text-to-SQL task\n"
        f"Database: {example['db_id']}\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {example['question']}\n\n"
        f"{SQL_START_TOKEN}\n"
    )

def create_training_sample(example, db_schemas):
    """Prepare final training sample with prompt and target."""
    prompt = create_deepseek_prompt(example, db_schemas)
    target = example['query'] + f"\n{SQL_END_TOKEN}"
    return {"prompt": prompt, "target": target}

def prepare_dataset(examples, db_schemas):
    """Process a batch of examples."""
    processed = []
    for example in examples:
        processed.append(create_training_sample(example, db_schemas))
    return processed

# --- Setup Local Dataset from Drive ---
def setup_local_dataset_from_drive():
    """Copy the Spider dataset JSON files from Google Drive to local storage."""
    print(f"\n--- Setting up Dataset ---")
    print(f"Copying dataset from Google Drive path: {DRIVE_DATASET_SOURCE_DIR}")
    drive_tables_path = f"{DRIVE_DATASET_SOURCE_DIR}/tables.json"
    drive_train_path = f"{DRIVE_DATASET_SOURCE_DIR}/train_spider.json"
    drive_dev_path = f"{DRIVE_DATASET_SOURCE_DIR}/dev.json"
    all_paths_exist = all(os.path.exists(p) for p in [drive_tables_path, drive_train_path, drive_dev_path])
    if all_paths_exist:
        print("Copying dataset files to local Colab storage...")
        try:
            !cp -v "{drive_tables_path}" "{LOCAL_DATASET_DIR}/"
            !cp -v "{drive_train_path}" "{LOCAL_DATASET_DIR}/"
            !cp -v "{drive_dev_path}" "{LOCAL_DATASET_DIR}/"
            print("Dataset files copied successfully.")
        except Exception as e:
             print(f"Error during file copy: {e}")
             raise
    else:
        missing = [p for p in [drive_tables_path, drive_train_path, drive_dev_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing files in Drive: {missing}")

# --- Run Setup Function ---
setup_local_dataset_from_drive()

# --- Load Database Schemas ---
print("\nLoading database schemas...")
db_schemas = load_tables_json('tables.json')
print(f"Loaded schemas for {len(db_schemas)} databases.")

# --- Load and Prepare Datasets ---
def load_and_prepare_data(file_path, db_schemas_dict):
    """Load and prepare dataset."""
    actual_file_path = os.path.join(LOCAL_DATASET_DIR, file_path)
    print(f"Loading data from: {actual_file_path}")
    try:
        with open(actual_file_path, 'r', encoding='utf-8') as f:
            spider_data = json.load(f)

        processed_data = prepare_dataset(spider_data, db_schemas_dict)
        dataset = Dataset.from_list(processed_data)
        print(f"Prepared {len(dataset)} examples from {actual_file_path}.")
        return dataset
    except Exception as e:
        print(f"Error processing data from {actual_file_path}: {e}")
        raise

print("Loading datasets...")
train_dataset = load_and_prepare_data('train_spider.json', db_schemas)
dev_dataset = load_and_prepare_data('dev.json', db_schemas)
if train_dataset is None or dev_dataset is None:
     raise ValueError("Failed to load train or dev dataset.")

# Log some examples to verify prompts
print("\nSample Prompts:")
for i in range(min(2, len(train_dataset))):
    print(f"\n--- Example {i+1} ---")
    print(f"Input length: {len(train_dataset[i]['prompt'])} characters")
    print(f"Prompt: {train_dataset[i]['prompt'][:500]}...")
    print(f"Target: {train_dataset[i]['target']}")

# --- Create Tokenizer and Model ---
print(f"\nLoading tokenizer and model: {MODEL_NAME}...")

# Initialize tokenizer with special tokens
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",  # Richtig für Decoder-Modelle
    use_fast=True
)

# Add special tokens
special_tokens = {"additional_special_tokens": [SQL_START_TOKEN, SQL_END_TOKEN]}
tokenizer.add_special_tokens(special_tokens)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Resize token embeddings to match tokenizer with special tokens
model.resize_token_embeddings(len(tokenizer))

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure & add LoRA adapter
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=TARGET_MODS,
    task_type="CAUSAL_LM"
)

# Create peft model with LoRA adapter
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

print("Model and tokenizer loaded.")
print(f"Special SQL tokens added: {SQL_START_TOKEN}, {SQL_END_TOKEN}")

# --- Tokenization and DataCollator ---
class SQLLabelMasker:
    def __init__(self, tokenizer, max_input_length=1024, max_target_length=256):
        self.tok  = tokenizer
        self.max_in, self.max_tgt = max_input_length, max_target_length

    def __call__(self, batch):
        prompts = batch["prompt"]
        targets = batch["target"]

        texts = [p + t for p, t in zip(prompts, targets)]

        tok_out = self.tok(
            texts,
            max_length=self.max_in + self.max_tgt,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids      = tok_out.input_ids
        attention_mask = tok_out.attention_mask
        labels         = input_ids.clone()

        sql_start_id = self.tok.convert_tokens_to_ids(SQL_START_TOKEN)
        for i, ids in enumerate(input_ids):
            pos = (ids == sql_start_id).nonzero(as_tuple=False)
            if len(pos):
                labels[i, : pos[0, 0].item() + 1] = -100
            else:
                labels[i].fill_(-100)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

# Initialize tokenize function
tokenize_function = SQLLabelMasker(tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

# Tokenize datasets
print("\nTokenizing datasets...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training dataset"
)

tokenized_dev = dev_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dev_dataset.column_names,
    desc="Tokenizing development dataset"
)

print(f"Training dataset tokenized: {len(tokenized_train)} examples")
print(f"Development dataset tokenized: {len(tokenized_dev)} examples")

# --- Training Arguments ---
print("\nConfiguring training arguments...")
steps_per_epoch = len(tokenized_train) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

training_args = TrainingArguments(
    output_dir=DRIVE_OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,
    bf16=USE_BF16,
    fp16=False,
    gradient_checkpointing=True,
    max_grad_norm=MAX_GRAD_NORM,
    logging_steps=50,
    save_strategy="steps",
    save_steps=steps_per_epoch,
    #eval_strategy="steps",
    #eval_steps=steps_per_epoch,
    logging_dir=DRIVE_LOGS_DIR,
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    optim="paged_adamw_8bit",
    lr_scheduler_type="linear",
    seed=42,
    remove_unused_columns=False,
)

# --- Trainer Initialization ---
print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=default_data_collator,  # Unser eigener Collator
)
print("Trainer initialized.")

# --- Monitor for loss changes ---
class LossMonitorCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and logs['loss'] > 10:
            print(f"WARNING: High loss detected: {logs['loss']}")
        elif logs and 'loss' in logs and np.isnan(logs['loss']):
            print("WARNING: NaN loss detected!")

# Callback hinzufügen
trainer.add_callback(LossMonitorCallback())

# --- Main Training Execution ---
print(f"\n--- Starting Training ---")
start_time = time.time()
checkpoint = None
if RESUME_FROM_CHECKPOINT:
    if os.path.isdir(DRIVE_OUTPUT_DIR):
        checkpoints = [os.path.join(DRIVE_OUTPUT_DIR, d) for d in os.listdir(DRIVE_OUTPUT_DIR) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(DRIVE_OUTPUT_DIR, d))]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = latest_checkpoint
        else: print(f"No checkpoint found in {DRIVE_OUTPUT_DIR}.")
    else: print(f"Output directory {DRIVE_OUTPUT_DIR} does not exist.")

try:
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ---------- post-training actions ----------
    print("\n--- Training Finished ---")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\nSaving the final model…")
    # Mit QLoRA speichern wir nur den Adapter
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")

except KeyboardInterrupt:
    print("Training interrupted by user.")
    raise

except Exception as e:
    print("Training failed:", e)
    raise

# ----------------------------------------------------------
#  Generate training report
# ----------------------------------------------------------
from pathlib import Path
model_label = Path(MODEL_NAME).name

elapsed      = time.time() - start_time
h, rem       = divmod(elapsed, 3600)
m, s         = divmod(rem, 60)

report_path  = os.path.join(training_args.output_dir, "training_summary.txt")

try:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment          : {EXPERIMENT_NAME}\n")
        f.write(f"Model               : {model_label}\n")
        f.write(f"Schema Format       : {SCHEMA_FORMAT} (with types)\n")
        f.write(f"4-bit Quantization  : {LOAD_IN_4BIT}\n")
        f.write(f"bf16 Enabled        : {training_args.bf16}\n")
        f.write(f"Epochs Configured   : {EPOCHS}\n")
        f.write(f"Epochs Trained      : {metrics.get('epoch', 0):.2f}\n")
        f.write(f"Training Time       : {int(h)}h {int(m)}m {int(s)}s\n")
        f.write(f"Learning Rate       : {LEARNING_RATE}\n")
        f.write(f"Batch Size / Device : {PER_DEVICE_TRAIN_BATCH_SIZE}\n")
        f.write(f"Grad Accum Steps    : {GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Effective BatchSize : {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Weight Decay        : {WEIGHT_DECAY}\n")
        f.write(f"Warm-up Ratio       : {WARMUP_RATIO} (≈{warmup_steps} steps)\n")
        f.write(f"Max Grad-Norm       : {MAX_GRAD_NORM}\n")
        f.write(f"Max Input Length    : {MAX_INPUT_LENGTH}\n")
        f.write(f"Max Target Length   : {MAX_TARGET_LENGTH}\n")
        f.write(f"LoRA Rank           : {LORA_R}\n")
        f.write(f"LoRA Alpha          : {LORA_ALPHA}\n")
        f.write(f"LoRA Dropout        : {LORA_DROPOUT}\n")

        # ---- training metrics ----
        f.write("\nTraining Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k:<25} : {v}\n")

        f.write(f"\nModel saved to: {training_args.output_dir}\n")
        f.write(f"Model size: LoRA adapter (~{LORA_R * len(TARGET_MODS) * 2 * 4 / 1024:.2f} MB)\n")
        f.write("To evaluate the model, load both base model and adapter.\n")

    print(f"Training summary saved ⇒ {report_path}")

except Exception as e:
    print("Could not write summary:", e)

print(f"Training completed in {int(h)}h {int(m)}m {int(s)}s")
