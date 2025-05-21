# ======================================================================
# DeepSeek Text-to-SQL Fine-Tuning Script
# For bachelor thesis comparing instruction-tuned vs code-pretrained models
# Optimized for A100 GPU on Colab
# ======================================================================
#@title { vertical-output: true}

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
!pip install -q \
        "transformers>=4.39.3" \
        "datasets>=2.18.0" \
        "accelerate>=0.28.0" \
        "bitsandbytes>=0.42.0" \
        "peft>=0.4.0" \
        "trl>=0.7.10"

# Try installing flash-attn with fallback
try:
    !pip install -q flash-attn --no-build-isolation
    print("Flash Attention installed successfully!")
except:
    print("Flash Attention installation failed, continuing without it")

import pathlib, sys, importlib, torch
if "transformers" in sys.modules: del sys.modules["transformers"]
import transformers, datasets, accelerate, peft
print("transformers:", transformers.__version__)
print("datasets    :", datasets.__version__)
print("accelerate  :", accelerate.__version__)
print("peft        :", peft.__version__)
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
import site
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    StoppingCriteriaList,
    StoppingCriteria,
    GenerationConfig
)
# Import PEFT functions
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

set_seed(42)

# --- Configuration ---
SCHEMA_FORMAT = "sql"
MODEL_NAME      = "deepseek-ai/deepseek-coder-6.7b-instruct"
EXPERIMENT_NAME = "deepseek_coder_6.7b_lora_v1.4.2"
MODEL_LABEL = MODEL_NAME.split("/")[-1]

# 2)  TRAINING HYPER-PARAMS - A100 OPTIMIZED
EPOCHS                     = 4
LEARNING_RATE              = 2e-4
BATCH_SIZE                 = 2
GRADIENT_ACCUMULATION_STEPS= 8
WEIGHT_DECAY               = 0.0
MAX_INPUT_LENGTH           = 1024
MAX_TARGET_LENGTH          = 256
MAX_GRAD_NORM              = 1.0
RESUME_FROM_CHECKPOINT     = False
SUBSET_RATIO               = 1.0

# 3)  MIXED PRECISION & QUANTIZATION
USE_FP16 = False
USE_BF16 = True
LOAD_IN_4BIT = True

# 4) LoRA Parameters
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 5) Special tokens for SQL formatting
SQL_START_TOKEN = "<SQL_START>"
SQL_END_TOKEN = "<SQL_END>"

# Google Drive paths
DRIVE_BASE_DIR = "/content/drive/MyDrive/text2sql"
DRIVE_OUTPUT_DIR = f"{DRIVE_BASE_DIR}/{EXPERIMENT_NAME}"
DRIVE_DATASET_SOURCE_DIR = f"{DRIVE_BASE_DIR}/datasets/spider"
DRIVE_LOGS_DIR = f"{DRIVE_BASE_DIR}/logs/{EXPERIMENT_NAME}"

# Local paths
LOCAL_DATASET_DIR = "/content/datasets/spider"
local_db_path     = f"{LOCAL_DATASET_DIR}/database"
drive_db_path = f"{DRIVE_DATASET_SOURCE_DIR}/database"
TRAIN_PATH = os.path.join(LOCAL_DATASET_DIR, "train_spider.json")
DEV_PATH   = os.path.join(LOCAL_DATASET_DIR, "dev.json")

# Ensure local root exists
os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

# One-off copy of databases to the Colab SSD
if not os.path.exists(local_db_path) and os.path.exists(drive_db_path):
    print("Copying Spider databases to local scratch (≈300 MB)…")
    !cp -r "{drive_db_path}" "{local_db_path}"

# Create directories
os.makedirs(DRIVE_BASE_DIR, exist_ok=True)
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
os.makedirs(DRIVE_LOGS_DIR, exist_ok=True)
os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

print(f"--- Running Experiment: {EXPERIMENT_NAME} ---")
print(f"Schema Format: {SCHEMA_FORMAT} (with Types)")
print(f"Model: {MODEL_NAME}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Per Device Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Gradient Clipping: {MAX_GRAD_NORM}")
print(f"Max Input Length: {MAX_INPUT_LENGTH}")
print(f"LoRA r: {LORA_R}, alpha: {LORA_ALPHA}")
print(f"4-bit Quantization: {LOAD_IN_4BIT}")
print(f"BF16 Enabled: {USE_BF16}")
print(f"Using {SUBSET_RATIO*100}% of training data")

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
                col_info = f"col:{col_name} ({col_type})"
                if col_idx in primary_keys: col_info += " (PRIMARY KEY)"
                if col_idx in fk_dict:
                    ref_col_idx = fk_dict[col_idx]
                    if 0 <= ref_col_idx < len(columns):
                         ref_tab_idx, ref_col_name = columns[ref_col_idx]
                         if 0 <= ref_tab_idx < len(tables):
                              ref_table = tables[ref_tab_idx]
                              col_info += f" (FOREIGN KEY -> {ref_table}.{ref_col_name})"
                table_columns.append(col_info)
        if table_columns:
            table_columns.sort()
            table_def = (
                f"tab:{table}\n"
                f"    columns: {', '.join(table_columns)}"
            )
            table_defs.append(table_def)
    table_defs.sort()
    return "\n".join(table_defs)

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

# ----------------------------------------------------------
# One-off sanitation: write files without the `sql` column
# ----------------------------------------------------------
import pathlib

def strip_sql(src_path):
    """Remove the 'sql' column that causes Arrow conversion errors."""
    dst_path = pathlib.Path(src_path).with_suffix(".json").with_stem(
        pathlib.Path(src_path).stem + "_clean"
    )

    if dst_path.exists():  # don't repeat work after a runtime reset
        return dst_path.as_posix()

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ex in data:
        ex.pop("sql", None)  # remove if present

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"✔︎ wrote cleaned file → {dst_path.name}")
    return dst_path.as_posix()

TRAIN_CLEAN = strip_sql(TRAIN_PATH)
DEV_CLEAN = strip_sql(DEV_PATH)

# --- Load Database Schemas ---
print("\nLoading database schemas...")
db_schemas = load_tables_json('tables.json')
print(f"Loaded schemas for {len(db_schemas)} databases.")

# --- Process SQL queries to normalize format ---
def normalize_sql(sql):
    """Normalize SQL queries to match Spider format (lowercase keywords, no semicolons)."""
    sql = sql.strip()
    # Remove trailing semicolons
    sql = sql.rstrip(';')

    # Convert SQL keywords to lowercase
    keywords = [
        "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
        "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN",
        "LIMIT", "OFFSET", "UNION", "INTERSECT", "EXCEPT",
        "COUNT", "AVG", "SUM", "MIN", "MAX", "ALL", "DISTINCT",
        "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN", "AS", "WITH"
    ]

    # Create a regex pattern that matches whole words only
    pattern = r'\b(' + '|'.join(keywords) + r')\b'

    # Convert keywords to lowercase
    sql = re.sub(pattern, lambda m: m.group(0).lower(), sql, flags=re.IGNORECASE)

    return sql

# --- Improved DeepSeek Prompt Builder ---
def build_prompt(question: str, db_id: str, sql: str | None = None) -> str:
    schema = get_sql_schema_string(db_id, db_schemas)

    prompt = (
        f"-- Database schema for `{db_id}`\n"
        f"{schema}\n\n"
        f"-- User question\n"
        f"{question}\n\n"
        f"-- Write one correct SQL query.\n"
        f"{SQL_START_TOKEN}\n"
    )

    if sql is None:
        return prompt  # inference
    sql = normalize_sql(sql)
    return prompt + sql + f"\n{SQL_END_TOKEN}"

# --- Load and Prepare Datasets ---
print("Loading Spider datasets …")

from datasets import load_dataset, Features, Value

spider_feats = Features({
    "question": Value("string"),
    "query":    Value("string"),
    "db_id":    Value("string"),
})

# read the train and dev files separately
train_raw = load_dataset(
    "json",
    data_files=TRAIN_CLEAN,
    split="train",
)

dev_raw = load_dataset(
    "json",
    data_files=DEV_CLEAN,
    split="train",
)

dev_raw = dev_raw.select(range(1034))

# ----------------------------------------------------------
# build the prompt *before* tokenisation so we can condense the schema
# ----------------------------------------------------------
def add_prompt(example):
    example["prompt"] = build_prompt(
        example["question"],
        example["db_id"],
        example["query"]
    )
    return example

train_dataset = train_raw.map(add_prompt, num_proc=4)
dev_dataset   = dev_raw.map(add_prompt,   num_proc=4)

del train_raw, dev_raw   # free RAM
print("Datasets loaded and prompts added.")

# After loading and preparing train_dataset
print("Adding 'finish' examples to help model learn when to stop...")
finish_examples = []
for i in range(min(70, len(train_dataset) // 100)):
    finish_examples.append({
        "question": f"Do nothing {i}",
        "db_id": "concert_singer",
        "query": "SELECT 1"
    })

# Convert to Dataset and combine with train_dataset
finish_dataset = Dataset.from_list(finish_examples)
train_dataset = datasets.concatenate_datasets([train_dataset, finish_dataset])
print(f"Added {len(finish_examples)} 'finish' examples. New training size: {len(train_dataset)}")

# Apply data subset if needed
if SUBSET_RATIO < 1.0:
    train_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset)*SUBSET_RATIO)))
    print(f"Using {len(train_dataset)} examples ({SUBSET_RATIO*100}% of training data)")

# Calculate steps per epoch AFTER applying subset if needed
steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(total_steps * 0.1)  # Increased to 10% for smoother LoRA warmup
half_epoch_steps = steps_per_epoch // 2

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Half-epoch steps: {half_epoch_steps}")
print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

# --- Load Model and Tokenizer ---
print(f"\nLoading tokenizer and adding special tokens...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    use_fast=True
)

# Add special tokens for SQL formatting
special_tokens = {
    "additional_special_tokens": [SQL_START_TOKEN, SQL_END_TOKEN]
}
tokenizer.add_special_tokens(special_tokens)

# Make sure pad token is set - critical for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Added special tokens: {SQL_START_TOKEN}, {SQL_END_TOKEN}")
print(f"SQL_START_TOKEN ID: {tokenizer.convert_tokens_to_ids(SQL_START_TOKEN)}")
print(f"SQL_END_TOKEN ID: {tokenizer.convert_tokens_to_ids(SQL_END_TOKEN)}")

# Try to configure with Flash Attention 2 - improved error handling
try:
    import torch.nn as nn

    # Check if flash-attn is available
    has_flash_attn = False
    try:
        import flash_attn
        has_flash_attn = True
        print("Flash Attention is available")
    except ImportError:
        print("Flash Attention not available")

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Add Flash Attention if available
    if has_flash_attn:
        config.attn_implementation = "flash_attention_2"
        print("Using Flash Attention 2")

    print("Loading model with custom config...")

    # Load model with 4-bit quantization for memory efficiency
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
except Exception as e:
    print(f"Error loading model with custom config: {e}")
    print("Falling back to standard loading...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto",
        trust_remote_code=True
    )

# Resize token embeddings to account for new special tokens
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False

# Prepare the model for LoRA fine-tuning
if LOAD_IN_4BIT:
    # Set compute dtype based on settings
    compute_dtype = torch.bfloat16 if USE_BF16 else torch.float16

    try:
        model = prepare_model_for_kbit_training(model)
        print("Successfully prepared model for kbit training")
    except Exception as e:
        print(f"Warning: prepare_model_for_kbit_training failed: {e}")
        print("Continuing without it - model might still work with LoRA")

    # Set gradient checkpointing to disabled for A100 (for speed)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

# Add LoRA adapters
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Model and tokenizer loaded.")

# --- Custom SQL stopping criteria for generation ---
class StopOnSQLEnd(StoppingCriteria):
    # class-level constant for convenience
    eos_token_id = tokenizer.convert_tokens_to_ids(SQL_END_TOKEN)

    def __init__(self, sql_end_token_id=None):
        # allow override but default to class constant
        self.sql_end_token_id = (
            sql_end_token_id if sql_end_token_id is not None else self.eos_token_id
        )

    def __call__(self, input_ids, scores, **kwargs):
        # stop when SQL_END_TOKEN is generated at the end
        for ids in input_ids:
            ids_list = ids.tolist()
            if len(ids_list) >= 1 and ids_list[-1] == self.sql_end_token_id:
                return True
            if len(ids_list) >= 2 and ids_list[-2] == self.sql_end_token_id:
                return True
        return False

# --- Tokenization with proper SQL masking ---
def tokenize_examples(examples):
    """Tokenize examples with mask for loss computation only on SQL part."""
    # Build the full prompt+answer strings
    texts = [
        build_prompt(q, db, a)
        for q, db, a in zip(
            examples["question"],
            examples["db_id"],
            examples["query"],
        )
    ]

    # Tokenize (prompt + SQL answer), truncating only the prompt if too long
    encodings = tokenizer(
        texts,
        truncation="only_first",
        max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH,
    )

    # ensure it's a mutable dict
    encodings = {k: v for k, v in encodings.items()}

    # lookup our SQL markers
    sql_start_id = tokenizer.convert_tokens_to_ids(SQL_START_TOKEN)
    sql_end_id = tokenizer.convert_tokens_to_ids(SQL_END_TOKEN)

    labels = []
    for ids in encodings["input_ids"]:
        ids_list = ids.tolist() if not isinstance(ids, list) else ids

        # find begin/end of SQL answer
        starts = [i for i, x in enumerate(ids_list) if x == sql_start_id]
        ends = [i for i, x in enumerate(ids_list) if x == sql_end_id]

        if starts and ends:
            start_idx = starts[-1] + 1
            # first end after the last start
            end_idx = next((i for i in ends if i > start_idx), len(ids_list))

            if end_idx == len(ids_list):
                ids_list[-1] = sql_end_id
                end_idx = len(ids_list) - 1

            # mask out everything except the SQL span (inclusive of end token)
            ex_labels = [-100] * len(ids_list)
            ex_labels[start_idx : end_idx + 1] = ids_list[start_idx : end_idx + 1]
        else:
            ex_labels = [-100] * len(ids_list)

        labels.append(ex_labels)

    encodings["labels"] = labels
    return encodings


# --- Tokenize the Datasets ---
print("\nTokenizing datasets with SQL masking...")
tokenized_train = train_dataset.map(
    tokenize_examples,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training dataset"
)
tokenized_dev = dev_dataset.map(
    tokenize_examples,
    batched=True,
    remove_columns=dev_dataset.column_names,
    desc="Tokenizing development dataset"
)
print(f"Training dataset tokenized: {len(tokenized_train)} examples")
print(f"Development dataset tokenized: {len(tokenized_dev)} examples")

# --- Custom data collator ---
print("\nInitializing data collator...")
def collate_with_masked_labels(features):
    # Pad input_ids/attention_mask with the tokenizer helper
    pad_batch = tokenizer.pad(
        {k: [f[k] for f in features] for k in ["input_ids", "attention_mask"]},
        padding="longest",
        return_tensors="pt",
        pad_to_multiple_of=8  # For better GPU efficiency
    )

    # Pad labels by hand (use -100 so they are ignored in the loss)
    max_len = pad_batch["input_ids"].shape[1]
    labels = []
    for f in features:
        l = f["labels"]
        l = l + [-100] * (max_len - len(l))  # Pad with -100 which is ignored in loss
        labels.append(torch.tensor(l, dtype=torch.long))

    pad_batch["labels"] = torch.stack(labels)
    return pad_batch

# Use the custom collator
data_collator = collate_with_masked_labels
print("Custom data collator initialized with proper label masking.")

# Compute_metrics function
import sqlite3, tempfile, subprocess, json, os
from sklearn.metrics import accuracy_score

def extract_sql(text: str) -> str:
    if SQL_START_TOKEN in text and SQL_END_TOKEN in text:
        return text.split(SQL_START_TOKEN)[1].split(SQL_END_TOKEN)[0].strip()
    # fallback
    return text.splitlines()[-1]

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if preds.ndim == 3:  # logits → ids
        preds = preds.argmax(-1)

    pred_txt = tokenizer.batch_decode(preds, skip_special_tokens=False)
    gold_txt = []
    for lab in labels:
        gold_ids = [i for i in lab if i != -100]
        gold_txt.append(tokenizer.decode(gold_ids, skip_special_tokens=True))

    pred_sql = [normalize_sql(extract_sql(t)) for t in pred_txt]
    gold_sql = [normalize_sql(extract_sql(t)) for t in gold_txt]

    em = accuracy_score(pred_sql, gold_sql)

    return {"exact_match": round(100 * em, 2)}

# --- Training Arguments ---
import inspect
from transformers.training_args import TrainingArguments as HFTrainingArguments

# Create training arguments directly without args_dict
training_args = HFTrainingArguments(
    output_dir=DRIVE_OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,
    bf16=USE_BF16,
    fp16=USE_FP16,
    gradient_checkpointing=False,
    max_grad_norm=MAX_GRAD_NORM,
    logging_steps=50,
    logging_dir=DRIVE_LOGS_DIR,
    save_total_limit=2,
    evaluation_strategy="no",
    #eval_steps=steps_per_epoch // 2,
    load_best_model_at_end=False,
    report_to="tensorboard",
    optim="paged_adamw_8bit",
    seed=42,
    remove_unused_columns=False,
    dataloader_num_workers=4,
)

training_args.load_best_model_at_end = True
training_args.metric_for_best_model = "exact_match"
training_args.greater_is_better = True

# --- Trainer Initialization ---
print("\nInitializing Trainer...")

# ---------- generation params used by Trainer.evaluate / predict ----------
generation_config = GenerationConfig.from_model_config(model.config)
generation_config.update(
    decoder_start_token_id=tokenizer.convert_tokens_to_ids(SQL_START_TOKEN),
    eos_token_id=tokenizer.convert_tokens_to_ids(SQL_END_TOKEN),
    max_new_tokens=256,
    num_beams=4,
    length_penalty=0.6,
    no_repeat_ngram_size=3,
    early_stopping=True
)
model.generation_config = generation_config

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("Trainer initialized.")

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
    # Full training
    print("Skipping initial evaluation to save memory")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ---------- post-training actions ----------
    print("\n--- Training Finished ---")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\nSaving the final model …")
    # Save only the LoRA adapters to save space
    model.save_pretrained(DRIVE_OUTPUT_DIR)
    tokenizer.save_pretrained(DRIVE_OUTPUT_DIR)
    print(f"Model saved to {DRIVE_OUTPUT_DIR}")

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
        f.write(f"bf16 Enabled        : {training_args.bf16}\n")
        f.write(f"4-bit Quantization  : {LOAD_IN_4BIT}\n")
        f.write(f"LoRA Config         : r={LORA_R}, alpha={LORA_ALPHA}\n")
        f.write(f"Epochs Configured   : {EPOCHS}\n")
        f.write(f"Epochs Trained      : {metrics.get('epoch', 0):.2f}\n")
        f.write(f"Training Time       : {int(h)}h {int(m)}m {int(s)}s\n")
        f.write(f"Learning Rate       : {LEARNING_RATE}\n")
        f.write(f"Batch Size / Device : {BATCH_SIZE}\n")
        f.write(f"Grad Accum Steps    : {GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Effective BatchSize : {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Weight Decay        : {WEIGHT_DECAY}\n")
        f.write(f"Warm-up Steps       : {warmup_steps}\n")
        f.write(f"Max Grad-Norm       : {MAX_GRAD_NORM}\n")
        f.write(f"Max Input Length    : {MAX_INPUT_LENGTH}\n")
        f.write(f"Data Subset Ratio   : {SUBSET_RATIO}\n")
        f.write(f"Special Tokens      : {SQL_START_TOKEN}, {SQL_END_TOKEN}\n")
        f.write(f"Optimizer           : {training_args.optim}\n")
        f.write(f"GPU Type            : {gpu_name}\n")
        f.write(f"GPU Memory          : {gpu_memory:.2f} GB\n")

        # ---- training metrics ----
        f.write("\nTraining Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k:<25} : {v}\n")

        f.write(f"\nModel saved to: {training_args.output_dir}\n")
        f.write("To evaluate the model, run the evaluation script with appropriate SQL stopping criteria.\n")

    print(f"Training summary saved ⇒ {report_path}")

except Exception as e:
    print("Could not write summary:", e)

print(f"Training completed in {int(h)}h {int(m)}m {int(s)}s")

# ---------- post-training actions ----------
print("\n--- Training Finished ---")
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print("\nSaving the final model …")
# Save only the LoRA adapters to save space
model.save_pretrained(DRIVE_OUTPUT_DIR)
tokenizer.save_pretrained(DRIVE_OUTPUT_DIR)
print(f"Model saved to {DRIVE_OUTPUT_DIR}")

# ----------------------------------------------------------
#  Generate training report
# ----------------------------------------------------------
from pathlib import Path
import time, os

model_label = Path(MODEL_NAME).name

elapsed = time.time() - start_time
h, rem   = divmod(elapsed, 3600)
m, s     = divmod(rem, 60)

report_path = os.path.join(training_args.output_dir, "training_summary.txt")

try:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment          : {EXPERIMENT_NAME}\n")
        f.write(f"Model               : {model_label}\n")
        f.write(f"Schema Format       : {SCHEMA_FORMAT} (with types)\n")
        f.write(f"bf16 Enabled        : {training_args.bf16}\n")
        f.write(f"4-bit Quantization  : {LOAD_IN_4BIT}\n")
        f.write(f"LoRA Config         : r={LORA_R}, alpha={LORA_ALPHA}\n")
        f.write(f"Epochs Configured   : {EPOCHS}\n")
        f.write(f"Epochs Trained      : {metrics.get('epoch', 0):.2f}\n")
        f.write(f"Training Time       : {int(h)}h {int(m)}m {int(s)}s\n")
        f.write(f"Learning Rate       : {LEARNING_RATE}\n")
        f.write(f"Batch Size / Device : {BATCH_SIZE}\n")
        f.write(f"Grad Accum Steps    : {GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Effective BatchSize : {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Weight Decay        : {WEIGHT_DECAY}\n")
        f.write(f"Warm-up Steps       : {warmup_steps}\n")
        f.write(f"Max Grad-Norm       : {MAX_GRAD_NORM}\n")
        f.write(f"Max Input Length    : {MAX_INPUT_LENGTH}\n")
        f.write(f"Data Subset Ratio   : {SUBSET_RATIO}\n")
        f.write(f"Special Tokens      : {SQL_START_TOKEN}, {SQL_END_TOKEN}\n")
        f.write(f"Optimizer           : {training_args.optim}\n")
        f.write(f"GPU Type            : {gpu_name}\n")
        f.write(f"GPU Memory          : {gpu_memory:.2f} GB\n")

        # ---- training metrics ----
        f.write("\nTraining Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k:<25} : {v}\n")

        f.write(f"\nModel saved to: {training_args.output_dir}\n")
        f.write("To evaluate the model, run the evaluation script with appropriate SQL stopping criteria.\n")

    print(f"Training summary saved ⇒ {report_path}")
except Exception as e:
    print("Could not write summary:", e)

print(f"Training completed in {int(h)}h {int(m)}m {int(s)}s")
