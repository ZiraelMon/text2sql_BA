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

import pathlib, sys, importlib, torch, os, glob, json
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
EXPERIMENT_NAME = "deepseek_coder_6.7b_lora_v1.5"
MODEL_LABEL = MODEL_NAME.split("/")[-1]

# 2)  TRAINING HYPER-PARAMS - A100 OPTIMIZED
# Updated based on recommendations
EPOCHS                     = 10
LEARNING_RATE              = 2e-5
BATCH_SIZE                 = 2
GRADIENT_ACCUMULATION_STEPS= 8
WEIGHT_DECAY               = 0.01
MAX_INPUT_LENGTH           = 1024
MAX_TARGET_LENGTH          = 256
MAX_GRAD_NORM             = 1.0
RESUME_FROM_CHECKPOINT     = False
SUBSET_RATIO               = 1.0

# 3)  MIXED PRECISION & QUANTIZATION
USE_FP16 = False
USE_BF16 = True
# Changed from 4-bit to 8-bit quantization for better precision
LOAD_IN_4BIT = False
LOAD_IN_8BIT = True

# 4) LoRA Parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "dense"]

# 5) Special tokens for SQL formatting
SQL_START_TOKEN = "<SQL_START>"
SQL_END_TOKEN = "<SQL_END>"

# 6) Generation config for inference
NUM_BEAMS = 4

# Google Drive paths
DRIVE_BASE_DIR = "/content/drive/MyDrive/text2sql"
DRIVE_OUTPUT_DIR = f"{DRIVE_BASE_DIR}/{EXPERIMENT_NAME}"
DRIVE_DATASET_SOURCE_DIR = f"{DRIVE_BASE_DIR}/datasets/spider"
DRIVE_LOGS_DIR = f"{DRIVE_BASE_DIR}/logs/{EXPERIMENT_NAME}"

# Local paths
LOCAL_DATASET_DIR = "/content/datasets/spider"

# ==========================================================
# PATH DETECTION - Find where dataset files actually exist
# ==========================================================
# Create directories
os.makedirs(DRIVE_BASE_DIR, exist_ok=True)
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
os.makedirs(DRIVE_LOGS_DIR, exist_ok=True)

# Check if local dataset dir exists, create if not
if not os.path.exists(LOCAL_DATASET_DIR):
    print(f"Local dataset directory doesn't exist. Creating it...")
    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

# Define potential path locations
local_tables_path = f"{LOCAL_DATASET_DIR}/tables.json"
local_train_path = f"{LOCAL_DATASET_DIR}/train_spider.json"
local_dev_path = f"{LOCAL_DATASET_DIR}/dev.json"

drive_tables_path = f"{DRIVE_DATASET_SOURCE_DIR}/tables.json"
drive_train_path = f"{DRIVE_DATASET_SOURCE_DIR}/train_spider.json"
drive_dev_path = f"{DRIVE_DATASET_SOURCE_DIR}/dev.json"

local_db_path = f"{LOCAL_DATASET_DIR}/database"
drive_db_path = f"{DRIVE_DATASET_SOURCE_DIR}/database"

# Check which paths exist
print("\n=== Path Existence Check ===")
print(f"Local tables.json exists: {os.path.exists(local_tables_path)}")
print(f"Local train_spider.json exists: {os.path.exists(local_train_path)}")
print(f"Local dev.json exists: {os.path.exists(local_dev_path)}")
print(f"Local database path exists: {os.path.exists(local_db_path)}")

print(f"Drive tables.json exists: {os.path.exists(drive_tables_path)}")
print(f"Drive train_spider.json exists: {os.path.exists(drive_train_path)}")
print(f"Drive dev.json exists: {os.path.exists(drive_dev_path)}")
print(f"Drive database path exists: {os.path.exists(drive_db_path)}")

# Function to count database files
def count_sqlite_files(dir_path):
    if not os.path.exists(dir_path):
        return 0
    return len(glob.glob(f"{dir_path}/*/*.sqlite"))

local_db_count = count_sqlite_files(local_db_path)
drive_db_count = count_sqlite_files(drive_db_path)

print(f"Local database SQLite files: {local_db_count}")
print(f"Drive database SQLite files: {drive_db_count}")

# Decide which paths to use for files
TABLES_JSON = local_tables_path if os.path.exists(local_tables_path) else drive_tables_path
TRAIN_PATH = local_train_path if os.path.exists(local_train_path) else drive_train_path
DEV_PATH = local_dev_path if os.path.exists(local_dev_path) else drive_dev_path

# Decide which path to use for database
if local_db_count > 0:
    DB_PATH = local_db_path
    print(f"Using local database path: {DB_PATH}")
elif drive_db_count > 0:
    DB_PATH = drive_db_path
    print(f"Using Drive database path: {DB_PATH}")
else:
    print("No database files found in either location")
    DB_PATH = drive_db_path  # Default to Drive path

print(f"\n=== Paths Being Used ===")
print(f"Tables JSON: {TABLES_JSON}")
print(f"Train data: {TRAIN_PATH}")
print(f"Dev data: {DEV_PATH}")
print(f"Database: {DB_PATH}")

print(f"\n--- Running Experiment: {EXPERIMENT_NAME} ---")
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
print(f"8-bit Quantization: {LOAD_IN_8BIT}")
print(f"4-bit Quantization: {LOAD_IN_4BIT}")
print(f"BF16 Enabled: {USE_BF16}")
print(f"Using {SUBSET_RATIO*100}% of training data")
print(f"Beam Search: {NUM_BEAMS} beams")

# --- Schema Utilities ---
def load_tables_json(fp=TABLES_JSON):
    """Load the tables.json file containing schema information."""
    print(f"Loading tables.json from: {fp}")
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: tables.json not found at {fp}")
        raise
    db_schemas = {db_info['db_id']: db_info for db_info in tables_data}
    return db_schemas

def get_sql_schema_string(db_id, db_schemas):
    """Create improved SQL schema string with bullet points, including types, PKs, FKs."""
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

    # Create a mapping from table index to table name
    table_map = {i: name for i, name in enumerate(tables)}

    # Group columns by their table
    table_columns = {table: [] for table in tables}

    for col_idx, (tab_idx, col_name) in enumerate(columns):
        if tab_idx >= 0:  # Skip the * column
            table = table_map[tab_idx]
            col_type = column_types[col_idx].upper()
            col_attrs = []

            # Add PRIMARY KEY tag if applicable
            if col_idx in primary_keys:
                col_attrs.append("PK")

            # Add FOREIGN KEY tag if applicable
            if col_idx in fk_dict:
                ref_col_idx = fk_dict[col_idx]
                if 0 <= ref_col_idx < len(columns):
                    ref_tab_idx, ref_col_name = columns[ref_col_idx]
                    if 0 <= ref_tab_idx < len(tables):
                        ref_table = tables[ref_tab_idx]
                        col_attrs.append(f"FK→{ref_table}.{ref_col_name}")

            # Format column with its attributes
            attr_str = " ".join(col_attrs)
            if attr_str:
                col_info = f"{col_name} ({col_type}, {attr_str})"
            else:
                col_info = f"{col_name} ({col_type})"

            table_columns[table].append(col_info)

    # Build the schema string with bullet points
    schema_lines = []

    for table in sorted(tables):
        # Sort columns alphabetically for consistency
        cols = sorted(table_columns[table])
        col_str = ", ".join(cols)
        schema_lines.append(f"• {table} (columns: {col_str})")

    return "\n".join(schema_lines)

# --- Copy necessary files from Drive to local if needed ---
def setup_local_dataset():
    """Copy necessary dataset files from Drive to local if they don't exist locally."""
    files_copied = False

    # Check tables.json
    if not os.path.exists(local_tables_path) and os.path.exists(drive_tables_path):
        print(f"Copying tables.json from Drive to local...")
        !cp -v "{drive_tables_path}" "{local_tables_path}"
        files_copied = True

    # Check train_spider.json
    if not os.path.exists(local_train_path) and os.path.exists(drive_train_path):
        print(f"Copying train_spider.json from Drive to local...")
        !cp -v "{drive_train_path}" "{local_train_path}"
        files_copied = True

    # Check dev.json
    if not os.path.exists(local_dev_path) and os.path.exists(drive_dev_path):
        print(f"Copying dev.json from Drive to local...")
        !cp -v "{drive_dev_path}" "{local_dev_path}"
        files_copied = True

    # Check if we need to copy database (will be skipped by default)
    copy_db = False  # Set to True if you want to copy database files
    if copy_db and local_db_count == 0 and drive_db_count > 0:
        print(f"Copying database directory from Drive to local (this might take a while)...")
        os.makedirs(local_db_path, exist_ok=True)
        !cp -r "{drive_db_path}"/* "{local_db_path}/"
        files_copied = True

    if files_copied:
        print("Files copied successfully.")
    else:
        print("No files needed to be copied.")

# --- Run Setup Function ---
print("\n--- Setting up dataset ---")
setup_local_dataset()

# --- Load Database Schemas ---
print("\nLoading database schemas...")
db_schemas = load_tables_json(TABLES_JSON)
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
def build_prompt(question, schema, db_id, answer=None):
    """Build improved prompt with explicit SQL start/end tokens and clearer formatting."""
    # Updated prompt format with SQL: line before the SQL_START_TOKEN
    prefix = (
        f"### Text-to-SQL task\n"
        f"Database: {db_id}\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"SQL:\n{SQL_START_TOKEN}\n"
    )

    if answer is None:
        return prefix
    else:
        # Normalize the SQL query
        normalized_answer = normalize_sql(answer)
        return prefix + normalized_answer + f"\n{SQL_END_TOKEN}"

# --- Load and Prepare Datasets ---
def load_and_prepare_data(file_path, db_schemas_dict, schema_fmt):
    """Load and prepare dataset with improved prompt format."""
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            spider_data = json.load(f)
        print(f"Preparing {len(spider_data)} examples...")

        # Create a list to store the processed examples
        processed_examples = []

        # Process each example
        for i, example in enumerate(spider_data):
            if i > 0 and i % 100 == 0:
                print(f"  Processed {i}/{len(spider_data)} examples...")

            db_id = example['db_id']
            try:
                # Get schema string
                schema_str = get_sql_schema_string(db_id, db_schemas_dict) if schema_fmt == "sql" else ""

                # Create a new entry with relevant fields
                processed_examples.append({
                    "question": example['question'],
                    "schema": schema_str,
                    "db_id": db_id,
                    "query": example['query']
                })

            except Exception as e:
                print(f"Warning: Skipping example for db_id '{db_id}': {e}")

        print(f"  Processed {len(processed_examples)}/{len(spider_data)} examples.")

        # Convert to Dataset
        dataset = Dataset.from_list(processed_examples)
        return dataset

    except Exception as e:
        print(f"Error processing data from {file_path}: {e}")
        raise

print("Loading datasets...")
train_dataset = load_and_prepare_data(TRAIN_PATH, db_schemas, SCHEMA_FORMAT)
dev_dataset = load_and_prepare_data(DEV_PATH, db_schemas, SCHEMA_FORMAT)

if train_dataset is None or dev_dataset is None:
     raise ValueError("Failed to load train or dev dataset.")

# After loading and preparing train_dataset
print("Adding 'finish' examples to help model learn when to stop...")
finish_examples = []

# Simple dummy examples
for i in range(min(70, len(train_dataset) // 100)):
    finish_examples.append({
        "question": f"Do nothing {i}",
        "schema": "Table: dummy\nColumns: id (NUMBER)",
        "db_id": "none",
        "query": "SELECT 1"
    })

# Add table-only queries to reinforce FROM clause
print("Adding table-only examples to reinforce FROM clause...")
table_examples = []
for db_id in list(db_schemas.keys())[:10]:  # Use first 10 databases
    schema = db_schemas[db_id]
    tables = schema["table_names_original"]

    for table in tables[:3]:  # Add examples for first 3 tables of each database
        schema_str = get_sql_schema_string(db_id, db_schemas)
        table_examples.append({
            "question": f"List all columns of {table}",
            "schema": schema_str,
            "db_id": db_id,
            "query": f"select * from {table}"
        })

print(f"Added {len(table_examples)} table-only examples")

# Convert to Dataset and combine with train_dataset
finish_dataset = Dataset.from_list(finish_examples)
table_dataset = Dataset.from_list(table_examples)
train_dataset = datasets.concatenate_datasets([train_dataset, finish_dataset, table_dataset])
print(f"Combined dataset size: {len(train_dataset)} examples")

# Apply data subset if needed
if SUBSET_RATIO < 1.0:
    train_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset)*SUBSET_RATIO)))
    print(f"Using {len(train_dataset)} examples ({SUBSET_RATIO*100}% of training data)")

# Calculate steps per epoch AFTER applying subset if needed
steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
total_steps = steps_per_epoch * EPOCHS
# Increased warmup to 15% of total steps
warmup_steps = int(total_steps * 0.15)
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

# Try to configure with Flash Attention 2
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

    # Load model with quantization for memory efficiency
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        # Configure 4-bit quantization
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
    elif LOAD_IN_8BIT:
        # Use 8-bit quantization instead
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            load_in_8bit=True,
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

    if LOAD_IN_8BIT:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_8bit=True,
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            load_in_4bit=LOAD_IN_4BIT,
            device_map="auto",
            trust_remote_code=True
        )

# Resize token embeddings to account for new special tokens
model.resize_token_embeddings(len(tokenizer))

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

    # Set gradient checkpointing to disabled
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
        build_prompt(q, s, db, a)
        for q, s, db, a in zip(
            examples["question"],
            examples["schema"],
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
            start_idx = starts[-1] + 1  # Start after the SQL_START_TOKEN
            # first end after the last start
            end_idx = next((i for i in ends if i > start_idx), len(ids_list))

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
        pad_to_multiple_of=8
    )

    # Pad labels by hand
    max_len = pad_batch["input_ids"].shape[1]
    labels = []
    for f in features:
        l = f["labels"]
        l = l + [-100] * (max_len - len(l))
        labels.append(torch.tensor(l, dtype=torch.long))

    pad_batch["labels"] = torch.stack(labels)
    return pad_batch

# Use the custom collator
data_collator = collate_with_masked_labels
print("Custom data collator initialized with proper label masking.")

# --- Simple metric function that won't crash ---
def compute_metrics(eval_pred):
    # Simple placeholder that won't crash
    return {}

# --- Training Arguments ---
import inspect
from transformers import TrainingArguments as HFTrainingArguments

# Create training arguments with evaluation strategy set to steps
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
    report_to="tensorboard",
    optim="paged_adamw_8bit",
    seed=42,
    remove_unused_columns=False,
    dataloader_num_workers=4,
)

training_args.evaluation_strategy     = "steps"
training_args.eval_steps              = half_epoch_steps
training_args.load_best_model_at_end  = True
training_args.metric_for_best_model   = "eval_loss"

# --- Trainer Initialization ---
print("\nInitializing Trainer...")

# Create generation config with proper tokens and beam search
generation_config = GenerationConfig.from_model_config(model.config)

# --- set each field directly ----
generation_config.decoder_start_token_id  = tokenizer.convert_tokens_to_ids(SQL_START_TOKEN)
generation_config.eos_token_id            = tokenizer.convert_tokens_to_ids(SQL_END_TOKEN)
generation_config.num_beams               = NUM_BEAMS
generation_config.early_stopping          = True
generation_config.no_repeat_ngram_size    = 3
generation_config.length_penalty          = 1.0

# attach it to the model
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
    model.save_pretrained(DRIVE_OUTPUT_DIR)  # Saves only LoRA if it's a peft model
    tokenizer.save_pretrained(DRIVE_OUTPUT_DIR)
    print(f"Model saved to {DRIVE_OUTPUT_DIR}")

    # Also save the generation config for inference
    generation_config.save_pretrained(DRIVE_OUTPUT_DIR)
    print(f"Generation config saved to {DRIVE_OUTPUT_DIR}")

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
        f.write(f"8-bit Quantization  : {LOAD_IN_8BIT}\n")
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
        f.write(f"Beam Search         : {NUM_BEAMS} beams\n")
        f.write(f"Evaluation Strategy : {training_args.evaluation_strategy}\n")
        f.write(f"Evaluation Steps    : {training_args.eval_steps}\n")
        f.write(f"Database Path       : {DB_PATH}\n")
        f.write(f"Tables JSON Path    : {TABLES_JSON}\n")

        # ---- training metrics ----
        f.write("\nTraining Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k:<25} : {v}\n")

        f.write(f"\nModel saved to: {training_args.output_dir}\n")
        f.write("To evaluate the model, run the evaluation script with appropriate SQL stopping criteria.\n")
        f.write("\nModifications made in this version:\n")
        f.write("- Improved schema formatting with bullet points\n")
        f.write("- Added SQL: line before SQL_START token\n")
        f.write("- Increased epochs from 6 to 10\n")
        f.write("- Lowered learning rate to 2e-5\n")
        f.write("- Changed from 4-bit to 8-bit quantization\n")
        f.write("- Added beam search with 4 beams\n")
        f.write("- Increased warmup steps to 15% of total steps\n")
        f.write("- Added table-only examples to reinforce FROM clauses\n")
        f.write("- Added evaluation every half-epoch\n")
        f.write("- Enabled load_best_model_at_end for better model selection\n")
        f.write("- Fixed path handling to work with Drive or local paths\n")

    print(f"Training summary saved ⇒ {report_path}")

except Exception as e:
    print("Could not write summary:", e)

print(f"Training completed in {int(h)}h {int(m)}m {int(s)}s")
