# T5 Text-to-SQL Fine-Tuning Script
# For bachelor thesis on schema-enhanced Text-to-SQL generation
# VERSION: Disabled fp16 to debug NaN loss

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
    if 'A100' not in gpu_name and 'H100' not in gpu_name:
         print("Warning: T5-Large is memory intensive. Ensure sufficient GPU RAM.")

# Show GPU info
!nvidia-smi

# Install required packages
!pip uninstall -y -q transformers tokenizers > /dev/null 2>&1
!pip install --no-cache-dir -q \
        "transformers[torch]==4.51.3" \
        "tokenizers>=0.19.0" \
        "datasets>=2.18.0" \
        "accelerate>=0.27.0"

import pathlib, sys, importlib, torch
if "transformers" in sys.modules: del sys.modules["transformers"]
import transformers, datasets, accelerate
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
import site
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed
)

set_seed(42)

# --- Configuration ---
SCHEMA_FORMAT = "sql"
MODEL_NAME      = "google/flan-t5-large"
EXPERIMENT_NAME = "flan_t5_large_sql_bf16_adafactor_ls0.1"
MODEL_LABEL = MODEL_NAME.split("/")[-1]

# 2)  TRAINING HYPER-PARAMS
EPOCHS                     = 10
LEARNING_RATE              = 1e-4
BATCH_SIZE                 = 2
GRADIENT_ACCUMULATION_STEPS= 8
WEIGHT_DECAY               = 0.0
LABEL_SMOOTHING            = 0.1
WARMUP_RATIO               = 0.06
MAX_INPUT_LENGTH           = 1024
MAX_TARGET_LENGTH          = 256
MAX_GRAD_NORM             = 1.0
RESUME_FROM_CHECKPOINT     = False

# 3)  MIXED PRECISION
USE_FP16 = False
USE_BF16 = True

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
print(f"Schema Format: {SCHEMA_FORMAT} (with Types)")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Per Device Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Gradient Clipping: {MAX_GRAD_NORM}")
print(f"Max Input Length: {MAX_INPUT_LENGTH}")
print(f"Max Target Length: {MAX_TARGET_LENGTH}")
print(f"Warmup Ratio: {WARMUP_RATIO}")
print(f"FP16 Enabled: False")

import torch, transformers

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
            table_def = f"Table: {table}\\nColumns: {', '.join(table_columns)}"
            table_defs.append(table_def)
    table_defs.sort()
    return "\\n".join(table_defs)

def enhance_prompts_with_schema(data_df, db_schemas, schema_format="sql"):
    """Enhance input prompts with schema information."""
    enhanced_rows = []
    skipped_count = 0
    total_count = len(data_df)
    print_interval = max(1, total_count // 10)
    print(f"Enhancing {total_count} prompts...")
    for index, example in data_df.iterrows():
        if index > 0 and index % print_interval == 0: print(f"  Processed {index}/{total_count} examples...")
        db_id = example['db_id']
        try:
            if schema_format == "sql":
                schema_str = get_sql_schema_string(db_id, db_schemas) # Uses func with types
                input_text = f"translate English to SQL: {example['question']} | database: {db_id} | schema:\\n{schema_str}"
            else: input_text = f"translate English to SQL: {example['question']} | database: {db_id}"
            output_text = example['query']
            enhanced_rows.append({"input_text": input_text, "output_text": output_text})
        except Exception as e:
             print(f"Warning: Skipping example for db_id '{db_id}': {e}")
             skipped_count += 1
    print(f"  Processed {total_count}/{total_count} examples...")
    if skipped_count > 0: print(f"Skipped {skipped_count} examples.")
    if not enhanced_rows: print("Warning: No rows enhanced.")
    return pd.DataFrame(enhanced_rows)

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
def load_and_prepare_data(file_path, db_schemas_dict, schema_fmt):
    """Load, enhance and prepare dataset."""
    actual_file_path = os.path.join(LOCAL_DATASET_DIR, file_path)
    print(f"Loading data from: {actual_file_path}")
    try:
        with open(actual_file_path, 'r', encoding='utf-8') as f:
            spider_data = json.load(f)
        df = pd.DataFrame(spider_data)
        t5_data_df = enhance_prompts_with_schema(df, db_schemas_dict, schema_format=schema_fmt)
        if t5_data_df is None or t5_data_df.empty: return None
        dataset = Dataset.from_pandas(t5_data_df)
        print(f"Prepared {len(dataset)} examples from {actual_file_path}.")
        return dataset
    except Exception as e:
        print(f"Error processing data from {actual_file_path}: {e}")
        raise
print("Loading datasets...")
train_dataset = load_and_prepare_data('train_spider.json', db_schemas, SCHEMA_FORMAT)
dev_dataset = load_and_prepare_data('dev.json', db_schemas, SCHEMA_FORMAT)
if train_dataset is None or dev_dataset is None:
     raise ValueError("Failed to load train or dev dataset.")

# Log some examples to verify prompts
print("\nSample Prompts (with types):")
for i in range(min(2, len(train_dataset))):
    print(f"\n--- Example {i+1} ---")
    print(f"Input length: {len(train_dataset[i]['input_text'])} characters")
    print(f"Input: {train_dataset[i]['input_text'][:600]}...") # Show more
    print(f"Output: {train_dataset[i]['output_text']}")

# --- Load Model and Tokenizer ---
#model_name = f"t5-{MODEL_SIZE}"
print(f"\nLoading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    return_dict=True
)
print("Model and tokenizer loaded.")

# --- Tokenization Function ---
def tokenize_function(examples):
    """Tokenizes input and target text."""
    input_texts = [text if text is not None else "" for text in examples['input_text']]
    output_texts = [text if text is not None else "" for text in examples['output_text']]
    model_inputs = tokenizer(
        input_texts, max_length=MAX_INPUT_LENGTH, truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            output_texts, max_length=MAX_TARGET_LENGTH, truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Tokenize the Datasets ---
print("\nTokenizing datasets...")
tokenized_train = train_dataset.map(
    tokenize_function, batched=True, remove_columns=train_dataset.column_names, desc="Tokenizing training dataset"
)
tokenized_dev = dev_dataset.map(
    tokenize_function, batched=True, remove_columns=dev_dataset.column_names, desc="Tokenizing development dataset"
)
print(f"Training dataset tokenized: {len(tokenized_train)} examples")
print(f"Development dataset tokenized: {len(tokenized_dev)} examples")

# --- Data Collator ---
print("\nData collator initializing...")
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding="longest",
    pad_to_multiple_of=None
)
print(f"Data collator using label_pad_token_id: {data_collator.label_pad_token_id}")

# --- Training Arguments ---
print("\nConfiguring training arguments...")
steps_per_epoch = len(tokenized_train) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

training_args = Seq2SeqTrainingArguments(
    output_dir=DRIVE_OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,
    bf16= True,
    fp16=False,
    gradient_checkpointing=True,
    max_grad_norm=MAX_GRAD_NORM,
    logging_steps=100,
    save_strategy="steps",
    eval_steps=steps_per_epoch,
    save_steps=steps_per_epoch,
    logging_dir=DRIVE_LOGS_DIR,
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    optim="adafactor",
    label_smoothing_factor=LABEL_SMOOTHING,
    seed=42,
)

# --- Trainer Initialization ---
print("\nInitializing Trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    data_collator=data_collator,
    tokenizer=tokenizer,
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
    print("Running initial evaluation to check for NaN...")
    initial_eval = trainer.evaluate()
    if np.isnan(initial_eval.get("eval_loss", float("inf"))):
        print("NaN eval loss detected – continuing anyway.")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ---------- post-training actions ----------
    print("\n--- Training Finished ---")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\nSaving the final model …")
    trainer.save_model()
    trainer.save_state()
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

try:                                              # ← opens the block
    with open(report_path, "w", encoding="utf-8") as f:     # ← still inside try
        f.write(f"Experiment          : {EXPERIMENT_NAME}\n")
        f.write(f"Model               : {model_label}\n")
        f.write(f"Schema Format       : {SCHEMA_FORMAT} (with types)\n")
        f.write(f"bf16 Enabled        : {training_args.bf16}\n")
        f.write(f"Epochs Configured   : {EPOCHS}\n")
        f.write(f"Epochs Trained      : {metrics.get('epoch', 0):.2f}\n")
        f.write(f"Training Time       : {int(h)}h {int(m)}m {int(s)}s\n")
        f.write(f"Learning Rate       : {LEARNING_RATE}\n")
        f.write(f"Batch Size / Device : {BATCH_SIZE}\n")
        f.write(f"Grad Accum Steps    : {GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Effective BatchSize : {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n")
        f.write(f"Weight Decay        : {WEIGHT_DECAY}\n")
        f.write(f"Warm-up Ratio       : {WARMUP_RATIO} (≈{warmup_steps} steps)\n")
        f.write(f"Max Grad-Norm       : {MAX_GRAD_NORM}\n")
        f.write(f"Max Input Length    : {MAX_INPUT_LENGTH}\n")
        f.write(f"Max Target Length   : {MAX_TARGET_LENGTH}\n")

        # ---- training metrics ----
        f.write("\nTraining Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k:<25} : {v}\n")

        f.write(f"\nModel saved to: {training_args.output_dir}\n")
        f.write("To evaluate the model, run the evaluation script.\n")

    print(f"Training summary saved ⇒ {report_path}")

except Exception as e:
    print("Could not write summary:", e)

print(f"Training completed in {int(h)}h {int(m)}m {int(s)}s")