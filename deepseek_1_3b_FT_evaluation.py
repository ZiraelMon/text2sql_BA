#@title { vertical-output: true }
# ════════════════════════════════════════════════════════════════
# Text-to-SQL EVALUATION – DeepSeek Coder 1.3B (Full fine-tuning)
# ════════════════════════════════════════════════════════════════

# ---------- Google Drive ----------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ---------- GPU info -------------
import torch, os, sys, json, pathlib, time
print(f"CUDA: {torch.cuda.is_available()} | device: "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# ---------- libs (same as training) -------------
!pip uninstall -y -q transformers tokenizers evaluate > /dev/null 2>&1
!pip install --no-cache-dir -q \
        "transformers[torch]==4.51.3" \
        "tokenizers>=0.19.0"          \
        "datasets>=2.18.0"            \
        "accelerate>=0.28.0"          \
        "evaluate>=0.4.1"             \
        "tqdm>=4.66.0"                \
        "nltk>=3.8.1"

if "transformers" in sys.modules: del sys.modules["transformers"]
import transformers, datasets, accelerate, evaluate
from tqdm.auto import tqdm
import nltk
nltk.download('punkt', quiet=True)

print("transformers:", transformers.__version__,
      "| datasets:", datasets.__version__,
      "| accelerate:", accelerate.__version__)

# ==========================================================
# 🔧  CONFIG
# ==========================================================
DRIVE_BASE_DIR   = "/content/drive/MyDrive/text2sql"
EXPERIMENT_NAME  = "deepseek_coder_1.3b_full_ft"
MODEL_ROOT       = f"{DRIVE_BASE_DIR}/{EXPERIMENT_NAME}"
OUTPUT_DIR       = f"{DRIVE_BASE_DIR}/eval_results/{EXPERIMENT_NAME}"
SCHEMA_FORMAT    = "sql"

# Special tokens used in training
SQL_START_TOKEN = "<SQL_START>"
SQL_END_TOKEN = "<SQL_END>"

# Generation parameters
MAX_INPUT_LEN    = 1024
MAX_NEW_TOKENS   = 256
BATCH_SIZE       = 8
NUM_BEAMS        = 4

# Paths for Spider
LOCAL_DATASET_DIR = "/content/datasets/spider"
TABLES_JSON       = f"{LOCAL_DATASET_DIR}/tables.json"
DEV_JSON          = f"{LOCAL_DATASET_DIR}/dev.json"
DB_PATH           = f"{LOCAL_DATASET_DIR}/database"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

# Clear CUDA cache before starting
torch.cuda.empty_cache()

# ==========================================================
# 1) copy Spider dev/tables from Drive -----------
# ==========================================================
def copy_from_drive(src, dst):
    if not os.path.exists(dst):
        print(f"⋄ copying {src} ➜ {dst}")
        !cp -v "{src}" "{dst}"

copy_from_drive(f"{DRIVE_BASE_DIR}/datasets/spider/tables.json", TABLES_JSON)
copy_from_drive(f"{DRIVE_BASE_DIR}/datasets/spider/dev.json", DEV_JSON)

# ==========================================================
# 2) helper functions (identical to training) ---------------
# ==========================================================
def load_tables_json(fp=TABLES_JSON):
    with open(fp, "r") as f: return {d["db_id"]: d for d in json.load(f)}

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

def create_deepseek_prompt(q_row, db_schemas):
    """Create prompt with SQL_START token."""
    schema = get_sql_schema_string(q_row['db_id'], db_schemas)
    return (
        f"### Text-to-SQL task\n"
        f"Database: {q_row['db_id']}\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {q_row['question']}\n\n"
        f"{SQL_START_TOKEN}\n"
    )

# ==========================================================
# 3) build evaluation dataset -------------------------------
# ==========================================================
print("\n⋄ Building prompts …")
db_schemas = load_tables_json()
with open(DEV_JSON) as f: dev_raw = json.load(f)

from datasets import Dataset
eval_records = [{"prompt": create_deepseek_prompt(x, db_schemas),
                 "gold":   x["query"],
                 "db_id":  x["db_id"],
                 "question": x["question"]} for x in dev_raw]
eval_ds = Dataset.from_list(eval_records)
print("dev examples:", len(eval_ds))

# ==========================================================
# 4) load model and setup tokenizer for generation ----------
# ==========================================================
print(f"\n⋄ Loading model from: {MODEL_ROOT} ...")
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# Define custom stopping criteria for SQL_END_TOKEN
class StopOnSQLEnd(StoppingCriteria):
    def __init__(self, sql_end_token_id):
        self.sql_end_token_id = sql_end_token_id

    def __call__(self, input_ids, scores, **kwargs):
        for ids in input_ids:
            if self.sql_end_token_id in ids.tolist():
                return True
        return False

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT,
        padding_side="right",
        use_fast=True
    )
    print("Tokenizer loaded successfully.")

    # Check and get special token IDs
    if SQL_START_TOKEN not in tokenizer.additional_special_tokens:
        print(f"Warning: {SQL_START_TOKEN} not found in tokenizer. Adding it.")
        special_tokens = {"additional_special_tokens": [SQL_START_TOKEN, SQL_END_TOKEN]}
        tokenizer.add_special_tokens(special_tokens)

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get SQL_END_TOKEN ID for stopping criteria
    sql_end_token_id = tokenizer.convert_tokens_to_ids(SQL_END_TOKEN)
    print(f"SQL_END_TOKEN ID: {sql_end_token_id}")

    # Setup stopping criteria
    stopping_criteria = StoppingCriteriaList([StopOnSQLEnd(sql_end_token_id)])

except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

# Load model
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ROOT,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Make sure model and tokenizer are aligned on vocab size
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        print("Resizing token embeddings to match tokenizer...")
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded successfully and placed on {device}")

except Exception as e:
    print(f"Error loading model: {e}")
    raise

# ==========================================================
# 5) batched generation ------------------------------------
# ==========================================================
print("\n⋄ Generating SQL predictions...")
predictions, references, db_ids, questions = [], [], [], []

# Performance tracking
start_time = time.time()
total_tokens = 0

# Create progress bar
progress_bar = tqdm(range(0, len(eval_ds), BATCH_SIZE), desc="Generating")

# Process batches
for start_idx in progress_bar:
    # Get batch
    end_idx = min(start_idx + BATCH_SIZE, len(eval_ds))
    batch = eval_ds[start_idx:end_idx]
    batch_size = len(batch["prompt"])

    # Keep track of metadata
    questions.extend(batch["question"])
    references.extend(batch["gold"])
    db_ids.extend(batch["db_id"])

    # Tokenize prompts
    inputs = tokenizer(
        batch["prompt"],
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
        return_tensors="pt"
    ).to(device)

    # Generate with stopping criteria
    batch_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False,
            early_stopping=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Process outputs
    batch_time = time.time() - batch_start
    generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    total_tokens += generated_tokens

    # Decode and postprocess predictions
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    # Extract SQL predictions (between SQL_START and SQL_END)
    batch_preds = []
    for text in decoded:
        # Find everything after SQL_START_TOKEN
        start_pos = text.find(SQL_START_TOKEN)
        if start_pos != -1:
            text = text[start_pos + len(SQL_START_TOKEN):].strip()

            # Remove SQL_END_TOKEN if present
            end_pos = text.find(SQL_END_TOKEN)
            if end_pos != -1:
                text = text[:end_pos].strip()

        # Clean up the prediction
        text = text.strip().rstrip(";")
        batch_preds.append(text)

    # Add to predictions
    predictions.extend(batch_preds)

    # Update progress bar with stats
    tokens_per_second = generated_tokens / (batch_time + 1e-6)
    examples_per_second = batch_size / (batch_time + 1e-6)
    progress_bar.set_postfix({
        "tok/s": f"{tokens_per_second:.1f}",
        "ex/s": f"{examples_per_second:.2f}"
    })

# Summarize generation stats
total_time = time.time() - start_time
print(f"\nGeneration completed in {total_time:.2f}s")
print(f"Average speed: {total_tokens/total_time:.1f} tokens/s")
print(f"Generated {len(predictions)} SQL predictions")

# ==========================================================
# 6) quick BLEU (diagnostic only) ---------------------------
# ==========================================================
bleu = evaluate.load("bleu").compute(
            predictions=[p for p in predictions],
            references=[[r] for r in references])["bleu"]
print(f"\nBLEU (diagnostic) : {bleu:.4f}")

# ==========================================================
# 7) save Spider-format files -------------------------------
# ==========================================================
print("\nPreparing files for evaluation and error analysis...")

# Files for Spider evaluation
gold_file = os.path.join(OUTPUT_DIR, "gold_sql.txt")
pred_file = os.path.join(OUTPUT_DIR, "pred_sql.txt")

# File for error analysis
predictions_json = os.path.join(OUTPUT_DIR, "predictions.json")

# Save files for Spider evaluation
with open(gold_file, "w", encoding="utf-8") as g, \
     open(pred_file, "w", encoding="utf-8") as p:

    for g_sql, p_sql, db in zip(references, predictions, db_ids):
        # Clean the strings *before* writing
        clean_gold = g_sql.replace("\t", " ").replace("\n", " ")
        clean_pred = p_sql.replace("\t", " ").replace("\n", " ")

        g.write(f"{clean_gold}\t{db}\n")
        p.write(f"{clean_pred}\t{db}\n")

# Save predictions.json for error analysis
predictions_data = []
for g_sql, p_sql, db, question in zip(references, predictions, db_ids, questions):
    predictions_data.append({
        "gold_sql": g_sql,
        "pred_sql": p_sql,
        "db_id": db,
        "question": question
    })

with open(predictions_json, "w", encoding="utf-8") as f:
    json.dump(predictions_data, f, indent=2)

print(f"Evaluation files written to {OUTPUT_DIR}:")
print(f"  • {os.path.basename(gold_file)}")
print(f"  • {os.path.basename(pred_file)}")
print(f"  • {os.path.basename(predictions_json)} (for error analysis)")

# ==========================================================
# 8) save evaluation summary -------------------------------
# ==========================================================
summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(f"Evaluation summary for {EXPERIMENT_NAME}\n")
    f.write(f"Model: {MODEL_ROOT}\n")
    f.write(f"Number of examples: {len(eval_ds)}\n")
    f.write(f"BLEU score: {bleu:.4f}\n")
    f.write(f"Beam search width: {NUM_BEAMS}\n")
    f.write(f"Generation parameters:\n")
    f.write(f"  - Batch size: {BATCH_SIZE}\n")
    f.write(f"  - Max input length: {MAX_INPUT_LEN}\n")
    f.write(f"  - Max new tokens: {MAX_NEW_TOKENS}\n")
    f.write(f"\nNote: This is just a preliminary evaluation. For the full assessment,\n")
    f.write(f"run the official Spider evaluation script on the generated files.\n")

print(f"Evaluation summary saved to {summary_file}")

# Display a few examples for reference
print("\n=== Sample Predictions ===")
for i in range(min(3, len(predictions))):
    print(f"\nQuestion: {questions[i]}")
    print(f"Gold: {references[i]}")
    print(f"Pred: {predictions[i]}")

print("\nRun the Spider evaluation script:")
print(f"python evaluation.py --gold {gold_file} --pred {pred_file} --db /path/to/database --table {TABLES_JSON} --etype all")
