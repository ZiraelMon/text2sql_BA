#@title { vertical-output: true }
# ==============================================
#  RELIABLE FAST E V A L U A T I O N - DeepSeek-Coder
#  (Spider dev split, schema-enhanced prompts)
# ==============================================

# ---------- Google Drive ----------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ---------- GPU info -------------
import torch, os, sys, json, pathlib, time, glob
print(f"CUDA: {torch.cuda.is_available()} | device: "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# ---------- libs (same as training) -------------
!pip uninstall -y -q transformers tokenizers peft > /dev/null 2>&1
!pip install --no-cache-dir -q \
        "transformers[torch]==4.51.3" \
        "tokenizers>=0.19.0"          \
        "datasets>=2.18.0"            \
        "accelerate>=0.28.0"          \
        "peft>=0.4.0"                 \
        "tqdm>=4.66.0"                \
        "bitsandbytes>=0.42.0"

if "transformers" in sys.modules: del sys.modules["transformers"]
import transformers, datasets, accelerate, peft
from tqdm.auto import tqdm
print("transformers:", transformers.__version__,
      "| datasets:", datasets.__version__,
      "| accelerate:", accelerate.__version__,
      "| peft:", peft.__version__)

# ==========================================================
# 🔧  C O N F I G - Optimized for A100 GPU
# ==========================================================
DRIVE_BASE_DIR   = "/content/drive/MyDrive/text2sql"
EXPERIMENT_NAME  = "deepseek_coder_6.7b_lora_v1.5"
OUT_DIR          = f"{DRIVE_BASE_DIR}/eval_results/{EXPERIMENT_NAME}"
SCHEMA_FORMAT    = "sql"

# Special tokens added during improved training
SQL_START_TOKEN = "<SQL_START>"
SQL_END_TOKEN = "<SQL_END>"

MAX_INPUT_LEN    = 1024
MAX_NEW_TOKENS   = 256
GEN_BATCH        = 4
BEAM_SIZE        = 4
SAVE_EVERY       = 100

# ==========================================================
# PATH DETECTION - Find where database files actually exist
# ==========================================================
DRIVE_DATASET_DIR = f"{DRIVE_BASE_DIR}/datasets/spider"
LOCAL_DATASET_DIR = "/content/datasets/spider"

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# Check if local dataset dir exists, create if not
if not os.path.exists(LOCAL_DATASET_DIR):
    print(f"Local dataset directory doesn't exist. Creating it...")
    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

# Define potential database paths
local_db_path = f"{LOCAL_DATASET_DIR}/database"
drive_db_path = f"{DRIVE_DATASET_DIR}/database"

# Check which path to use for tables.json and dev.json
drive_tables_path = f"{DRIVE_DATASET_DIR}/tables.json"
drive_dev_path = f"{DRIVE_DATASET_DIR}/dev.json"
local_tables_path = f"{LOCAL_DATASET_DIR}/tables.json"
local_dev_path = f"{LOCAL_DATASET_DIR}/dev.json"

# Copy tables.json and dev.json to local if needed
if not os.path.exists(local_tables_path) and os.path.exists(drive_tables_path):
    print(f"Copying tables.json to local...")
    !cp -v "{drive_tables_path}" "{local_tables_path}"

if not os.path.exists(local_dev_path) and os.path.exists(drive_dev_path):
    print(f"Copying dev.json to local...")
    !cp -v "{drive_dev_path}" "{local_dev_path}"

# Determine which database path to use
print("Checking database paths...")
# Function to count SQLite files in a directory
def count_sqlite_files(dir_path):
    if not os.path.exists(dir_path):
        return 0
    return len(glob.glob(f"{dir_path}/*/*.sqlite"))

local_db_count = count_sqlite_files(local_db_path)
drive_db_count = count_sqlite_files(drive_db_path)

print(f"Found {local_db_count} SQLite files locally")
print(f"Found {drive_db_count} SQLite files in Drive")

if local_db_count > 0:
    print(f"Using local database path: {local_db_path}")
    DB_PATH = local_db_path
    TABLES_JSON = local_tables_path
    DEV_JSON = local_dev_path
elif drive_db_count > 0:
    print(f"Using Drive database path: {drive_db_path}")
    DB_PATH = drive_db_path
    TABLES_JSON = drive_tables_path
    DEV_JSON = drive_dev_path
else:
    # If Drive has databases but local doesn't, copy from Drive to local
    if os.path.exists(drive_db_path) and os.path.isdir(drive_db_path):
        print(f"No databases found. Copying from Drive to local...")
        os.makedirs(local_db_path, exist_ok=True)
        !cp -r "{drive_db_path}" "{LOCAL_DATASET_DIR}/"
        DB_PATH = local_db_path
        TABLES_JSON = local_tables_path
        DEV_JSON = local_dev_path
    else:
        raise FileNotFoundError("No database directory found!")

print(f"Using database path: {DB_PATH}")
print(f"Using tables.json: {TABLES_JSON}")
print(f"Using dev.json: {DEV_JSON}")

# Clear CUDA cache before starting
torch.cuda.empty_cache()

# ==========================================================
# 2) helper functions (identical to training) ---------------
# ==========================================================
def load_tables_json(fp=TABLES_JSON):
    with open(fp, "r") as f: return {d["db_id"]: d for d in json.load(f)}

def sql_schema_str(db_id, db_schemas):
    """Create SQL schema string with bullet points for better clarity."""
    sch = db_schemas[db_id]
    tables, cols, types = sch["table_names_original"], sch["column_names_original"], sch["column_types"]
    pk = set(sch["primary_keys"])
    fk_map = {a:b for a,b in sch["foreign_keys"]}

    # Group columns by table
    table_columns = {t: [] for t in tables}

    for c_idx, (tbl_idx, cname) in enumerate(cols):
        if tbl_idx >= 0:  # Skip * column
            table = tables[tbl_idx]
            col_attrs = []

            # Type
            type_str = types[c_idx].upper()

            # PK
            if c_idx in pk:
                col_attrs.append("PK")

            # FK
            if c_idx in fk_map:
                ref_idx = fk_map[c_idx]
                rt, rc = cols[ref_idx]
                col_attrs.append(f"FK→{tables[rt]}.{rc}")

            # Format
            if col_attrs:
                col_str = f"{cname} ({type_str}, {', '.join(col_attrs)})"
            else:
                col_str = f"{cname} ({type_str})"

            table_columns[table].append(col_str)

    # Create bullet list
    out = []
    for t in sorted(tables):
        cols_sorted = sorted(table_columns[t])
        out.append(f"• {t} (columns: {', '.join(cols_sorted)})")

    return "\n".join(out)

# Updated prompt function to match improved training format with SQL_START token
def deepseek_prompt(q_row, db_schemas):
    """
    Matches improved training prompt format with SQL_START token and clearer formatting
    """
    schema = sql_schema_str(q_row['db_id'], db_schemas)
    return (
        f"### Text-to-SQL task\n"
        f"Database: {q_row['db_id']}\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {q_row['question']}\n\n"
        f"SQL:\n{SQL_START_TOKEN}\n"
    )

# ==========================================================
# 3) build evaluation dataset -------------------------------
# ==========================================================
print("\n⋄ Building prompts …")
db_schemas = load_tables_json()
with open(DEV_JSON) as f: dev_raw = json.load(f)

from datasets import Dataset
eval_records = [{"prompt": deepseek_prompt(x, db_schemas),
                 "gold":   x["query"],
                 "db_id":  x["db_id"],
                 "question": x["question"]} for x in dev_raw]
eval_ds = Dataset.from_list(eval_records)
print("dev examples:", len(eval_ds))

# ==========================================================
# 4) load model + LoRA adapters -----------------------------
# ==========================================================
BASE_MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, PeftConfig

# Load tokenizer first
print("\n⋄ Loading tokenizer...")
# Load tokenizer from the improved model dir to get special tokens
tokenizer = AutoTokenizer.from_pretrained(
    f"{DRIVE_BASE_DIR}/{EXPERIMENT_NAME}",  # Load from improved model dir
    padding_side="left",  # For decoder models
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get the token IDs for the special tokens
sql_start_token_id = tokenizer.convert_tokens_to_ids(SQL_START_TOKEN)
sql_end_token_id = tokenizer.convert_tokens_to_ids(SQL_END_TOKEN)
print(f"SQL_START_TOKEN ID: {sql_start_token_id}")
print(f"SQL_END_TOKEN ID: {sql_end_token_id}")

# Define custom stopping criteria for SQL_END_TOKEN
class StopOnSQLEnd(StoppingCriteria):
    def __init__(self, sql_end_token_id):
        self.sql_end_token_id = sql_end_token_id

    def __call__(self, input_ids, scores, **kwargs):
        for ids in input_ids:
            if self.sql_end_token_id in ids.tolist():
                return True
        return False

# Create stopping criteria list
stopping_criteria = StoppingCriteriaList([StopOnSQLEnd(sql_end_token_id)])

print("\n⋄ Loading base model...")
# Load base model first
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Resize token embeddings to match tokenizer with special tokens
base.resize_token_embeddings(len(tokenizer))

print("⋄ Loading LoRA adapters...")
try:
    model = PeftModel.from_pretrained(
        base,
        f"{DRIVE_BASE_DIR}/{EXPERIMENT_NAME}",
        is_trainable=False
    )
    model = model.merge_and_unload()  # merges LoRA; makes inference fast
    print("Successfully merged LoRA weights!")
except Exception as e:
    print(f"Error loading or merging LoRA weights: {e}")
    print("Using base model without LoRA weights.")
    model = base

model.eval()  # Set to evaluation mode
device = next(model.parameters()).device
print("model device:", device)

# ==========================================================
# 5) batched generation ------------------------------------
# ==========================================================
print("\n⋄ Starting generation...")
predictions, references, db_ids, questions = [], [], [], []

# Track performance statistics
generation_time = 0
total_start_time = time.time()
total_tokens_generated = 0
num_batches = 0

try:
    for start in tqdm(range(0, len(eval_ds), GEN_BATCH), desc="Generating SQL"):
        batch = eval_ds[start : start + GEN_BATCH]
        batch_size = len(batch["prompt"])

        # Tokenize with left-side padding for each batch
        tok = (
            tokenizer(
                batch["prompt"],
                max_length=MAX_INPUT_LEN,
                truncation="only_first",
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            .to(device)
        )

        # Mark start time for this batch
        batch_start_time = time.time()

        # Generate with beam search and stopping criteria
with torch.no_grad():
    outs = model.generate(
        **tok,
        do_sample=False,
        num_beams=5,                 # explore more hypotheses
        early_stopping=True,
        max_new_tokens=512,          # give it room to finish
        min_new_tokens=10,           # avoid ultra-short outputs
        stopping_criteria=stopping_criteria,
        eos_token_id=sql_end_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        length_penalty=1.2,          # favor longer, complete queries
        repetition_penalty=1.1       # discourage loops
    )

        # Measure generation time
        batch_gen_time = time.time() - batch_start_time
        generation_time += batch_gen_time

        # Slice off the prompt part – only keep generated completion
        gen_only = outs[:, tok["input_ids"].shape[1] :]
        total_tokens_generated += gen_only.numel()
        num_batches += 1

        # Decode and process
        preds_raw = tokenizer.batch_decode(gen_only, skip_special_tokens=False)
        preds = []
        for pred_text in preds_raw:
            # Extract content up to SQL_END_TOKEN
            if SQL_END_TOKEN in pred_text:
                pred_text = pred_text.split(SQL_END_TOKEN)[0].strip()

            # Clean up any leftover SQL_START_TOKEN
            if SQL_START_TOKEN in pred_text:
                pred_text = pred_text.split(SQL_START_TOKEN)[1].strip()

            # Normalize the SQL (remove trailing semicolons, etc.)
            pred_text = pred_text.strip().rstrip(';')
            preds.append(pred_text)

        # Accumulate results
        predictions.extend(preds)
        references.extend(batch["gold"])
        db_ids.extend(batch["db_id"])
        questions.extend(batch["question"])

        # Print throughput & ETA if desired
        tokens_per_second = gen_only.numel() / batch_gen_time
        examples_per_second = batch_size / batch_gen_time
        total_examples = len(predictions)
        if num_batches > 1:
            avg_examples_per_second = total_examples / (time.time() - total_start_time)
            eta_seconds = (len(eval_ds) - total_examples) / avg_examples_per_second
            eta_minutes = eta_seconds / 60
            print(
                f"\rBatch {num_batches}: "
                f"{tokens_per_second:.1f} tok/s, "
                f"{examples_per_second:.2f} ex/s, "
                f"ETA: {eta_minutes:.1f} min",
                end="",
            )

        # Save progress periodically
        if (start + GEN_BATCH) % SAVE_EVERY < GEN_BATCH or (start + GEN_BATCH) >= len(eval_ds):
            temp_gold = os.path.join(OUT_DIR, "gold_sql_temp.txt")
            temp_pred = os.path.join(OUT_DIR, "pred_sql_temp.txt")
            with open(temp_gold, "w", encoding="utf-8") as g, \
                 open(temp_pred, "w", encoding="utf-8") as p:
                for gold, pred, db in zip(references, predictions, db_ids):
                    g.write(f"{gold.replace(chr(9),' ').replace(chr(10),' ')}\t{db}\n")
                    p.write(f"{pred.replace(chr(9),' ').replace(chr(10),' ')}\t{db}\n")
            print(f"\nProgress saved: {len(predictions)}/{len(eval_ds)} examples")

except Exception as e:
    print(f"\nError during generation: {e}")
    # Save whatever was completed so far
    if predictions:
        print("Saving partial results...")
        temp_gold = os.path.join(OUT_DIR, "gold_sql_partial.txt")
        temp_pred = os.path.join(OUT_DIR, "pred_sql_partial.txt")
        with open(temp_gold, "w", encoding="utf-8") as g, \
             open(temp_pred, "w", encoding="utf-8") as p:
            for gold, pred, db in zip(references, predictions, db_ids):
                g.write(f"{gold.replace(chr(9),' ').replace(chr(10),' ')}\t{db}\n")
                p.write(f"{pred.replace(chr(9),' ').replace(chr(10),' ')}\t{db}\n")

# Print performance statistics
total_time = time.time() - total_start_time
if total_tokens_generated > 0:
    print("\n\n=== Performance Statistics ===")
    print(f"Total generation time    : {generation_time:.2f} seconds")
    print(f"Total time (incl. setup) : {total_time:.2f} seconds")
    print(f"Tokens generated         : {total_tokens_generated}")
    print(f"Tokens per second        : {total_tokens_generated / generation_time:.2f}")
    print(f"Examples per second      : {len(predictions) / generation_time:.2f}")
    print(f"Avg tokens per example   : {total_tokens_generated / len(predictions):.2f}")


# ==========================================================
# 6) write Spider evaluation files -------------------------
# ==========================================================
print("\n⋄ Writing gold / pred files for official evaluator …")
gold_file = os.path.join(OUT_DIR, "gold_sql.txt")
pred_file = os.path.join(OUT_DIR, "pred_sql.txt")

with open(gold_file, "w", encoding="utf-8") as g, \
     open(pred_file, "w", encoding="utf-8") as p:

    for gold, pred, db in zip(references, predictions, db_ids):
        g.write(f"{gold.replace(chr(9),' ').replace(chr(10),' ')}\t{db}\n")
        p.write(f"{pred.replace(chr(9),' ').replace(chr(10),' ')}\t{db}\n")

print(f"Evaluation files saved to {OUT_DIR}")
print(f"Generated {len(predictions)} predictions out of {len(eval_ds)} examples")

# Optional: Save a sample of the predictions for inspection
sample_size = min(5, len(predictions))
if sample_size > 0:
    print("\n=== Sample Predictions ===")
    for i in range(sample_size):
        print(f"\nQuestion: {questions[i]}")
        print(f"Gold: {references[i]}")
        print(f"Pred: {predictions[i]}")

# ==========================================================
# 7) run batch evaluation ---------------------------------
# ==========================================================
print("\n⋄ Running batch evaluation...")

# Create batch evaluation directory
BATCH_EVAL_DIR = os.path.join(OUT_DIR, "batch_eval")
os.makedirs(BATCH_EVAL_DIR, exist_ok=True)

# Function to split files into batches
def split_files_into_batches(gold_path, pred_path, batch_size, output_dir):
    print(f"Splitting files into batches of {batch_size}...")

    # Read files
    with open(gold_path, 'r') as g_file:
        gold_lines = g_file.readlines()

    with open(pred_path, 'r') as p_file:
        pred_lines = p_file.readlines()

    # Ensure same number of lines
    assert len(gold_lines) == len(pred_lines), "Gold and pred files have different number of lines"

    # Split into batches
    batch_files = []
    for i in range(0, len(gold_lines), batch_size):
        batch_num = i // batch_size
        end_idx = min(i + batch_size, len(gold_lines))

        batch_gold_path = os.path.join(output_dir, f"gold_batch_{batch_num}.txt")
        batch_pred_path = os.path.join(output_dir, f"pred_batch_{batch_num}.txt")

        with open(batch_gold_path, 'w') as g_out:
            g_out.writelines(gold_lines[i:end_idx])

        with open(batch_pred_path, 'w') as p_out:
            p_out.writelines(pred_lines[i:end_idx])

        batch_files.append((batch_num, batch_gold_path, batch_pred_path))

    print(f"Created {len(batch_files)} batches")
    return batch_files

# Split the files into batches
batch_files = split_files_into_batches(gold_file, pred_file, 20, BATCH_EVAL_DIR)

# Clone the Spider repository if not exists
SPIDER_EVAL_PATH = "/content/spider-eval/evaluation.py"
if not os.path.exists(SPIDER_EVAL_PATH):
    print("Cloning Spider repository...")
    !git clone https://github.com/taoyds/spider.git /content/spider-eval

# Download required NLTK data
print("Downloading NLTK data...")
!python -m nltk.downloader all

# Function to run evaluation on batch
def evaluate_batch(batch_num, gold_path, pred_path, db_path, table_path):
    print(f"\n=== Evaluating batch {batch_num} ===")

    # Run the Spider evaluator
    cmd = f"python {SPIDER_EVAL_PATH} --gold {gold_path} --pred {pred_path} --db {db_path} --table {table_path} --etype all"

    print(f"Running: {cmd}")

    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout

        # Save output to file
        output_path = os.path.join(BATCH_EVAL_DIR, f"result_batch_{batch_num}.txt")
        with open(output_path, 'w') as f:
            f.write(output)

        # Extract accuracy metrics
        import re
        exact_match = re.search(r"exact matching accuracy: (\d+\.\d+)%", output)
        execution = re.search(r"execution accuracy: (\d+\.\d+)%", output)

        exact_acc = float(exact_match.group(1)) if exact_match else 0
        exec_acc = float(execution.group(1)) if execution else 0

        print(f"Batch {batch_num} - Exact Match: {exact_acc}%, Execution: {exec_acc}%")

        return {
            'batch': batch_num,
            'exact_match': exact_acc,
            'execution': exec_acc,
            'output': output
        }
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating batch {batch_num}: {e}")
        error_path = os.path.join(BATCH_EVAL_DIR, f"error_batch_{batch_num}.txt")
        with open(error_path, 'w') as f:
            f.write(str(e))
            if e.stdout:
                f.write("\n\nSTDOUT:\n")
                f.write(e.stdout)
            if e.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(e.stderr)

        return {
            'batch': batch_num,
            'error': str(e)
        }

# Tables path for evaluation
TABLES_PATH = TABLES_JSON

# Run evaluation on each batch
results = []
for batch_num, gold_path, pred_path in batch_files:
    # Free up memory before evaluation
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Run evaluation
    result = evaluate_batch(batch_num, gold_path, pred_path, DB_PATH, TABLES_PATH)
    results.append(result)

    # Save intermediate results
    with open(os.path.join(BATCH_EVAL_DIR, "intermediate_results.txt"), 'w') as f:
        f.write(f"Processed {len(results)}/{len(batch_files)} batches\n\n")

        # Calculate averages
        exact_matches = [r.get('exact_match', 0) for r in results if 'exact_match' in r]
        executions = [r.get('execution', 0) for r in results if 'execution' in r]

        if exact_matches:
            avg_exact = sum(exact_matches) / len(exact_matches)
            f.write(f"Average Exact Match Accuracy: {avg_exact:.2f}%\n")

        if executions:
            avg_exec = sum(executions) / len(executions)
            f.write(f"Average Execution Accuracy: {avg_exec:.2f}%\n")

# Calculate overall metrics
exact_matches = [r.get('exact_match', 0) for r in results if 'exact_match' in r]
executions = [r.get('execution', 0) for r in results if 'execution' in r]

if exact_matches:
    avg_exact = sum(exact_matches) / len(exact_matches)
    print(f"\nOverall Exact Match Accuracy: {avg_exact:.2f}%")

if executions:
    avg_exec = sum(executions) / len(executions)
    print(f"Overall Execution Accuracy: {avg_exec:.2f}%")

# Save final results
with open(os.path.join(OUT_DIR, "evaluation_results.txt"), 'w') as f:
    f.write("=== SPIDER EVALUATION RESULTS ===\n\n")

    if exact_matches:
        f.write(f"Exact Match Accuracy: {avg_exact:.2f}%\n")

    if executions:
        f.write(f"Execution Accuracy: {avg_exec:.2f}%\n")

    f.write(f"\nModel: {EXPERIMENT_NAME}\n")
    f.write(f"Database Path: {DB_PATH}\n")
    f.write(f"Beam Size: {BEAM_SIZE}\n")
    f.write(f"Total Examples: {len(predictions)}\n")

    f.write("\n=== BATCH RESULTS ===\n")
    for r in results:
        if 'exact_match' in r:
            f.write(f"Batch {r['batch']}: Exact={r['exact_match']:.2f}%, Execution={r['execution']:.2f}%\n")
        else:
            f.write(f"Batch {r['batch']}: Error - {r.get('error', 'Unknown error')}\n")

print(f"\n⋄ Evaluation complete! Results saved to {os.path.join(OUT_DIR, 'evaluation_results.txt')}")

# If you want to run the official evaluator on the full dataset separately, print the command
print("\nTo run the official Spider evaluator on the full dataset in one go (may require a lot of memory):")
print(f"python {SPIDER_EVAL_PATH} --gold {gold_file} --pred {pred_file} --db {DB_PATH} --table {TABLES_PATH} --etype all")
