#@title { vertical-output: true }
# ==============================================
#  RELIABLE FAST E V A L U A T I O N - DeepSeek-Coder
#  (Spider dev split, schema-enhanced prompts)
# ==============================================

# ---------- Google Drive ----------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ---------- GPU info -------------
import torch, os, sys, json, pathlib, time
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
EXPERIMENT_NAME  = "deepseek_coder_6.7b_lora_v1.4"
OUT_DIR          = f"{DRIVE_BASE_DIR}/eval_results/{EXPERIMENT_NAME}"
SCHEMA_FORMAT    = "sql"

# Special tokens added during improved training
SQL_START_TOKEN = "<SQL_START>"
SQL_END_TOKEN = "<SQL_END>"

MAX_INPUT_LEN    = 1024
MAX_NEW_TOKENS   = 256
GEN_BATCH        = 4
BEAM_SIZE        = 2
SAVE_EVERY       = 100

# Paths for Spider
LOCAL_DATASET_DIR = "/content/datasets/spider"
TABLES_JSON       = f"{LOCAL_DATASET_DIR}/tables.json"
DEV_JSON          = f"{LOCAL_DATASET_DIR}/dev.json"
DB_PATH           = f"{LOCAL_DATASET_DIR}/database"


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

# Clear CUDA cache before starting
torch.cuda.empty_cache()

# ==========================================================
# 1) copy Spider dev/tables from Drive -----------
# ==========================================================
def cp(src, dst):
    if not os.path.exists(dst):
        print(f"⋄ copying {src}  ➜  {dst}")
        !cp -v "{src}" "{dst}"

cp(f"{DRIVE_BASE_DIR}/datasets/spider/tables.json", TABLES_JSON)
cp(f"{DRIVE_BASE_DIR}/datasets/spider/dev.json", DEV_JSON)

print("⋄ copying Spider database folder…")
!cp -r "{DRIVE_BASE_DIR}/datasets/spider/database" "{LOCAL_DATASET_DIR}/database"

# ==========================================================
# 2) helper functions (identical to training) ---------------
# ==========================================================
def load_tables_json(fp=TABLES_JSON):
    with open(fp, "r") as f: return {d["db_id"]: d for d in json.load(f)}

def sql_schema_str(db_id, db_schemas):
    sch = db_schemas[db_id]
    tables, cols, types = sch["table_names_original"], sch["column_names_original"], sch["column_types"]
    pk     = set(sch["primary_keys"])
    fk_map = {a:b for a,b in sch["foreign_keys"]}
    out=[]
    for t_idx, t in enumerate(tables):
        col_str=[]
        for c_idx,(tbl_idx,cname) in enumerate(cols):
            if tbl_idx!=t_idx: continue
            s=f"{cname} ({types[c_idx].upper()})"
            if c_idx in pk: s+=" (PK)"
            if c_idx in fk_map:
                ref_idx=fk_map[c_idx]; rt,rc=cols[ref_idx]
                s+=f" (FK→{tables[rt]}.{rc})"
            col_str.append(s)
        out.append(f"Table: {t}\nColumns: {', '.join(sorted(col_str))}")
    return "\n".join(out)

# Updated prompt function to match improved training format with SQL_START token
def deepseek_prompt(q_row, db_schemas):
    """
    Matches improved training prompt format with SQL_START token
    """
    schema = sql_schema_str(q_row['db_id'], db_schemas)
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

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
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
                truncation=True,
                padding=True,  # Pad to longest in batch
                return_tensors="pt",
            )
            .to(device)
        )

        # Mark start time for this batch
        batch_start_time = time.time()

        # Generate with stopping criteria
        with torch.no_grad():
            outs = model.generate(
                **tok,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                max_new_tokens=MAX_NEW_TOKENS,
                stopping_criteria=stopping_criteria,
                eos_token_id=sql_end_token_id,
                pad_token_id=tokenizer.pad_token_id,
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

print("\n⋄ Complete! To evaluate with the Spider evaluator, run:")
print(f"python /content/spider-eval/evaluation.py --gold {gold_file} --pred {pred_file} --db {DB_PATH} --etype all")
