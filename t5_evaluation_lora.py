# ==============================================
#  E V A L U A T I O N   –   T5  text-to-SQL with LoRA
#  (Spider dev split, schema-enhanced prompts)
# ==============================================

#@title { vertical-output: true}

# ---------- Google Drive ----------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ---------- GPU info -------------
import torch, os, sys, pathlib, json, time, gc, importlib, pkg_resources
print(f"CUDA: {torch.cuda.is_available()}  |  device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# ---------- libraries (same as training) -------
!pip uninstall -y -q transformers tokenizers evaluate > /dev/null 2>&1
!pip install --no-cache-dir -q \
        "transformers[torch]==4.51.3" \
        "tokenizers>=0.19.0"       \
        "datasets>=2.18.0"         \
        "accelerate>=0.27.0"       \
        "evaluate>=0.4.1"          \
        "peft>=0.4"                \
        "bitsandbytes>=0.42"

if "transformers" in sys.modules: del sys.modules["transformers"]
import transformers, datasets, accelerate, evaluate
from peft import PeftModel, PeftConfig
print("transformers:", transformers.__version__,
      "| datasets:", datasets.__version__,
      "| accelerate:", accelerate.__version__)

# ==========================================================
# 🔧  C O N F I G
# ==========================================================
DRIVE_BASE_DIR        = "/content/drive/MyDrive/text2sql"
EXPERIMENT_NAME       = "flan_t5_large_LoRA"
MODEL_ROOT            = f"{DRIVE_BASE_DIR}/{EXPERIMENT_NAME}"
BASE_MODEL_NAME       = "google/flan-t5-large"
SCHEMA_FORMAT         = "sql"
NUM_BEAMS             = 8
BATCH_SIZE_GEN        = 8
MAX_INPUT_LEN         = 1024
MAX_GEN_LEN           = 256
OUTPUT_DIR            = f"{DRIVE_BASE_DIR}/eval_results/{EXPERIMENT_NAME}"
# ----------------------------------------------------------
# LoRA specific settings
USE_BF16              = True
LOAD_IN_4BIT          = True 
MERGE_ADAPTER         = False
# ----------------------------------------------------------

LOCAL_DATASET_DIR     = "/content/datasets/spider"
TABLES_JSON           = f"{LOCAL_DATASET_DIR}/tables.json"
DEV_JSON              = f"{LOCAL_DATASET_DIR}/dev.json"
DB_PATH               = f"{LOCAL_DATASET_DIR}/database"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

# ==========================================================
# 1) copy Spider dev/tables from Drive ----------------------
# ==========================================================
def copy_from_drive(src, dst):
    if not os.path.exists(dst):
        print(f"⋄ copying {src} ➜ {dst}")
        !cp -v "{src}" "{dst}"

copy_from_drive(f"{DRIVE_BASE_DIR}/datasets/spider/tables.json", TABLES_JSON)
copy_from_drive(f"{DRIVE_BASE_DIR}/datasets/spider/dev.json",    DEV_JSON)

# ==========================================================
# 2) helpers reused from training (no extra .py file) -------
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

def enhance_prompt(q_row, db_schemas):
    return (f"translate English to SQL: {q_row['question']} | database: {q_row['db_id']} "
            f"| schema:\n{sql_schema_str(q_row['db_id'], db_schemas)}")

# ==========================================================
# 3) load dev set & build prompts ---------------------------
# ==========================================================
print("\n⋄ building prompts …")
db_schemas = load_tables_json()
with open(DEV_JSON) as f: dev_raw = json.load(f)

from datasets import Dataset
prompts = [{"input_text": enhance_prompt(x, db_schemas),
            "output_text": x["query"],
            "db_id": x["db_id"]} for x in dev_raw]
eval_ds = Dataset.from_list(prompts)
print("dev examples:", len(eval_ds))

# ==========================================================
# 4) find usable model checkpoint folder -------------------
# ==========================================================
def last_checkpoint(root):
    cks=[p for p in os.listdir(root) if p.startswith("checkpoint-")]
    return os.path.join(root, max(cks, key=lambda x:int(x.split('-')[-1]))) if cks else root

MODEL_DIR = last_checkpoint(MODEL_ROOT)
print("will load LoRA adapters from:", MODEL_DIR)

# ==========================================================
# 5) load model with LoRA adapters -------------------------
# ==========================================================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

print("\n⋄ loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Configure quantization if used during training
if LOAD_IN_4BIT:
    print("Using 4-bit quantization for the base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16
    )
else:
    # Load model without quantization
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16
    )

# Load LoRA adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

# Optionally merge adapter weights for faster inference
if MERGE_ADAPTER:
    print("Merging adapter weights with base model for faster inference...")
    model = model.merge_and_unload()

print(f"Model loaded successfully.")

# ==========================================================
# 6) batched generation -------------------------------------
# ==========================================================
print("\n⋄ generating sql predictions...")
predictions, references = [], []
for start in range(0, len(eval_ds), BATCH_SIZE_GEN):
    batch = eval_ds[start:start+BATCH_SIZE_GEN]
    tok = tokenizer(batch["input_text"],
                    max_length=MAX_INPUT_LEN,
                    truncation=True, padding=True,
                    return_tensors="pt").to(model.device)

    with torch.no_grad():
        outs = model.generate(**tok,
                              max_length=MAX_GEN_LEN,
                              num_beams=NUM_BEAMS,
                              early_stopping=True)
    predictions += tokenizer.batch_decode(outs, skip_special_tokens=True)
    references  += batch["output_text"]

# ==========================================================
# 7) quick BLEU (diagnostic only) ---------------------------
# ==========================================================
bleu = evaluate.load("bleu").compute(
            predictions=[p for p in predictions],
            references=[[r] for r in references])["bleu"]
print(f"\nBLEU (diagnostic) : {bleu:.4f}")

# ==========================================================
# 8) save Spider-format files and predictions.json ----------
# ==========================================================
print("\nPreparing files for evaluation and error analysis...")

gold_file = os.path.join(OUTPUT_DIR, "gold_sql.txt")
pred_file = os.path.join(OUTPUT_DIR, "pred_sql.txt")
predictions_json = os.path.join(OUTPUT_DIR, "predictions.json")

# Save files for Spider evaluation
with open(gold_file, "w", encoding="utf-8") as gold_f, \
     open(pred_file, "w", encoding="utf-8") as pred_f:

    # all have the same length
    for g_sql, p_sql, db in zip(references, predictions, eval_ds["db_id"]):

        # clean the strings *before* writing
        clean_gold = g_sql.replace("\t", " ").replace("\n", " ")
        clean_pred = p_sql.replace("\t", " ").replace("\n", " ")

        gold_f.write(f"{clean_gold}\t{db}\n")
        pred_f.write(f"{clean_pred}\t{db}\n")

# Save predictions.json for error analysis
predictions_data = []
for g_sql, p_sql, db in zip(references, predictions, eval_ds["db_id"]):
    predictions_data.append({
        "gold_sql": g_sql,
        "pred_sql": p_sql,
        "db_id": db
    })

with open(predictions_json, "w", encoding="utf-8") as f:
    json.dump(predictions_data, f, indent=2)

print(f"Evaluation files written to {OUTPUT_DIR}:")
print(f"  • {os.path.basename(gold_file)}")
print(f"  • {os.path.basename(pred_file)}")
print(f"  • {os.path.basename(predictions_json)} (for error analysis)")

print("\nRun the official Spider evaluation script locally:")
print(f"python evaluation.py --gold {gold_file} "
      f"--pred {pred_file} --db {DB_PATH}")

print("\nFor error analysis, run:")
print(f"python error_analysis.py  # with TARGET_EXPERIMENT_NAME set to '{EXPERIMENT_NAME}'")

# ==========================================================
# 9) save evaluation summary -------------------------------
# ==========================================================
summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(f"Evaluation summary for {EXPERIMENT_NAME}\n")
    f.write(f"Base model: {BASE_MODEL_NAME}\n")
    f.write(f"LoRA adapters: {MODEL_DIR}\n")
    f.write(f"Number of examples: {len(eval_ds)}\n")
    f.write(f"BLEU score: {bleu:.4f}\n")
    f.write(f"Beam search width: {NUM_BEAMS}\n")
    f.write(f"4-bit quantization: {LOAD_IN_4BIT}\n")
    f.write(f"BF16 precision: {USE_BF16}\n")
    f.write(f"Adapter merged: {MERGE_ADAPTER}\n")
    f.write(f"\nNote: This is just a preliminary evaluation. For the full assessment,\n")
    f.write(f"run the official Spider evaluation script on the generated files.\n")

print(f"Evaluation summary saved to {summary_file}")
