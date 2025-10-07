# Usage:
#   python embed_glue_split.py --task sst2 --split train --out_dir emb_glue/sst2_train
#   python embed_glue_split.py --task sst2 --split validation --out_dir emb_glue/sst2_dev
import os, argparse, torch, numpy as np
from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertModel

# Which text columns to use per task
TASK_TEXT_FIELDS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "mnli_matched": ("premise", "hypothesis"),
    "cola": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def join_text(ex1, ex2):
    return ex1 if ex2 is None else f"{ex1} [SEP] {ex2}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="sst2|mrpc|qqp|mnli|mnli_matched|mnli_mismatched|...")
    ap.add_argument("--split", default="train",
                    help="train|validation|validation_matched|validation_mismatched|test|...")
    ap.add_argument("--out_dir", default="emb_glue/out")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    # handle mnli special naming
    ds_name = "mnli" if args.task in ("mnli","mnli_matched","mnli_mismatched") else args.task
    split = args.split
    if args.task == "mnli_matched" and split == "validation": split = "validation_matched"
    if args.task == "mnli_mismatched" and split == "validation": split = "validation_mismatched"

    ds = load_dataset("glue", ds_name, split=split)
    f1, f2 = TASK_TEXT_FIELDS[args.task]

    texts = [join_text(ex[f1], (ex[f2] if f2 and f2 in ex else None)) for ex in ds]

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AlbertTokenizer.from_pretrained("albert-base-v2")
    mdl = AlbertModel.from_pretrained("albert-base-v2"); mdl.eval()

    embs = []
    for i in range(0, len(texts), args.batch):
        batch = texts[i:i+args.batch]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        with torch.no_grad():
            H = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        sent = (H * mask).sum(dim=1) / mask.sum(dim=1)  # mean pool
        embs.append(sent.cpu())
    E = torch.cat(embs, dim=0).numpy() if embs else np.zeros((0, 768), dtype=np.float32)

    np.save(os.path.join(args.out_dir, "embeddings.npy"), E)
    with open(os.path.join(args.out_dir, "ids.csv"), "w", encoding="utf-8") as f:
        f.write("idx,text\n")
        for i, t in enumerate(texts):
            # escape newlines/commas minimally
            t = t.replace("\n", " ")
            f.write(f"{i},{t}\n")
    print(f"Saved {E.shape} to {args.out_dir}")

if __name__ == "__main__":
    main()
