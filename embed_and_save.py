# Reads one sentence per line from input.txt, writes embeddings.npy + ids.csv
from transformers import AlbertTokenizer, AlbertModel
import torch, numpy as np, csv, sys, os

inp = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "emb_out"
os.makedirs(out_dir, exist_ok=True)

tok = AlbertTokenizer.from_pretrained("albert-base-v2")
mdl = AlbertModel.from_pretrained("albert-base-v2")
mdl.eval()

# load texts
with open(inp, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# batch for speed
batch_size = 32
embs = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        hidden = mdl(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    sent = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean pool
    embs.append(sent.cpu())
embs = torch.cat(embs, dim=0).numpy()

# save
np.save(os.path.join(out_dir, "embeddings.npy"), embs)
with open(os.path.join(out_dir, "ids.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["idx","text"])
    for i,t in enumerate(texts): w.writerow([i,t])

print("Saved:", os.path.join(out_dir, "embeddings.npy"), "and ids.csv")
print("Shape:", embs.shape)
