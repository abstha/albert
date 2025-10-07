# semantic_search.py
# Usage:
#   python semantic_search.py --query "your text here" --k 5
#   (optional) --emb emb_out/embeddings.npy --ids emb_out/ids.csv

import argparse, csv, numpy as np, torch
from sklearn.neighbors import NearestNeighbors
from transformers import AlbertTokenizer, AlbertModel

def load_corpus(emb_path, ids_path):
    E = np.load(emb_path)  # [N, 768]
    texts = []
    with open(ids_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:  # skip header
                continue
            texts.append(row[1])
    assert len(texts) == E.shape[0], "ids.csv rows must match embeddings rows"
    return E, texts

def embed_query(q, tok, mdl, max_len=128):
    enc = tok(q, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    with torch.no_grad():
        hidden = mdl(**enc).last_hidden_state  # [1, T, H]
    mask = enc["attention_mask"].unsqueeze(-1)  # [1, T, 1]
    sent = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean pool -> [1, H]
    return sent.numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Free-text query to search with")
    ap.add_argument("--k", type=int, default=5, help="Top-K neighbors to return")
    ap.add_argument("--emb", default="emb_out/embeddings.npy", help="Path to embeddings.npy")
    ap.add_argument("--ids", default="emb_out/ids.csv", help="Path to ids.csv")
    ap.add_argument("--model", default="albert-base-v2", help="HF model name or local dir")
    args = ap.parse_args()

    # 1) Load corpus vectors + texts
    E, texts = load_corpus(args.emb, args.ids)

    # 2) Fit cosine NN index
    nn = NearestNeighbors(metric="cosine").fit(E)

    # 3) Embed the query with ALBERT
    tok = AlbertTokenizer.from_pretrained(args.model)
    mdl = AlbertModel.from_pretrained(args.model)
    mdl.eval()
    q_vec = embed_query(args.query, tok, mdl)  # [1, H]

    # 4) Search
    dists, idxs = nn.kneighbors(q_vec, n_neighbors=min(args.k, len(E)))
    print(f"\nQuery: {args.query}\n")
    for rank, (j, dist) in enumerate(zip(idxs[0], dists[0]), 1):
        sim = 1.0 - float(dist)
        print(f"{rank:>2}. idx={j:<4}  sim={sim:.3f}  |  {texts[j]}")

if __name__ == "__main__":
    main()
