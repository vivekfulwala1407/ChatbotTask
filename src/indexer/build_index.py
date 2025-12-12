#!/usr/bin/env python3
"""
FAISS-free index builder.
Produces:
 - index/embeddings.npy
 - index/metadata.jsonl
This avoids faiss binary issues on macOS. Use the sklearn retriever for similarity search.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers.SentenceTransformer import SentenceTransformer
from typing import List, Dict, Any

MODEL_NAME = "all-MiniLM-L6-v2"

def make_doc_text(source: str, row: Dict[str, Any]) -> str:
    if source == "master_outlet":
        cols = ["outlet_id","outlet","city","total_revenue","total_units_sold","unique_customers","avg_quality_score","avg_sentiment_signal","outlet_score"]
    else:
        cols = ["product_id","product_name","category","price","units_sold_total","revenue_total","avg_quality_score","avg_sentiment_signal","product_score"]
    parts = []
    for c in cols:
        if c in row:
            parts.append(f"{c}: {row[c]}")
    return " | ".join(parts)

def build_index(data_dir: Path, out_dir: Path, model_name: str = MODEL_NAME) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mo = pd.read_csv(data_dir / "master_outlet_v2.csv")
    mp = pd.read_csv(data_dir / "master_product_v2.csv")

    docs: List[Dict[str, Any]] = []
    for _, r in mo.iterrows():
        row = r.to_dict()
        text = make_doc_text("master_outlet", row)
        docs.append({"id": f"master_outlet:{row['outlet_id']}", "source":"master_outlet", "pk": str(row['outlet_id']), "text": text, "raw": row})
    for _, r in mp.iterrows():
        row = r.to_dict()
        text = make_doc_text("master_product", row)
        docs.append({"id": f"master_product:{row['product_id']}", "source":"master_product", "pk": str(row['product_id']), "text": text, "raw": row})

    texts = [d["text"] for d in docs]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences=texts, convert_to_numpy=True, show_progress_bar=True)
    if embeddings is None:
        raise RuntimeError("Embedding generation failed")

    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "metadata.jsonl").open("w", encoding="utf8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("Index (embeddings + metadata) saved to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="folder with master_outlet_v2.csv and master_product_v2.csv")
    parser.add_argument("--out-dir", default="./index", help="output index folder")
    args = parser.parse_args()
    build_index(Path(args.data_dir), Path(args.out_dir))