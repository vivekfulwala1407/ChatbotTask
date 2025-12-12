"""
Retriever using sklearn NearestNeighbors over precomputed embeddings.
Requires:
 - index/embeddings.npy
 - index/metadata.jsonl
"""
from pathlib import Path
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from sentence_transformers.SentenceTransformer import SentenceTransformer
from typing import List, Dict, Any
import numpy.typing as npt

MODEL_NAME = "all-MiniLM-L6-v2"

def load_meta(index_dir: Path) -> List[Dict[str, Any]]:
    meta_path = index_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise RuntimeError(f"Index missing: Run 'python src/indexer/build_index.py --data-dir ./output'")
    meta = []
    with meta_path.open("r", encoding="utf8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def load_embeddings(index_dir: Path) -> npt.NDArray[np.float32]:
    emb_path = index_dir / "embeddings.npy"
    if not emb_path.exists():
        raise RuntimeError(f"Index missing: Run 'python src/indexer/build_index.py --data-dir ./output'")
    return np.load(emb_path)

def embed_query_local(query: str, model_name: str = MODEL_NAME) -> npt.NDArray[np.float32]:
    model = SentenceTransformer(model_name)
    emb_result = model.encode([query], convert_to_numpy=True)
    emb_array = np.asarray(emb_result, dtype=np.float32)
    return emb_array[0]

def sanitize_float(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)

def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for key, value in d.items():
        if isinstance(value, dict):
            clean[key] = sanitize_dict(value)
        elif isinstance(value, float):
            clean[key] = sanitize_float(value)
        elif isinstance(value, (list, tuple)):
            clean[key] = [sanitize_float(v) if isinstance(v, float) else v for v in value]
        else:
            clean[key] = value
    return clean

class RetrieverSK:
    def __init__(self, index_dir: str = "./index"):
        self.index_dir = Path(index_dir)
        try:
            self.meta = load_meta(self.index_dir)
            self.embs = load_embeddings(self.index_dir)
        except RuntimeError as e:
            print(f"Warning: {e}. Falling back to no-retrieval mode.")
            self.meta = []
            self.embs = np.empty((0, 384))  # Empty for 384-dim model
            self._empty_index = True
            return
        
        if len(self.meta) == 0 or self.embs.shape[0] == 0:
            print("Warning: Empty index. Falling back to no-retrieval mode.")
            self._empty_index = True
            return
        
        self._empty_index = False
        
        # normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embs_norm = self.embs / norms
        
        self.embs_norm = np.nan_to_num(self.embs_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # NEW: Guard n_neighbors > 0
        n_neighbors = max(10, len(self.embs_norm))  # At least 1
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.nn.fit(self.embs_norm)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._empty_index or len(self.meta) == 0:
            # Fallback: Return empty or mock (no embeddings)
            return [{"raw": {"fallback": "No index available—build it first."}, "score": 0.0}]
        
        qv = embed_query_local(query)
        
        qn = np.linalg.norm(qv)
        if qn == 0 or np.isnan(qn) or np.isinf(qn):
            qv_norm = qv
        else:
            qv_norm = qv / qn
        
        qv_norm = np.nan_to_num(qv_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        qv_norm_2d = np.asarray(qv_norm, dtype=np.float32).reshape(1, -1)
        
        # Guard k > 0
        safe_k = min(k, len(self.embs_norm))
        if safe_k == 0:
            safe_k = 1  # Min 1
        
        dists, idxs = self.nn.kneighbors(qv_norm_2d, n_neighbors=safe_k)
        
        results = []
        for dist, i in zip(dists[0], idxs[0]):
            item = self.meta[int(i)].copy()
            
            similarity = 1.0 - float(dist)
            item['score'] = sanitize_float(similarity)
            
            item = sanitize_dict(item)
            
            results.append(item)
        
        return results

_retriever_instance: RetrieverSK | None = None

def retrieve(query: str, k: int = 5, index_dir: str = "./index") -> List[Dict[str, Any]]:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RetrieverSK(index_dir=index_dir)
    return _retriever_instance.retrieve(query, k=k)