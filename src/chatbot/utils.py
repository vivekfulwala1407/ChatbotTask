"""
Utility functions for best outlet/product and relevant rows (Required by Task).
"""
from typing import Dict, Any, List
from src.data_loader import load_masters

def get_best_outlet(filters: Dict[str, Any] = {}) -> Dict[str, Any]:
    mo, _ = load_masters()
    df = mo.copy()
    if city := filters.get("city"):
        df = df[df['city'].str.contains(city, case=False, na=False)]
    if df.empty:
        return {"error": "No outlets found for filters."}
    best_row = df.nlargest(1, 'performance_score').iloc[0]
    return {
        "outlet": best_row['outlet'],
        "city": best_row['city'],
        "score": best_row['performance_score'],
        "revenue": best_row['total_revenue'],
        "why": "Highest weighted performance (40% revenue + 30% customers + 30% score)"
    }

def get_best_product(filters: Dict[str, Any] = {}) -> Dict[str, Any]:
    _, mp = load_masters()
    df = mp.copy()
    if category := filters.get("category"):
        df = df[df['category'].str.contains(category, case=False, na=False)]
    if df.empty:
        return {"error": "No products found for filters."}
    best_row = df.nlargest(1, 'performance_score').iloc[0]
    return {
        "product": best_row['product_name'],
        "category": best_row['category'],
        "score": best_row['performance_score'],
        "revenue": best_row['revenue_total'],
        "why": "Highest weighted performance (40% revenue + 30% units + 30% score)"
    }

def get_relevant_rows_for_question(question: str, k: int = 5) -> List[Dict[str, Any]]:
    from .rag_engine import retrieve
    return retrieve(question, k=k)