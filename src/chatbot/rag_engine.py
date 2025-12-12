from typing import List, Dict, Any, Optional, Tuple, Sequence, Hashable
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
import re
import pandas as pd
import numpy as np

# Assuming these imports work in your environment
from ..retriever.retriever import retrieve
from ..groq_client.client import groq_chat
from ..data_loader import load_masters


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class QueryType(Enum):
    """Comprehensive query classification"""
    SIMPLE_FACT = "simple_fact"
    COMPARISON = "comparison"
    AGGREGATE = "aggregate"
    RANKING = "ranking"
    ANALYTICAL = "analytical"
    LIST = "list"
    LOCATION = "location"
    DETAIL = "detail"


class EntityType(Enum):
    """Entity types in the system"""
    PRODUCT = "product"
    OUTLET = "outlet"
    CATEGORY = "category"
    CITY = "city"
    GENERAL = "general"


@dataclass
class QueryIntent:
    """Structured intent representation with all fields"""
    query_type: QueryType
    entity: EntityType
    metrics: List[str]
    filters: Dict[str, Any]
    superlative: Optional[str]
    comparison_entities: List[str]
    aggregate_type: Optional[str]
    ranking_n: Optional[int]
    needs_calculation: bool
    question_words: List[str]
    raw_question: str


# Optimized system prompt for all query types
SYSTEM_PROMPT = """You are an expert café business analyst. Provide ACCURATE, DATA-DRIVEN answers.

RULES:
1. Use ONLY the provided data - never hallucinate
2. Prefer natural-language descriptions over dense mathematical notation. Avoid using expressions like "16.13/5" or long decimal scores.
    - Use basic numbers sparingly when helpful, but favour comparative language (e.g. "much higher", "slightly lower", "strong performer").
3. Always cite specific numbers, names, and metrics when necessary
3. If data is incomplete, say "Not available in dataset"
4. Be conversational but precise
5. Format numbers with Indian notation (₹, lakhs)

RESPONSE STYLE BY QUERY TYPE:
- Simple facts: Direct answer (1 sentence)
  Example: "The Margherita Pizza costs ₹180."

- Comparisons: Side-by-side with key metrics
  Example: "Margherita Pizza: ₹180, 15K units, ₹2.7M revenue vs Pepperoni: ₹220, 12K units, ₹2.64M revenue."

- Rankings/Lists: Numbered list with key metrics
  Example: "Top 3 by revenue: 1. Tandoori Paneer Pizza (₹4.2M) 2. Classic Burger (₹3.8M) 3. Cold Coffee (₹2.1M)"

- Location queries: Full address and context
  Example: "The Green Café is located at 123 MG Road, Surat, Gujarat. Operating hours: 8 AM - 11 PM."

- Analytical: Insights with supporting data
  Example: "The top outlet succeeds due to: prime location, diverse menu (15 items), high ratings (4.6/5)."

Always start with the direct answer, then provide supporting details.
"""


def _humanize_rating(rating: float) -> str:
    """Return a short, human-friendly rating phrase and a rounded number.

    Example: 4.12 -> "about 4.1 (good)"
    """
    try:
        r = round(float(rating), 1)
    except Exception:
        return "No rating"

    if r >= 4.5:
        desc = "excellent"
    elif r >= 4.0:
        desc = "good"
    elif r >= 3.5:
        desc = "average"
    else:
        desc = "below average"

    return f"about {r} ({desc})"


def _humanize_performance(score: float) -> str:
    """Return concise performance descriptor.

    Example: 87.3 -> "strong performer (score ~87)"
    """
    try:
        s = int(round(float(score)))
    except Exception:
        return "No performance score"

    if s >= 80:
        desc = "strong performer"
    elif s >= 60:
        desc = "solid performer"
    elif s >= 40:
        desc = "moderate performance"
    else:
        desc = "needs improvement"

    return f"{desc} (score ~{s})"


def _humanize_number(value: Any, currency: bool = False) -> str:
    """Return a concise human readable number string. Keeps numbers but avoids long decimals.

    - If `currency` is True, prefix with ₹ and round to nearest rupee.
    - For large numbers, uses comma separators.
    """
    try:
        if isinstance(value, float):
            v = round(value, 2)
        else:
            v = int(value)
    except Exception:
        return str(value)

    if currency:
        if isinstance(v, float):
            return f"about ₹{v:,.2f}"
        return f"about ₹{v:,.0f}"

    if isinstance(v, int):
        return f"{v:,}"
    return str(v)

# Precompiled regex patterns
PATTERNS = {
    "outlet": re.compile(r'\b(outlet|shop|store|restaurant|café|cafe|branch|location)\b', re.I),
    "product": re.compile(r'\b(product|item|dish|menu|food|pizza|burger|coffee|fries)\b', re.I),
    "location": re.compile(r'\b(where|location|address|located|find|near|area|city|place)\b', re.I),
    "comparison": re.compile(r'\b(compare|versus|vs|between|difference|better)\b', re.I),
    "ranking": re.compile(r'\b(top|best|worst|bottom|rank|highest|lowest)\b', re.I),
    "count": re.compile(r'\b(how many|count|number of|total count)\b', re.I),
    "aggregate": re.compile(r'\b(total|sum|overall|average|mean|median)\b', re.I),
    "analytical": re.compile(r'\b(why|what makes|reason|factor|cause|correlation|trend)\b', re.I),
    "list_all": re.compile(r'\b(all|list|show|display|enumerate)\b', re.I),
}

# Metric vocabulary with expanded coverage
METRIC_VOCAB = {
    "revenue": ["revenue", "sales", "earnings", "income", "turnover"],
    "volume": ["units", "sold", "quantity", "volume", "sales volume"],
    "price": ["price", "cost", "expensive", "cheap", "value"],
    "rating": ["rating", "quality", "score", "review", "feedback", "satisfaction"],
    "customers": ["customer", "traffic", "footfall", "visitor", "patron"],
    "performance": ["performance", "success", "efficiency", "productivity"],
    "profit": ["profit", "margin", "profitability"],
    "growth": ["growth", "increase", "trend", "change"],
    "location": ["location", "address", "where", "area", "place", "situated"],
}

QUESTION_WORDS = {
    "what": ["what", "which"],
    "where": ["where"],
    "how": ["how"],
    "why": ["why"],
    "when": ["when"],
    "who": ["who"],
}

CATEGORIES = ["burger", "pizza", "french fries", "coffee"]


# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

@lru_cache(maxsize=1)
def get_masters_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache master data with comprehensive preprocessing."""
    try:
        result = load_masters()
        if not result or not isinstance(result, tuple) or len(result) < 2:
            return pd.DataFrame(), pd.DataFrame()
        
        outlets_df = result[0] if isinstance(result[0], pd.DataFrame) else pd.DataFrame()
        products_df = result[1] if isinstance(result[1], pd.DataFrame) else pd.DataFrame()
        
        # Standardize column names and fill missing values
        for df in [outlets_df, products_df]:
            if not df.empty:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                df[numeric_cols] = df[numeric_cols].fillna(0)
                
                string_cols = df.select_dtypes(include=["object"]).columns
                df[string_cols] = df[string_cols].fillna("")
        
        return outlets_df, products_df
    except Exception as e:
        print(f"Error loading masters: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ============================================================================
# ADVANCED INTENT UNDERSTANDING
# ============================================================================

def parse_query_intent(question: str) -> QueryIntent:
    """
    Comprehensive intent parsing with support for 100+ query types.
    """
    q_lower = question.lower().strip()
    
    # Initialize
    query_type = QueryType.SIMPLE_FACT
    entity = EntityType.PRODUCT
    metrics: List[str] = []
    filters: Dict[str, Any] = {}
    superlative: Optional[str] = None
    comparison_entities: List[str] = []
    aggregate_type: Optional[str] = None
    ranking_n: Optional[int] = None
    needs_calculation = False
    question_words: List[str] = []
    
    # 1. Extract question words
    for q_type, words in QUESTION_WORDS.items():
        if any(w in q_lower for w in words):
            question_words.append(q_type)
    
    # 2. Determine entity type
    if PATTERNS["outlet"].search(question):
        entity = EntityType.OUTLET
    elif PATTERNS["product"].search(question):
        entity = EntityType.PRODUCT
    
    # 3. Detect query type (priority order matters)
    
    # Location queries (HIGHEST PRIORITY)
    if PATTERNS["location"].search(question) or "where" in question_words:
        query_type = QueryType.LOCATION
        entity = EntityType.OUTLET
    
    # Comparison queries
    elif PATTERNS["comparison"].search(question):
        query_type = QueryType.COMPARISON
        comp_match = re.search(r'(?:compare|vs|versus|between)\s+(.+?)\s+(?:and|vs|versus|with|,)\s+(.+?)(?:\?|$|by)', question, re.I)
        if comp_match:
            comparison_entities = [comp_match.group(1).strip(), comp_match.group(2).strip()]
    
    # Count/Aggregate queries
    elif PATTERNS["count"].search(question):
        query_type = QueryType.AGGREGATE
        aggregate_type = "count"
    
    elif PATTERNS["aggregate"].search(question):
        query_type = QueryType.AGGREGATE
        if "total" in q_lower or "sum" in q_lower:
            aggregate_type = "total"
        elif "average" in q_lower or "mean" in q_lower:
            aggregate_type = "average"
        elif "median" in q_lower:
            aggregate_type = "median"
    
    # Ranking queries
    elif PATTERNS["ranking"].search(question):
        query_type = QueryType.RANKING
        
        rank_match = re.search(r'\b(?:top|best|worst|bottom)\s+(\d+)\b', q_lower)
        if rank_match:
            ranking_n = int(rank_match.group(1))
        else:
            ranking_n = 1 if re.search(r'\b(best|worst|top|highest|lowest)\b(?!\s+\d)', q_lower) else 5
        
        if re.search(r'\b(best|top|highest|most)\b', q_lower):
            superlative = "max"
        elif re.search(r'\b(worst|bottom|lowest|least)\b', q_lower):
            superlative = "min"
    
    # Analytical queries
    elif PATTERNS["analytical"].search(question):
        query_type = QueryType.ANALYTICAL
        needs_calculation = True
    
    # List queries
    elif PATTERNS["list_all"].search(question):
        query_type = QueryType.LIST
    
    # Detail queries
    elif any(word in q_lower for word in ["details", "about", "information", "tell me"]):
        query_type = QueryType.DETAIL
    
    # 4. Extract metrics
    for metric, keywords in METRIC_VOCAB.items():
        if any(kw in q_lower for kw in keywords):
            metrics.append(metric)
    
    # 5. Extract filters
    outlets_df, products_df = get_masters_data()
    
    # City filter
    if not outlets_df.empty and 'city' in outlets_df.columns:
        cities = outlets_df['city'].dropna().unique()
        for city in cities:
            city_str = str(city).lower()
            if city_str in q_lower:
                filters["city"] = str(city).title()
                break
    
    # Category filter
    for category in CATEGORIES:
        if category in q_lower:
            filters["category"] = category.title()
            break
    
    # Price range filter
    price_range_match = re.search(r'between\s+₹?(\d+)(?:\s*-\s*|\s+and\s+)₹?(\d+)', q_lower)
    if price_range_match:
        filters["price_min"] = int(price_range_match.group(1))
        filters["price_max"] = int(price_range_match.group(2))
    
    # Rating filter
    rating_match = re.search(r'(?:rating|score)(?:\s+)?(?:above|over|greater than)\s+(\d+(?:\.\d+)?)', q_lower)
    if rating_match:
        filters["rating_min"] = float(rating_match.group(1))
    
    # 6. Extract specific outlet names from the question
    if query_type == QueryType.LOCATION or entity == EntityType.OUTLET:
        # Try to extract outlet name more intelligently
        if not outlets_df.empty and 'outlet' in outlets_df.columns:
            outlet_names = outlets_df['outlet'].dropna().unique()
            for outlet_name in outlet_names:
                outlet_str = str(outlet_name).lower()
                # Check if outlet name appears in question
                if outlet_str in q_lower:
                    comparison_entities.append(str(outlet_name))
                    break
        
        # Also look for quoted names
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        if quoted:
            comparison_entities.extend(quoted)
    
    return QueryIntent(
        query_type=query_type,
        entity=entity,
        metrics=metrics,
        filters=filters,
        superlative=superlative,
        comparison_entities=comparison_entities,
        aggregate_type=aggregate_type,
        ranking_n=ranking_n,
        needs_calculation=needs_calculation,
        question_words=question_words,
        raw_question=q_lower
    )


# ============================================================================
# INTELLIGENT DATA SELECTION
# ============================================================================

def _convert_dict_keys_to_str(record: Dict[Any, Any]) -> Dict[str, Any]:
    """Convert dictionary keys to strings for type safety."""
    return {str(k): v for k, v in record.items()}


def dataframe_to_evidence(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to evidence format with type safety."""
    if df is None or df.empty:
        return []
    
    raw_records: Sequence[Dict[Hashable, Any]] = df.to_dict(orient="records")
    records: List[Dict[str, Any]] = [
        _convert_dict_keys_to_str(record) for record in raw_records
    ]
    return [{"raw": record, "score": 1.0} for record in records]


def select_relevant_data(
    evidence: List[Dict[str, Any]], 
    intent: QueryIntent
) -> List[Dict[str, Any]]:
    """
    Advanced data selection supporting all query types.
    KEY FIX: Always use outlets_df for location queries.
    """
    outlets_df, products_df = get_masters_data()
    
    # CRITICAL FIX: For location queries, ALWAYS use outlets_df
    if intent.query_type == QueryType.LOCATION or intent.entity == EntityType.OUTLET:
        df = outlets_df
    else:
        df = products_df if intent.entity == EntityType.PRODUCT else outlets_df
    
    if df.empty:
        return evidence[:5] if evidence else []
    
    # Apply filters first
    filtered_df = _apply_comprehensive_filters(df, intent)
    
    # Route to appropriate handler
    if intent.query_type == QueryType.LOCATION:
        return _handle_location_query(filtered_df, intent, evidence)
    
    elif intent.query_type == QueryType.AGGREGATE:
        return _handle_aggregate_query(filtered_df, intent)
    
    elif intent.query_type == QueryType.COMPARISON:
        return _handle_comparison_query(filtered_df, intent, evidence)
    
    elif intent.query_type == QueryType.RANKING:
        return _handle_ranking_query(filtered_df, intent)
    
    elif intent.query_type == QueryType.LIST:
        return _handle_list_query(filtered_df, intent)
    
    elif intent.query_type == QueryType.ANALYTICAL:
        return _handle_analytical_query(filtered_df, intent, evidence)
    
    elif intent.query_type in [QueryType.SIMPLE_FACT, QueryType.DETAIL]:
        return _handle_detail_query(filtered_df, intent, evidence)
    
    return dataframe_to_evidence(_get_top_performers(filtered_df, 5))


def _apply_comprehensive_filters(df: pd.DataFrame, intent: QueryIntent) -> pd.DataFrame:
    """Apply all filters from intent."""
    filtered = df.copy()
    
    if city := intent.filters.get("city"):
        if 'city' in filtered.columns:
            filtered = filtered[filtered['city'].astype(str).str.lower() == city.lower()]
    
    if category := intent.filters.get("category"):
        if 'category' in filtered.columns:
            filtered = filtered[
                filtered['category'].astype(str).str.lower().str.contains(category.lower(), na=False)
            ]
    
    if 'price' in filtered.columns:
        if price_min := intent.filters.get("price_min"):
            filtered = filtered[filtered['price'] >= price_min]
        if price_max := intent.filters.get("price_max"):
            filtered = filtered[filtered['price'] <= price_max]
    
    if rating_col := _find_rating_column(filtered):
        if rating_min := intent.filters.get("rating_min"):
            filtered = filtered[filtered[rating_col] >= rating_min]
    
    return filtered if not filtered.empty else df


def _handle_location_query(
    df: pd.DataFrame, 
    intent: QueryIntent, 
    evidence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    ENHANCED: Handle location/address queries with better outlet matching.
    """
    
    # Strategy 1: If specific outlet mentioned, find it by name
    if intent.comparison_entities:
        for entity_name in intent.comparison_entities:
            if 'outlet' in df.columns:
                # Exact match first
                exact_match = df[df['outlet'].astype(str).str.lower() == entity_name.lower()]
                if not exact_match.empty:
                    return dataframe_to_evidence(exact_match.head(1))
                
                # Partial match
                mask = df['outlet'].astype(str).str.contains(entity_name, case=False, na=False, regex=False)
                matches = df[mask]
                if not matches.empty:
                    return dataframe_to_evidence(matches.head(1))
    
    # Strategy 2: Check if city filter is applied, return all outlets in that city
    if intent.filters.get("city"):
        return dataframe_to_evidence(df.head(10))
    
    # Strategy 3: Try to use evidence if it has outlet info
    if evidence:
        for ev in evidence:
            raw = ev.get("raw", {})
            if any(key in raw for key in ['outlet', 'address', 'city', 'location']):
                return [ev]
    
    # Strategy 4: Return all outlets with full details (default for "show all locations")
    return dataframe_to_evidence(df.head(20))


def _handle_aggregate_query(df: pd.DataFrame, intent: QueryIntent) -> List[Dict[str, Any]]:
    """Handle aggregate calculations."""
    agg_type = intent.aggregate_type
    result = {}
    
    if agg_type == "count":
        result["count"] = len(df)
        result["entity_type"] = intent.entity.value
    
    elif agg_type == "total":
        rev_col = _find_revenue_column(df)
        if rev_col:
            result["total_revenue"] = int(df[rev_col].sum())
        
        if 'units_sold_total' in df.columns or 'total_units_sold' in df.columns:
            units_col = 'units_sold_total' if 'units_sold_total' in df.columns else 'total_units_sold'
            result["total_units"] = int(df[units_col].sum())
    
    elif agg_type == "average":
        if "price" in intent.metrics and 'price' in df.columns:
            result["average_price"] = round(float(df['price'].mean()), 2)
        
        rev_col = _find_revenue_column(df)
        if rev_col:
            result["average_revenue"] = round(float(df[rev_col].mean()), 2)
    
    elif agg_type == "median":
        if "price" in intent.metrics and 'price' in df.columns:
            result["median_price"] = round(float(df['price'].median()), 2)
    
    return [{"raw": result, "score": 1.0}]


def _handle_comparison_query(
    df: pd.DataFrame, 
    intent: QueryIntent, 
    evidence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Handle comparison queries."""
    if not intent.comparison_entities:
        return dataframe_to_evidence(_get_top_performers(df, 2))
    
    matches: List[Dict[str, Any]] = []
    search_cols = [col for col in ['product_name', 'outlet'] if col in df.columns]
    
    if search_cols:
        for entity in intent.comparison_entities:
            if not entity.strip():
                continue
            
            mask = pd.Series(False, index=df.index)
            for col in search_cols:
                mask |= df[col].astype(str).str.contains(entity, case=False, na=False, regex=False)
            
            if mask.any():
                matched_records: Sequence[Dict[Hashable, Any]] = df[mask].head(1).to_dict(orient="records")
                typed_matches: List[Dict[str, Any]] = [
                    _convert_dict_keys_to_str(record) for record in matched_records
                ]
                matches.extend(typed_matches)
    
    if matches:
        return [{"raw": m, "score": 1.0} for m in matches]
    
    return dataframe_to_evidence(_get_top_performers(df, 2))


def _handle_ranking_query(df: pd.DataFrame, intent: QueryIntent) -> List[Dict[str, Any]]:
    """Handle ranking queries."""
    n = intent.ranking_n or 5
    
    sort_col = _determine_sort_column(df, intent)
    
    if sort_col and sort_col in df.columns:
        ascending = (intent.superlative == "min")
        sorted_df = df.sort_values(sort_col, ascending=ascending)
        return dataframe_to_evidence(sorted_df.head(n))
    
    return dataframe_to_evidence(_get_top_performers(df, n))


def _handle_list_query(df: pd.DataFrame, intent: QueryIntent) -> List[Dict[str, Any]]:
    """Handle list all queries."""
    max_results = 50
    return dataframe_to_evidence(df.head(max_results))


def _handle_analytical_query(
    df: pd.DataFrame, 
    intent: QueryIntent, 
    evidence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Handle analytical queries."""
    top_df = _get_top_performers(df, 3)
    result_data = dataframe_to_evidence(top_df)
    
    summary = {}
    if 'performance_score' in df.columns:
        summary["avg_performance"] = round(float(df['performance_score'].mean()), 2)
        summary["std_performance"] = round(float(df['performance_score'].std()), 2)
    
    if summary:
        result_data.insert(0, {"raw": {"summary_stats": summary}, "score": 1.0})
    
    return result_data


def _handle_detail_query(
    df: pd.DataFrame, 
    intent: QueryIntent, 
    evidence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Handle simple fact and detail queries."""
    
    if intent.comparison_entities:
        for entity_name in intent.comparison_entities:
            search_cols = [col for col in ['product_name', 'outlet'] if col in df.columns]
            
            for col in search_cols:
                mask = df[col].astype(str).str.contains(entity_name, case=False, na=False, regex=False)
                matches = df[mask]
                if not matches.empty:
                    return dataframe_to_evidence(matches.head(1))
    
    if evidence and len(evidence) > 0:
        return evidence[:1]
    
    return dataframe_to_evidence(_get_top_performers(df, 1))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _determine_sort_column(df: pd.DataFrame, intent: QueryIntent) -> Optional[str]:
    """Determine sort column based on metrics."""
    metrics = intent.metrics
    
    if "price" in metrics and 'price' in df.columns:
        return 'price'
    
    if "revenue" in metrics:
        return _find_revenue_column(df)
    
    if "volume" in metrics:
        for col in ['units_sold_total', 'total_units_sold', 'units_sold']:
            if col in df.columns:
                return col
    
    if "rating" in metrics:
        return _find_rating_column(df)
    
    if "customers" in metrics and 'unique_customers' in df.columns:
        return 'unique_customers'
    
    if 'performance_score' in df.columns:
        return 'performance_score'
    
    return None


def _find_revenue_column(df: pd.DataFrame) -> Optional[str]:
    """Find revenue column name."""
    for col in ['revenue_total', 'total_revenue', 'revenue']:
        if col in df.columns:
            return col
    return None


def _find_rating_column(df: pd.DataFrame) -> Optional[str]:
    """Find rating column name."""
    for col in ['avg_quality_score', 'product_score', 'outlet_score', 'rating', 'quality_score']:
        if col in df.columns:
            return col
    return None


def _get_top_performers(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Get top N performers."""
    if 'performance_score' in df.columns and not df.empty:
        return df.nlargest(n, 'performance_score')
    return df.head(n)


# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def build_context_for_llm(evidence: List[Dict[str, Any]], intent: QueryIntent) -> str:
    """Build rich context optimized for each query type."""
    if not evidence:
        return "No data available."
    
    if intent.query_type == QueryType.LOCATION:
        return _build_location_context(evidence)
    
    elif intent.query_type == QueryType.AGGREGATE:
        return _build_aggregate_context(evidence, intent)
    
    elif intent.query_type == QueryType.COMPARISON:
        return _build_comparison_context(evidence)
    
    elif intent.query_type == QueryType.RANKING:
        return _build_ranking_context(evidence, intent)
    
    elif intent.query_type == QueryType.LIST:
        return _build_list_context(evidence, intent)
    
    elif intent.query_type == QueryType.ANALYTICAL:
        return _build_analytical_context(evidence, intent)
    
    else:
        return _build_detail_context(evidence, intent)


def _build_location_context(evidence: List[Dict[str, Any]]) -> str:
    """
    ENHANCED: Build comprehensive location context with ALL available fields.
    """
    lines = ["OUTLET LOCATION INFORMATION:"]
    lines.append("=" * 60)
    
    for i, ev in enumerate(evidence, 1):
        raw = ev.get("raw", {})
        outlet = raw.get('outlet', 'Unknown Outlet')
        
        lines.append(f"\n{i}. {outlet}")
        lines.append("-" * 60)
        
        # Address details (check multiple possible column names)
        address_fields = ['address', 'full_address', 'street_address', 'location']
        for field in address_fields:
            if address := raw.get(field, ''):
                lines.append(f"   📍 Address: {address}")
                break
        
        # City
        if city := raw.get('city', ''):
            lines.append(f"   🏙️  City: {city}")
        
        # Area/Locality
        area_fields = ['area', 'locality', 'neighborhood', 'zone']
        for field in area_fields:
            if area := raw.get(field, ''):
                lines.append(f"   📌 Area: {area}")
                break
        
        # State
        if state := raw.get('state', ''):
            lines.append(f"   🗺️  State: {state}")
        
        # Pincode
        pincode_fields = ['pincode', 'pin_code', 'postal_code', 'zip']
        for field in pincode_fields:
            if pincode := raw.get(field, ''):
                lines.append(f"   📮 Pincode: {pincode}")
                break
        
        # Country
        if country := raw.get('country', ''):
            lines.append(f"   🌍 Country: {country}")
        
        lines.append("")
        
        # Additional operational details
        lines.append("   OPERATIONAL DETAILS:")
        
        # Operating hours
        hours_fields = ['operating_hours', 'business_hours', 'hours', 'timings']
        for field in hours_fields:
            if hours := raw.get(field, ''):
                lines.append(f"   ⏰ Hours: {hours}")
                break
        
        # Contact
        phone_fields = ['phone', 'contact', 'mobile', 'phone_number', 'contact_number']
        for field in phone_fields:
            if phone := raw.get(field, ''):
                lines.append(f"   ☎️  Phone: {phone}")
                break
        
        # Email
        if email := raw.get('email', ''):
            lines.append(f"   📧 Email: {email}")
        
        # Additional info
        if manager := raw.get('manager', ''):
            lines.append(f"   👤 Manager: {manager}")
        
        if rating_col := _find_rating_column(pd.DataFrame([raw])):
            if rating := raw.get(rating_col, 0):
                lines.append(f"   ⭐ Rating: {_humanize_rating(rating)}")
        
        if customers := raw.get('unique_customers', 0):
            if customers > 0:
                lines.append(f"   👥 Customers: {_humanize_number(customers)}")
        
        if revenue := raw.get('revenue_total') or raw.get('total_revenue', 0):
            if revenue > 0:
                lines.append(f"   💰 Revenue: {_humanize_number(revenue, currency=True)}")
        
        if perf := raw.get('performance_score', 0):
            if perf > 0:
                lines.append(f"   📊 Performance: {_humanize_performance(perf)}")
        
        lines.append("")
    
    return "\n".join(lines)


def _build_aggregate_context(evidence: List[Dict[str, Any]], intent: QueryIntent) -> str:
    """Build context for aggregate queries."""
    if not evidence:
        return "No aggregate data available."
    
    raw = evidence[0].get("raw", {})
    lines = ["AGGREGATE RESULTS:"]
    
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            if "revenue" in key:
                lines.append(f"{key.replace('_', ' ').title()}: ₹{value:,.0f}")
            else:
                lines.append(f"{key.replace('_', ' ').title()}: {value:,}")
        else:
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(lines)


def _build_comparison_context(evidence: List[Dict[str, Any]]) -> str:
    """Build context for comparison queries."""
    if len(evidence) < 2:
        return _build_detail_context(evidence, None)
    
    lines = ["COMPARISON DATA:"]
    
    for i, ev in enumerate(evidence, 1):
        raw = ev.get("raw", {})
        name = raw.get('product_name') or raw.get('outlet', 'Unknown')
        
        lines.append(f"\n{i}. {name}")
        
        if price := raw.get('price'):
            lines.append(f"   Price: ₹{price}")
        
        if category := raw.get('category'):
            lines.append(f"   Category: {category}")
        
        revenue = raw.get('revenue_total') or raw.get('total_revenue', 0)
        lines.append(f"   Revenue: ₹{revenue:,.0f}")
        
        units = raw.get('units_sold_total') or raw.get('total_units_sold', 0)
        if units:
            lines.append(f"   Units Sold: {units:,}")
        
        rating = raw.get('avg_quality_score') or raw.get('product_score') or raw.get('outlet_score', 0)
        if rating:
            lines.append(f"   Rating: {_humanize_rating(rating)}")
        
        if customers := raw.get('unique_customers'):
            lines.append(f"   Customers: {_humanize_number(customers)}")
        
        if perf := raw.get('performance_score'):
            lines.append(f"   Performance: {_humanize_performance(perf)}")
    
    return "\n".join(lines)


def _build_ranking_context(evidence: List[Dict[str, Any]], intent: QueryIntent) -> str:
    """Build context for ranking queries."""
    lines = [f"TOP {len(evidence)} RANKED RESULTS:"]
    
    for i, ev in enumerate(evidence, 1):
        raw = ev.get("raw", {})
        name = raw.get('product_name') or raw.get('outlet', f"Item {i}")
        
        lines.append(f"\n{i}. {name}")
        
        if price := raw.get('price'):
            lines.append(f"   Price: ₹{price}")
        
        revenue = raw.get('revenue_total') or raw.get('total_revenue', 0)
        if revenue:
            lines.append(f"   Revenue: ₹{revenue:,.0f}")
        
        units = raw.get('units_sold_total') or raw.get('total_units_sold', 0)
        if units:
            lines.append(f"   Units: {units:,}")
        
        rating = raw.get('avg_quality_score') or raw.get('product_score') or raw.get('outlet_score')
        if rating:
            lines.append(f"   Rating: {_humanize_rating(rating)}")
        
        if perf := raw.get('performance_score'):
            lines.append(f"   Performance: {_humanize_performance(perf)}")
    
    return "\n".join(lines)


def _build_list_context(evidence: List[Dict[str, Any]], intent: QueryIntent) -> str:
    """Build context for list queries."""
    lines = [f"COMPLETE LIST ({len(evidence)} items):"]
    
    for i, ev in enumerate(evidence, 1):
        raw = ev.get("raw", {})
        name = raw.get('product_name') or raw.get('outlet', f"Item {i}")
        
        info_parts = []
        if price := raw.get('price'):
            info_parts.append(f"₹{price}")
        
        if category := raw.get('category'):
            info_parts.append(category)
        
        rating = raw.get('avg_quality_score') or raw.get('product_score') or raw.get('outlet_score')
        if rating:
            info_parts.append(f"{_humanize_rating(rating)}")
        
        info_str = " | ".join(info_parts) if info_parts else ""
        lines.append(f"{i}. {name} {f'- {info_str}' if info_str else ''}")
    
    return "\n".join(lines)


def _build_analytical_context(evidence: List[Dict[str, Any]], intent: QueryIntent) -> str:
    """Build context for analytical queries."""
    lines = ["ANALYTICAL DATA:"]
    
    if evidence and "summary_stats" in evidence[0].get("raw", {}):
        summary = evidence[0]["raw"]["summary_stats"]
        lines.append("\nOVERALL STATISTICS:")
        for key, value in summary.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        lines.append("\nTOP PERFORMERS:")
        evidence = evidence[1:]
    
    for i, ev in enumerate(evidence, 1):
        raw = ev.get("raw", {})
        name = raw.get('product_name') or raw.get('outlet', f"Item {i}")
        
        lines.append(f"\n{i}. {name}")
        
        for key, value in raw.items():
            if key in ['product_name', 'outlet']:
                continue
            
            key_low = key.lower()
            if isinstance(value, float):
                if 'rating' in key_low:
                    lines.append(f"   {key.replace('_', ' ').title()}: {_humanize_rating(value)}")
                elif 'performance' in key_low or 'score' in key_low:
                    lines.append(f"   {key.replace('_', ' ').title()}: {_humanize_performance(value)}")
                elif 'revenue' in key_low:
                    lines.append(f"   {key.replace('_', ' ').title()}: {_humanize_number(value, currency=True)}")
                else:
                    lines.append(f"   {key.replace('_', ' ').title()}: {round(value,2)}")
            elif isinstance(value, int):
                if 'revenue' in key_low:
                    lines.append(f"   {key.replace('_', ' ').title()}: {_humanize_number(value, currency=True)}")
                else:
                    lines.append(f"   {key.replace('_', ' ').title()}: {_humanize_number(value)}")
            else:
                lines.append(f"   {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(lines)


def _build_detail_context(evidence: List[Dict[str, Any]], intent: Optional[QueryIntent]) -> str:
    """Build context for detail queries."""
    if not evidence:
        return "No data available."
    
    raw = evidence[0].get("raw", {})
    name = raw.get('product_name') or raw.get('outlet', 'Item')
    
    lines = [f"DETAILS FOR: {name}"]
    lines.append("=" * 60)
    
    for key, value in raw.items():
        if key in ['product_name', 'outlet']:
            continue
        
        display_key = key.replace('_', ' ').title()
        
        if isinstance(value, float):
            key_low = key.lower()
            if 'rating' in key_low:
                lines.append(f"{display_key}: {_humanize_rating(value)}")
            elif 'performance' in key_low or 'score' in key_low:
                lines.append(f"{display_key}: {_humanize_performance(value)}")
            else:
                lines.append(f"{display_key}: {round(value,2)}")
        elif isinstance(value, int):
            if 'revenue' in key.lower():
                lines.append(f"{display_key}: ₹{value:,}")
            elif value > 0:
                lines.append(f"{display_key}: {value:,}")
        elif value:
            lines.append(f"{display_key}: {value}")
    
    return "\n".join(lines)


# ============================================================================
# MAIN QUERY HANDLER
# ============================================================================

def answer_query(question: str) -> str:
    """
    Main entry point for answering queries.
    Handles all 100+ query types with intelligent routing.
    Returns clean, human-readable natural language answers.
    """
    try:
        # Step 1: Parse intent
        intent = parse_query_intent(question)
        
        # Step 2: Retrieve relevant data
        evidence = retrieve(question) if question else []
        
        # Step 3: Select and filter relevant data
        relevant_data = select_relevant_data(evidence, intent)
        
        # Step 4: Build context
        context = build_context_for_llm(relevant_data, intent)
        
        # Step 5: Generate answer using LLM
        prompt = f"""Question: {question}

{context}

Provide a clear, accurate answer based ONLY on the data above."""
        
        response = groq_chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract clean text from response
        answer = _extract_text_from_response(response)
        
        return answer
        
    except Exception as e:
        return f"Error processing query: {str(e)}"


def _extract_text_from_response(response: Any) -> str:
    """
    Extract clean text from various response formats.
    Handles: dict, string, or API response objects.
    """
    # If response is already a clean string, return it
    if isinstance(response, str):
        # Check if it looks like a JSON string
        if response.strip().startswith('{'):
            try:
                import json
                response = json.loads(response)
            except:
                return response
        else:
            return response
    
    # If response is a dict (parsed API response)
    if isinstance(response, dict):
        # Try standard OpenAI/Groq format
        if 'choices' in response and len(response['choices']) > 0:
            message = response['choices'][0].get('message', {})
            content = message.get('content', '')
            if content:
                return content.strip()
        
        # Try alternative formats
        if 'content' in response:
            return response['content'].strip()
        
        if 'text' in response:
            return response['text'].strip()
        
        if 'answer' in response:
            return response['answer'].strip()
    
    # Fallback: convert to string
    return str(response)