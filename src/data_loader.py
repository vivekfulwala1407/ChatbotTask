"""
Loads master tables used for RAG. Minimal cleaning only.
"""
import pandas as pd
import numpy as np
from .config import Config

def load_masters():
    mo = pd.read_csv(Config.MASTER_OUTLET)
    mp = pd.read_csv(Config.MASTER_PRODUCT)
    
    # String standardization
    string_cols_mo = ['outlet', 'city', 'address', 'main_promotion_method', 'quality_status_most_common']
    for col in string_cols_mo:
        if col in mo.columns:
            mo[col] = mo[col].astype(str).str.strip().str.lower() if col != 'outlet' else mo[col].astype(str).str.strip()
    
    string_cols_mp = ['category', 'product_name']
    for col in string_cols_mp:
        if col in mp.columns:
            mp[col] = mp[col].astype(str).str.strip().str.lower()
    
    # Handle inf/NaN
    mo = mo.replace([np.inf, -np.inf], np.nan)
    mp = mp.replace([np.inf, -np.inf], np.nan)
    for df in [mo, mp]:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('unknown')
            else:
                df[col] = df[col].fillna(0)
    
    # Derived fields
    mo['performance_score'] = (mo['total_revenue'] * 0.4 + mo['unique_customers'] * 0.3 + mo['outlet_score'] * 0.3 * 100)
    mp['performance_score'] = (mp['revenue_total'] * 0.4 + mp['units_sold_total'] * 0.3 + mp['product_score'] * 0.3 * 100)
    
    # BI Enhancements: Cached groupbys/aggs
    mp['category_rev'] = mp.groupby('category')['revenue_total'].transform('sum')
    mp['category_units'] = mp.groupby('category')['units_sold_total'].transform('sum')
    mo['city_rev'] = mo.groupby('city')['total_revenue'].transform('sum')
    mo['city_customers'] = mo.groupby('city')['unique_customers'].transform('sum')
    
    # Stats cache
    mp['price_rating_corr'] = mp['price'].corr(mp['avg_quality_score'])  # Constant for all rows
    mo['rev_std'] = mo['total_revenue'].std()
    mp['avg_order_value'] = mp['revenue_total'].sum() / mp['units_sold_total'].sum() * mo['unique_customers'].sum()  # Proxy avg order
    mp['is_premium'] = mp['price'] > 50  # Derived for #48
    mo['needs_improvement'] = mo['outlet_score'] < 0.5  # For #27
    
    return mo, mp