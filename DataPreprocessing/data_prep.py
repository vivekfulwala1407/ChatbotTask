"""
data_prep.py

Improved, production-style preprocessing pipeline that:
- Loads all CSVs found in data_dir
- Normalizes column names and types
- Handles missing values with flags and safe imputations
- Builds master_outlet_v2.csv and master_product_v2.csv
- Writes DATA_PROFILE.md and PREPROCESSING_LOG.md to document steps
/Users/vicky/Desktop
Usage:
    cd DataPreprocessing
    python3 data_prep.py --data-dir "/path/to/ajay'sDataset" --out-dir "./output"
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import textwrap
from datetime import datetime, timezone

# --------------------------
# Helpers
# --------------------------
def log(msg):
    ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
    print(f"[{ts}] {msg}")

def list_csvs(data_dir):
    p = Path(data_dir)
    files = {f.stem.lower(): f for f in p.iterdir() if f.suffix.lower() == ".csv"}
    return files

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        log(f"Failed to read {path}: {e}")
        return pd.DataFrame()

def norm_cols(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"\W", "_", regex=True)
    )
    return df

def safe_minmax(s: pd.Series) -> pd.Series:
    s = s.fillna(0)
    if s.empty:
        return s
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or (mx == mn):
        return pd.Series(0, index=s.index)
    return (s - mn) / (mx - mn)

def mk_flag(df, col):
    """Create availability boolean flag for a column based on not-null AND not-equal to zero for counts/amounts."""
    flag = f"{col}_available"
    if col not in df.columns:
        df[flag] = False
    else:
        # consider available if non-null and not all zeros for numeric-like fields
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            df[flag] = ~ser.isna() & (ser != 0)
        else:
            df[flag] = ~ser.isna() & (ser.astype(str).str.strip() != "")
    return df

def profile_df(df, name, max_examples=5):
    """Return profile dict for DATA_PROFILE.md"""
    p = {}
    p['rows'] = int(len(df))
    p['cols'] = int(len(df.columns))
    p['columns'] = []
    for c in df.columns:
        ser = df[c]
        n_missing = int(ser.isna().sum())
        n_zeros = 0
        if pd.api.types.is_numeric_dtype(ser):
            n_zeros = int((ser == 0).sum())
        p['columns'].append({
            'name': c,
            'dtype': str(ser.dtype),
            'n_missing': n_missing,
            'pct_missing': float(n_missing / max(1, len(ser))),
            'n_zeros': n_zeros,
            'pct_zeros': float(n_zeros / max(1, len(ser))),
            'examples': [str(x) for x in ser.dropna().unique()[:max_examples].tolist()]
        })
    return p

# --------------------------
# Main pipeline functions
# --------------------------
def load_and_normalize(data_dir):
    files = list_csvs(data_dir)
    log(f"Found CSV files: {list(files.keys())}")
    raw = {}
    for name, path in files.items():
        df = safe_read_csv(path)
        df = norm_cols(df)
        raw[name] = df
        log(f"Loaded '{name}' ({len(df)} rows, {len(df.columns)} cols)")
    return raw

def canonicalize_tables(raw):
    # safe get with copies
    menu = raw.get("menu", pd.DataFrame()).copy()
    outlet = raw.get("outlet", pd.DataFrame()).copy()
    outlet_menu = raw.get("outlet_menu", pd.DataFrame()).copy()
    daily_sales = raw.get("daily_sales", pd.DataFrame()).copy()
    order = raw.get("order", pd.DataFrame()).copy()
    review = raw.get("review", pd.DataFrame()).copy()
    food_quality = raw.get("food_quality", pd.DataFrame()).copy()
    transportation = raw.get("transportation", pd.DataFrame()).copy()
    customer = raw.get("customer_data", pd.DataFrame()).copy()
    promotion = raw.get("promotion", pd.DataFrame()).copy()
    employment = raw.get("employment", pd.DataFrame()).copy()
    emp_position = raw.get("emp_positon", pd.DataFrame()).copy()
    purchase_15 = raw.get("purchaseanalysis_15days", pd.DataFrame()).copy()
    purchase_month = raw.get("purchaseanalysis_month", pd.DataFrame()).copy()
    purchase_week = raw.get("purchaseanalysis", pd.DataFrame()).copy()
    purchase_all = raw.get("purchaseanalysis", pd.DataFrame()).copy() 
    return {
        'menu': menu, 'outlet': outlet, 'outlet_menu': outlet_menu, 'daily_sales': daily_sales,
        'order': order, 'review': review, 'food_quality': food_quality, 'transportation': transportation,
        'customer': customer, 'promotion': promotion, 'employment': employment, 'emp_position': emp_position,
        'purchase_15': purchase_15, 'purchase_month': purchase_month, 'purchase_week': purchase_week
    }

def standard_renames(tables):
    # menu product id
    menu = tables['menu']
    if 'productid' in menu.columns and 'product_id' not in menu.columns:
        menu = menu.rename(columns={'productid':'product_id', 'product':'product_name', 'price':'price'})
    if 'product' in menu.columns and 'product_name' not in menu.columns:
        menu = menu.rename(columns={'product':'product_name'})
    tables['menu'] = menu

    # outlet_menu
    om = tables['outlet_menu']
    if 'id' in om.columns and 'outlet_menu_id' not in om.columns:
        om = om.rename(columns={'id':'outlet_menu_id'})
    if 'productid' in om.columns and 'product_id' not in om.columns:
        om = om.rename(columns={'productid':'product_id'})
    if 'outletid' in om.columns and 'outlet_id' not in om.columns:
        om = om.rename(columns={'outletid':'outlet_id'})
    tables['outlet_menu'] = om

    # order
    o = tables['order']
    if 'outlet_product_id_' in o.columns and 'outlet_menu_id' not in o.columns:
        o = o.rename(columns={'outlet_product_id_':'outlet_menu_id'})
    if 'customer_id_' in o.columns and 'customer_id' not in o.columns:
        o = o.rename(columns={'customer_id_':'customer_id'})
    if 'quantity_' in o.columns and 'quantity' not in o.columns:
        o = o.rename(columns={'quantity_':'quantity'})
    if 'unitprice_' in o.columns and 'unit_price' not in o.columns:
        o = o.rename(columns={'unitprice_':'unit_price'})
    if 'date_' in o.columns and 'date' not in o.columns:
        o = o.rename(columns={'date_':'date'})
    tables['order'] = o

    # review
    r = tables['review']
    if 'outlet_product_id_' in r.columns and 'outlet_menu_id' not in r.columns:
        r = r.rename(columns={'outlet_product_id_':'outlet_menu_id'})
    if 'review_id' not in r.columns and 'reviewid' in r.columns:
        r = r.rename(columns={'reviewid':'review_id'})
    if 'positive' in r.columns and 'positive_score' not in r.columns:
        r = r.rename(columns={'positive':'positive_score', 'negative':'negative_score', 'neutral':'neutral_score'})
    tables['review'] = r

    # food_quality already OK (stage1..)
    # transportation used price -> transportation_cost
    tr = tables['transportation']
    if 'price' in tr.columns and 'transportation_cost' not in tr.columns:
        tr = tr.rename(columns={'price':'transportation_cost'})
    tables['transportation'] = tr

    # employment position link
    emp = tables['employment']
    if 'position_id' not in emp.columns:
        if 'position' in emp.columns:
            emp = emp.rename(columns={'position':'position_id'})
    tables['employment'] = emp

    return tables

def coerce_ids(tables):
    # cast common join keys to str for safe merging
    for name in ['outlet_menu', 'daily_sales', 'order', 'review', 'food_quality', 'transportation']:
        df = tables.get(name)
        if df is None or df.empty:
            continue
        if 'outlet_menu_id' in df.columns:
            df['outlet_menu_id'] = df['outlet_menu_id'].astype(str)
            tables[name] = df
    for name in ['menu', 'outlet', 'outlet_menu']:
        df = tables.get(name)
        if df is None or df.empty:
            continue
        if 'product_id' in df.columns:
            df['product_id'] = df['product_id'].astype(str)
            tables[name] = df
        if 'outlet_id' in df.columns:
            df['outlet_id'] = df['outlet_id'].astype(str)
            tables[name] = df
    return tables

def derive_master_outlet(tables):
    log("Deriving master_outlet_v2 ...")
    outlet = tables['outlet']
    outlet_menu = tables['outlet_menu']
    order = tables['order']
    daily_sales = tables['daily_sales']
    review = tables['review']
    food_quality = tables['food_quality']
    transportation = tables['transportation']
    promotion = tables['promotion']
    employment = tables['employment']

    # Start with outlet metadata
    master_out = pd.DataFrame()
    if 'outlet_id' in outlet.columns:
        master_out = outlet.copy()
        master_out['outlet_id'] = master_out['outlet_id'].astype(str)
    else:
        # if no outlet file, create from outlet_menu
        if 'outlet_menu_id' in outlet_menu.columns and 'outlet_id' in outlet_menu.columns:
            master_out = outlet_menu[['outlet_id']].drop_duplicates().rename(columns={'outlet_id':'outlet_id'})
            master_out['outlet_id'] = master_out['outlet_id'].astype(str)
        else:
            master_out = pd.DataFrame({'outlet_id':[]})

    # Prepare helper maps
    # Map outlet_menu -> outlet_id, product_id
    if not outlet_menu.empty:
        om_map = outlet_menu[['outlet_menu_id','outlet_id','product_id']].copy()
        om_map['outlet_menu_id'] = om_map['outlet_menu_id'].astype(str)
    else:
        om_map = pd.DataFrame(columns=['outlet_menu_id','outlet_id','product_id'])

    # ===== Sales aggregations =====
    # Order-based revenue & quantity
    if not order.empty and 'outlet_menu_id' in order.columns:
        order['outlet_menu_id'] = order['outlet_menu_id'].astype(str)
        order['quantity'] = pd.to_numeric(order.get('quantity', pd.Series(0)), errors='coerce').fillna(0)
        order['unit_price'] = pd.to_numeric(order.get('unit_price', pd.Series(0)), errors='coerce').fillna(0)
        order['revenue'] = order['quantity'] * order['unit_price']
        # merge order -> outlet_id using outlet_menu
        order = order.merge(om_map[['outlet_menu_id','outlet_id']], on='outlet_menu_id', how='left')
        order['outlet_id'] = order['outlet_id'].astype(str)
        o_agg = order.groupby('outlet_id', as_index=False).agg({
            'quantity':'sum',
            'revenue':'sum',
        }).rename(columns={'quantity':'total_units_sold','revenue':'total_revenue'})
    else:
        o_agg = pd.DataFrame(columns=['outlet_id','total_units_sold','total_revenue'])

    # daily_sales aggregated (units)
    if not daily_sales.empty and 'outlet_product_id' in daily_sales.columns:
        ds = daily_sales.rename(columns={'outlet_product_id':'outlet_menu_id'})
        ds['outlet_menu_id'] = ds['outlet_menu_id'].astype(str)
        if 'sales' in ds.columns:
            ds['sales'] = pd.to_numeric(ds['sales'], errors='coerce').fillna(0)
            ds = ds.merge(om_map[['outlet_menu_id','outlet_id']], on='outlet_menu_id', how='left')
            ds['outlet_id'] = ds['outlet_id'].astype(str)
            ds_agg = ds.groupby('outlet_id', as_index=False).agg({'sales':'sum'}).rename(columns={'sales':'total_sales'})
        else:
            ds_agg = pd.DataFrame(columns=['outlet_id','total_sales'])
    else:
        ds_agg = pd.DataFrame(columns=['outlet_id','total_sales'])

    # Merge sales measures into master_out
    for df in [o_agg, ds_agg]:
        if not df.empty and 'outlet_id' in df.columns:
            master_out = master_out.merge(df, on='outlet_id', how='left')

    # FIX: Apply fillna to the Series, not to the default value
    if 'total_units_sold' in master_out.columns:
        master_out['total_units_sold'] = master_out['total_units_sold'].fillna(0)
    else:
        master_out['total_units_sold'] = 0
    
    if 'total_revenue' in master_out.columns:
        master_out['total_revenue'] = master_out['total_revenue'].fillna(0)
    else:
        master_out['total_revenue'] = 0
    
    if 'total_sales' in master_out.columns:
        master_out['total_sales'] = master_out['total_sales'].fillna(0)
    else:
        master_out['total_sales'] = 0

    # ===== Customer metrics =====
    if not order.empty and 'customer_id' in order.columns:
        cust_count = order.groupby('outlet_id')['customer_id'].nunique().reset_index().rename(columns={'customer_id':'unique_customers'})
        master_out = master_out.merge(cust_count, on='outlet_id', how='left')
    
    if 'unique_customers' in master_out.columns:
        master_out['unique_customers'] = master_out['unique_customers'].fillna(0)
    else:
        master_out['unique_customers'] = 0

    master_out['avg_spend_per_customer'] = master_out.apply(
        lambda r: (r['total_revenue'] / r['unique_customers']) if r['unique_customers']>0 else 0, axis=1)

    # ===== Employee metrics =====
    if not employment.empty and 'outlet_id' in employment.columns:
        emp = employment.copy()
        if 'performance_rating' in emp.columns:
            emp['performance_rating'] = pd.to_numeric(emp['performance_rating'], errors='coerce')
            emp['outlet_id'] = emp['outlet_id'].astype(str)
            emp_agg = emp.groupby('outlet_id', as_index=False).agg({'performance_rating':'mean'})
            emp_agg = emp_agg.rename(columns={'performance_rating':'avg_employee_rating'})
            master_out = master_out.merge(emp_agg, on='outlet_id', how='left')
        # top role
        if 'position_id' in emp.columns:
            top_role = emp.groupby(['outlet_id','position_id']).size().reset_index(name='n').sort_values(['outlet_id','n'], ascending=[True,False])
            top_role = top_role.groupby('outlet_id').first().reset_index()[['outlet_id','position_id']].rename(columns={'position_id':'top_employee_role'})
            master_out = master_out.merge(top_role, on='outlet_id', how='left')
    
    if 'avg_employee_rating' in master_out.columns:
        master_out['avg_employee_rating'] = master_out['avg_employee_rating'].fillna(0)
    else:
        master_out['avg_employee_rating'] = 0

    # ===== Promotion & marketing =====
    if not promotion.empty and 'outlet_id' in promotion.columns:
        if 'cost' in promotion.columns:
            promotion['cost'] = pd.to_numeric(promotion['cost'], errors='coerce').fillna(0)
        else:
            promotion['cost'] = 0
        promotion['outlet_id'] = promotion['outlet_id'].astype(str)
        prom_agg = promotion.groupby('outlet_id', as_index=False).agg({
            'cost':'sum', 'sources':lambda s: s.mode().iloc[0] if not s.mode().empty else None, 'products':'sum'
        }).rename(columns={'cost':'total_promotion_cost','sources':'main_promotion_method','products':'num_promoted_products'})
        master_out = master_out.merge(prom_agg, on='outlet_id', how='left')
    
    if 'total_promotion_cost' in master_out.columns:
        master_out['total_promotion_cost'] = master_out['total_promotion_cost'].fillna(0)
    else:
        master_out['total_promotion_cost'] = 0
    
    if 'num_promoted_products' in master_out.columns:
        master_out['num_promoted_products'] = master_out['num_promoted_products'].fillna(0)
    else:
        master_out['num_promoted_products'] = 0

    # ===== Transportation costs =====
    if not transportation.empty and 'outlet_menu_id' in transportation.columns:
        tr = transportation.copy()
        if 'transportation_cost' in tr.columns:
            tr['transportation_cost'] = pd.to_numeric(tr['transportation_cost'], errors='coerce').fillna(0)
        else:
            tr['transportation_cost'] = 0
        tr = tr.merge(om_map[['outlet_menu_id','outlet_id']], on='outlet_menu_id', how='left')
        tr['outlet_id'] = tr['outlet_id'].astype(str)
        tr_agg = tr.groupby('outlet_id', as_index=False).agg({
            'transportation_cost':['mean','min','max']
        })
        tr_agg.columns = ['outlet_id','avg_transportation_cost','min_transportation_cost','max_transportation_cost']
        master_out = master_out.merge(tr_agg, on='outlet_id', how='left')
    
    for c in ['avg_transportation_cost','min_transportation_cost','max_transportation_cost']:
        if c in master_out.columns:
            master_out[c] = master_out[c].fillna(0)
        else:
            master_out[c] = 0

    # ===== Review & sentiment =====
    if not review.empty and 'outlet_menu_id' in review.columns:
        rv = review.copy()
        
        if 'positive_score' in rv.columns:
            rv['positive_score'] = pd.to_numeric(rv['positive_score'], errors='coerce').fillna(0)
        else:
            rv['positive_score'] = 0
            
        if 'negative_score' in rv.columns:
            rv['negative_score'] = pd.to_numeric(rv['negative_score'], errors='coerce').fillna(0)
        else:
            rv['negative_score'] = 0
            
        if 'neutral_score' in rv.columns:
            rv['neutral_score'] = pd.to_numeric(rv['neutral_score'], errors='coerce').fillna(0)
        else:
            rv['neutral_score'] = 0
            
        rv['sentiment_signal'] = rv['positive_score'] - rv['negative_score']
        rv = rv.merge(om_map[['outlet_menu_id','outlet_id']], on='outlet_menu_id', how='left')
        rv['outlet_id'] = rv['outlet_id'].astype(str)
        rv_agg = rv.groupby('outlet_id', as_index=False).agg({
            'positive_score':'mean','negative_score':'mean','neutral_score':'mean','sentiment_signal':'mean'
        }).rename(columns={'positive_score':'avg_positive_score','negative_score':'avg_negative_score','neutral_score':'avg_neutral_score','sentiment_signal':'avg_sentiment_signal'})
        rv_count = rv.groupby('outlet_id', as_index=False).size().reset_index().rename(columns={0:'total_reviews'})
        master_out = master_out.merge(rv_agg, on='outlet_id', how='left')
        master_out = master_out.merge(rv_count, on='outlet_id', how='left')
    
    for c in ['avg_positive_score','avg_negative_score','avg_neutral_score','avg_sentiment_signal','total_reviews']:
        if c in master_out.columns:
            master_out[c] = master_out[c].fillna(0)
        else:
            master_out[c] = 0

    # ===== Food quality =====
    if not food_quality.empty and 'outlet_menu_id' in food_quality.columns:
        fq = food_quality.copy()
        if 'total_stage' in fq.columns:
            fq['total_stage'] = pd.to_numeric(fq['total_stage'], errors='coerce').fillna(0)
        else:
            fq['total_stage'] = 0
        fq = fq.merge(om_map[['outlet_menu_id','outlet_id']], on='outlet_menu_id', how='left')
        fq_agg = fq.groupby('outlet_id', as_index=False).agg({
            'total_stage':'mean'
        }).rename(columns={'total_stage':'avg_quality_score'})
        # mode quality_status
        if 'quality_status' in fq.columns:
            qs = fq.groupby('outlet_id')['quality_status'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index().rename(columns={'quality_status':'quality_status_most_common'})
            fq_agg = fq_agg.merge(qs, on='outlet_id', how='left')
        master_out = master_out.merge(fq_agg, on='outlet_id', how='left')
    
    if 'avg_quality_score' in master_out.columns:
        master_out['avg_quality_score'] = master_out['avg_quality_score'].fillna(0)
    else:
        master_out['avg_quality_score'] = 0
    
    if 'quality_status_most_common' in master_out.columns:
        master_out['quality_status_most_common'] = master_out['quality_status_most_common'].fillna('unknown')
    else:
        master_out['quality_status_most_common'] = 'unknown'

    # ===== Derived metrics & availability flags =====
    for col in ['total_sales','total_units_sold','total_revenue','unique_customers','avg_employee_rating',
                'total_promotion_cost','avg_transportation_cost','total_reviews','avg_quality_score']:
        master_out = mk_flag(master_out, col)

    # If some metadata columns exist in outlet (employee_count), retain them
    if 'employee_count' in master_out.columns:
        master_out['employee_count'] = master_out['employee_count'].fillna(0)
    else:
        master_out['employee_count'] = 0

    # ===== Final outlet_score (explainable components) =====
    # produce normalized components
    master_out['norm_revenue'] = safe_minmax(master_out['total_revenue'])
    master_out['norm_quality'] = safe_minmax(master_out['avg_quality_score'])
    master_out['norm_employee'] = safe_minmax(master_out['avg_employee_rating'])
    master_out['norm_sentiment'] = safe_minmax(master_out['avg_sentiment_signal'])
    master_out['norm_customers'] = safe_minmax(master_out['unique_customers'])

    master_out['outlet_score'] = (
        0.30*master_out['norm_revenue'] +
        0.20*master_out['norm_quality'] +
        0.20*master_out['norm_employee'] +
        0.15*master_out['norm_sentiment'] +
        0.15*master_out['norm_customers']
    )

    # keep columns in sensible order
    keep_cols = ['outlet_id','outlet','city','phone_no','address','location','google_link','employee_count',
                 'total_sales','total_units_sold','total_revenue','avg_monthly_sales','unique_customers','avg_spend_per_customer',
                 'avg_employee_rating','top_employee_role','total_promotion_cost','num_promoted_products','main_promotion_method',
                 'avg_transportation_cost','min_transportation_cost','max_transportation_cost',
                 'avg_positive_score','avg_negative_score','avg_neutral_score','avg_sentiment_signal','total_reviews',
                 'avg_quality_score','quality_status_most_common','outlet_score']
    # intersect to avoid missing cols
    keep_cols = [c for c in keep_cols if c in master_out.columns]
    master_out = master_out[keep_cols]

    log("Derived master_outlet_v2 with %d rows and %d cols" % (len(master_out), len(master_out.columns)))
    return master_out

def derive_master_product(tables):
    log("Deriving master_product_v2 ...")
    menu = tables['menu']
    outlet_menu = tables['outlet_menu']
    order = tables['order']
    review = tables['review']
    food_quality = tables['food_quality']
    transportation = tables['transportation']
    purchase_month = tables['purchase_month']
    purchase_15 = tables['purchase_15']
    purchase_week = tables['purchase_week']

    # base: product metadata
    if not menu.empty and 'product_id' in menu.columns:
        products = menu.copy()
        products['product_id'] = products['product_id'].astype(str)
    else:
        products = pd.DataFrame(columns=['product_id'])

    # mapping outlet_menu -> product
    if not outlet_menu.empty:
        om_map = outlet_menu[['outlet_menu_id','product_id','outlet_id']].copy()
        om_map['outlet_menu_id'] = om_map['outlet_menu_id'].astype(str)
        om_map['product_id'] = om_map['product_id'].astype(str)
    else:
        om_map = pd.DataFrame(columns=['outlet_menu_id','product_id','outlet_id'])

    # ===== Sales & revenue per product (from order) =====
    if not order.empty and 'outlet_menu_id' in order.columns:
        ord_df = order.copy()
        ord_df['outlet_menu_id'] = ord_df['outlet_menu_id'].astype(str)
        ord_df = ord_df.merge(om_map[['outlet_menu_id','product_id']], on='outlet_menu_id', how='left')
        ord_df['product_id'] = ord_df['product_id'].astype(str)
        
        if 'quantity' in ord_df.columns:
            ord_df['quantity'] = pd.to_numeric(ord_df['quantity'], errors='coerce').fillna(0)
        else:
            ord_df['quantity'] = 0
            
        if 'unit_price' in ord_df.columns:
            ord_df['unit_price'] = pd.to_numeric(ord_df['unit_price'], errors='coerce').fillna(0)
        else:
            ord_df['unit_price'] = 0
            
        ord_df['revenue'] = ord_df['quantity'] * ord_df['unit_price']
        prod_sales = ord_df.groupby('product_id', as_index=False).agg({'quantity':'sum','revenue':'sum'}).rename(columns={'quantity':'units_sold_total','revenue':'revenue_total'})
    else:
        prod_sales = pd.DataFrame(columns=['product_id','units_sold_total','revenue_total'])

    products = products.merge(prod_sales, on='product_id', how='left')
    
    if 'units_sold_total' in products.columns:
        products['units_sold_total'] = products['units_sold_total'].fillna(0)
    else:
        products['units_sold_total'] = 0
    
    if 'revenue_total' in products.columns:
        products['revenue_total'] = products['revenue_total'].fillna(0)
    else:
        products['revenue_total'] = 0

    # ===== Number of outlets selling product =====
    if not om_map.empty:
        sold_outlets = om_map.groupby('product_id')['outlet_id'].nunique().reset_index().rename(columns={'outlet_id':'num_outlets_selling'})
        products = products.merge(sold_outlets, on='product_id', how='left')
    
    if 'num_outlets_selling' in products.columns:
        products['num_outlets_selling'] = products['num_outlets_selling'].fillna(0)
    else:
        products['num_outlets_selling'] = 0

    # ===== Profitability =====
    if 'price' in products.columns:
        products['price'] = pd.to_numeric(products['price'], errors='coerce').fillna(0)
    else:
        products['price'] = 0
        
    if 'product_cost' in products.columns:
        products['product_cost'] = pd.to_numeric(products['product_cost'], errors='coerce').fillna(0)
    else:
        products['product_cost'] = 0
        
    products['profit_per_unit'] = products['price'] - products['product_cost']
    products['total_profit'] = products['profit_per_unit'] * products['units_sold_total']
    products['margin_percentage'] = products.apply(lambda r: r['profit_per_unit']/r['price'] if r['price']>0 else 0, axis=1)

    # ===== Reviews & sentiment (product-level via outlet_menu) =====
    if not review.empty and 'outlet_menu_id' in review.columns:
        rv = review.copy()
        # Ensure we get a Series, not a scalar, before fillna
        if 'positive_score' in rv.columns:
            rv['positive_score'] = pd.to_numeric(rv['positive_score'], errors='coerce').fillna(0)
        else:
            rv['positive_score'] = 0
            
        if 'negative_score' in rv.columns:
            rv['negative_score'] = pd.to_numeric(rv['negative_score'], errors='coerce').fillna(0)
        else:
            rv['negative_score'] = 0
            
        if 'neutral_score' in rv.columns:
            rv['neutral_score'] = pd.to_numeric(rv['neutral_score'], errors='coerce').fillna(0)
        else:
            rv['neutral_score'] = 0
            
        rv['sentiment_signal'] = rv['positive_score'] - rv['negative_score']
        rv = rv.merge(om_map[['outlet_menu_id','product_id']], on='outlet_menu_id', how='left')
        rv['product_id'] = rv['product_id'].astype(str)
        rv_agg = rv.groupby('product_id', as_index=False).agg({
            'positive_score':'mean','negative_score':'mean','neutral_score':'mean','sentiment_signal':'mean'
        }).rename(columns={'positive_score':'avg_positive_score','negative_score':'avg_negative_score','neutral_score':'avg_neutral_score','sentiment_signal':'avg_sentiment_signal'})
        rv_count = rv.groupby('product_id', as_index=False).size().reset_index().rename(columns={0:'total_reviews'})
        products = products.merge(rv_agg, on='product_id', how='left')
        products = products.merge(rv_count, on='product_id', how='left')
    
    for c in ['avg_positive_score','avg_negative_score','avg_neutral_score','avg_sentiment_signal','total_reviews']:
        if c in products.columns:
            products[c] = products[c].fillna(0)
        else:
            products[c] = 0

    # ===== Food quality per product (via outlet_menu -> multiple outlets) =====
    if not food_quality.empty and 'outlet_menu_id' in food_quality.columns:
        fq = food_quality.copy()
        fq = fq.merge(om_map[['outlet_menu_id','product_id']], on='outlet_menu_id', how='left')
        fq['product_id'] = fq['product_id'].astype(str)
        if 'total_stage' in fq.columns:
            fq['total_stage'] = pd.to_numeric(fq['total_stage'], errors='coerce').fillna(0)
        else:
            fq['total_stage'] = 0
        fq_agg = fq.groupby('product_id', as_index=False).agg({'total_stage':'mean'}).rename(columns={'total_stage':'avg_quality_score'})
        products = products.merge(fq_agg, on='product_id', how='left')
    
    if 'avg_quality_score' in products.columns:
        products['avg_quality_score'] = products['avg_quality_score'].fillna(0)
    else:
        products['avg_quality_score'] = 0

    # ===== Transportation per product =====
    if not transportation.empty and 'outlet_menu_id' in transportation.columns:
        tr = transportation.copy()
        tr = tr.merge(om_map[['outlet_menu_id','product_id']], on='outlet_menu_id', how='left')
        tr['product_id'] = tr['product_id'].astype(str)
        if 'transportation_cost' in tr.columns:
            tr['transportation_cost'] = pd.to_numeric(tr['transportation_cost'], errors='coerce').fillna(0)
        else:
            tr['transportation_cost'] = 0
        tr_agg = tr.groupby('product_id', as_index=False).agg({'transportation_cost':['mean','min','max']})
        tr_agg.columns = ['product_id','avg_transportation_cost','min_transportation_cost','max_transportation_cost']
        products = products.merge(tr_agg, on='product_id', how='left')
    
    for c in ['avg_transportation_cost','min_transportation_cost','max_transportation_cost']:
        if c in products.columns:
            products[c] = products[c].fillna(0)
        else:
            products[c] = 0

    # ===== Customer-level product stats (repeat purchase etc) using purchaseanalysis files if available =====
    # Attempt to use purchase_month / purchase_week / purchase_15 to estimate repeat rates and unique customers
    try:
        # combine monthly and 15-day and weekly into a transaction summary if present
        pa_frames = []
        for key in ['purchase_month','purchase_15','purchase_week']:
            df = tables.get(key) if (tables and key in tables) else None
    except Exception:
        pass

    # quick estimate of num_unique_customers_bought and repeat rate from order if available
    if not order.empty and 'outlet_menu_id' in order.columns:
        ord2 = order.merge(om_map[['outlet_menu_id','product_id']], on='outlet_menu_id', how='left')
        cust_per_prod = ord2.groupby('product_id')['customer_id'].nunique().reset_index().rename(columns={'customer_id':'num_unique_customers_bought'})
        repeat = ord2.groupby('product_id').agg({'customer_id':'count'}).reset_index().rename(columns={'customer_id':'total_purchase_records'})
        rpt = repeat.merge(cust_per_prod, on='product_id', how='left')
        rpt['repeat_purchase_rate'] = rpt.apply(lambda r: (r['total_purchase_records']/r['num_unique_customers_bought']) if r['num_unique_customers_bought']>0 else 0, axis=1)
        products = products.merge(rpt[['product_id','num_unique_customers_bought','repeat_purchase_rate']], on='product_id', how='left')
    
    if 'num_unique_customers_bought' in products.columns:
        products['num_unique_customers_bought'] = products['num_unique_customers_bought'].fillna(0)
    else:
        products['num_unique_customers_bought'] = 0
    
    if 'repeat_purchase_rate' in products.columns:
        products['repeat_purchase_rate'] = products['repeat_purchase_rate'].fillna(0)
    else:
        products['repeat_purchase_rate'] = 0

    # ===== Derived columns exist flags =====
    for col in ['units_sold_total','revenue_total','num_outlets_selling','total_reviews','avg_quality_score','avg_sentiment_signal','total_profit']:
        products = mk_flag(products, col)

    # ===== product_score components and final score =====
    products['norm_units_sold'] = safe_minmax(products['units_sold_total'])
    products['norm_quality'] = safe_minmax(products['avg_quality_score'])
    
    # Fix for total_reviews - ensure it's a Series before passing to safe_minmax
    if 'total_reviews' in products.columns:
        products['norm_reviews'] = safe_minmax(products['total_reviews'])
    else:
        products['total_reviews'] = 0
        products['norm_reviews'] = 0
    
    # Fix for avg_sentiment_signal - ensure it's a Series before passing to safe_minmax
    if 'avg_sentiment_signal' in products.columns:
        products['norm_sentiment'] = safe_minmax(products['avg_sentiment_signal'])
    else:
        products['avg_sentiment_signal'] = 0
        products['norm_sentiment'] = 0
    
    # Fix for total_profit - ensure it exists as a column
    if 'total_profit' in products.columns:
        products['norm_profit'] = safe_minmax(products['total_profit'])
    else:
        products['total_profit'] = 0
        products['norm_profit'] = 0

    products['product_score'] = (
        0.35*products['norm_units_sold'] +
        0.20*products['norm_quality'] +
        0.15*products['norm_reviews'] +
        0.15*products['norm_sentiment'] +
        0.15*products['norm_profit']
    )

    # select sensible columns
    keep_cols = ['product_id','category','product_name','price','product_cost','profit_per_unit','total_profit','margin_percentage',
                 'units_sold_total','revenue_total','num_outlets_selling','num_unique_customers_bought','repeat_purchase_rate',
                 'avg_quality_score','avg_positive_score','avg_negative_score','avg_neutral_score','avg_sentiment_signal','total_reviews',
                 'avg_transportation_cost','product_score']
    keep_cols = [c for c in keep_cols if c in products.columns]
    products = products[keep_cols]

    log("Derived master_product_v2 with %d rows and %d cols" % (len(products), len(products.columns)))
    return products

# --------------------------
# Top-level runner
# --------------------------
def run(data_dir, out_dir):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else data_dir / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_and_normalize(data_dir)
    tables = canonicalize_tables(raw)
    tables = standard_renames(tables)
    tables = coerce_ids(tables)

    # Create data profiles (before processing) for documentation
    profiles = {name: profile_df(df, name) for name, df in tables.items()}

    # Derive masters
    master_out = derive_master_outlet(tables)
    master_prod = derive_master_product(tables)

    # Save outputs
    mo = out_dir / "master_outlet_v2.csv"
    mp = out_dir / "master_product_v2.csv"
    master_out.to_csv(mo, index=False)
    master_prod.to_csv(mp, index=False)
    log(f"Saved master_outlet_v2 -> {mo}")
    log(f"Saved master_product_v2 -> {mp}")

    # DATA_PROFILE.md
    dp = out_dir / "DATA_PROFILE.md"
    with dp.open('w', encoding='utf8') as f:
        f.write("# DATA PROFILE\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        for name, df in tables.items():
            prof = profile_df(df, name)
            f.write(f"## {name} (rows={prof['rows']}, cols={prof['cols']})\n\n")
            f.write("| column | dtype | n_missing | pct_missing | n_zeros | pct_zeros | examples |\n")
            f.write("|---|---:|---:|---:|---:|---:|---|\n")
            for col in prof['columns']:
                examples = ", ".join(col['examples'][:3])
                f.write(f"| {col['name']} | {col['dtype']} | {col['n_missing']} | {col['pct_missing']:.2f} | {col['n_zeros']} | {col['pct_zeros']:.2f} | {examples} |\n")
            f.write("\n\n")
    log(f"Wrote data profile to {dp}")

    # PREPROCESSING_LOG.md
    pl = out_dir / "PREPROCESSING_LOG.md"
    with pl.open('w', encoding='utf8') as f:
        f.write("# PREPROCESSING LOG\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("Actions summary:\n\n")
        f.write("- Normalized column names to snake_case for all CSVs.\n")
        f.write("- Renamed productid -> product_id, outlet_menu id columns normalized, order/review fields standardized.\n")
        f.write("- Cast join keys to string for safe merges (outlet_menu_id, product_id, outlet_id).\n")
        f.write("- Created availability flags for key derived columns (e.g., total_sales_available, review_count_available).\n")
        f.write("- Computed per-outlet and per-product aggregates, sentiment averages, transportation aggregates, promotion aggregates, and employee metrics.\n")
        f.write("- Derived explainable scores (outlet_score, product_score) and stored normalized components for traceability.\n")
        f.write("\nRefer to master_outlet_v2.csv and master_product_v2.csv for final fields.\n")
    log(f"Wrote preprocessing log to {pl}")

    log("Pipeline completed successfully.")

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder containing CSVs")
    parser.add_argument("--out-dir", default=None, help="Output folder (default: <data-dir>/output)")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)

if __name__ == "__main__":
    main()