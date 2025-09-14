# =========================================
# Setup: Libraries, Display, Helper Functions
# =========================================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

def qclip(s: pd.Series, q=0.99):
    """Clip upper tail for better axis readability."""
    s = pd.to_numeric(s, errors='coerce')
    return s.clip(upper=s.quantile(q))

# =========================================
# Load Data
# =========================================
PROJECT_ROOT = os.getcwd()
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RAW_LISTINGS_PATH = os.path.join(DATA_DIR, "listings.csv")
print("Using file:", RAW_LISTINGS_PATH)

df = pd.read_csv(RAW_LISTINGS_PATH, compression='infer', quotechar='"', encoding='utf-8', low_memory=False)
print(f"Dataset shape: {df.shape}")
print("\nHead:"); display(df.head(10))
print("\nTail:"); display(df.tail(10))

# =========================================
# Dtypes & Roles: Type Conversion & Quick Classification
# =========================================
if 'last_review' in df.columns:
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

for c in ['price','minimum_nights','availability_365',
          'number_of_reviews','reviews_per_month',
          'number_of_reviews_ltm','calculated_host_listings_count']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

def is_identifier(colname: str) -> bool:
    colname = colname.lower()
    return (colname == 'id' or colname.endswith('_id') or 'identity' in colname or 'תעודת' in colname)

all_numeric = df.select_dtypes(include='number').columns.tolist()
numeric_cols = [c for c in all_numeric if not is_identifier(c)]
datetime_cols = df.select_dtypes(include='datetime').columns.tolist()
non_num = df.columns.difference(all_numeric + datetime_cols)
categorical_cols = [c for c in non_num if df[c].nunique(dropna=True) <= 10]
text_cols = list(set(non_num) - set(categorical_cols))

print("Numeric:", numeric_cols)
print("Datetime:", datetime_cols)
print("Categorical (<=10 unique):", categorical_cols)
print("Text:", text_cols)

# =========================================
# Numeric Summary
# =========================================
if numeric_cols:
    num_summary = df[numeric_cols].describe().T.round(2)
    display(num_summary.style.format("{:.2f}"))
else:
    print("No numeric columns detected.")

# =========================================
# Categorical Values
# =========================================
if categorical_cols:
    for col in categorical_cols:
        vals = df[col].dropna().unique().tolist()
        print(f"\n=== {col} (nunique={len(vals)}) ===")
        display(pd.Series(vals, name='values'))
else:
    print("No categorical columns (<=10 unique) detected.")

# =========================================
# Rename Columns
# =========================================
rename_map = {
    'id': 'listing_id',
    'number_of_reviews_ltm': 'reviews_12m',
    'calculated_host_listings_count': 'host_listings',
    'minimum_nights': 'min_nights',
}
df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
print("Columns after rename:", list(df.columns))

# =========================================
# Missingness Report + Visuals
# =========================================
missing_count = df.isna().sum()
missing_pct   = (missing_count / len(df) * 100).round(1)
miss_tbl = (pd.DataFrame({'missing_count': missing_count, 'missing_%': missing_pct})
              .query('missing_count > 0')
              .sort_values('missing_count', ascending=False))

print("=== Missing values per column (only columns with missing) ===")
display(miss_tbl)

if not miss_tbl.empty:
    plt.figure(figsize=(10, max(4, 0.35 * len(miss_tbl))))
    sns.barplot(x='missing_count', y=miss_tbl.index, data=miss_tbl, palette="Reds_r")
    plt.title('Missing count by column')
    plt.xlabel('Missing count'); plt.ylabel('Column')
    plt.tight_layout(); plt.show()

cols_with_na = miss_tbl.index.tolist()
if cols_with_na:
    plt.figure(figsize=(12, max(6, 0.6 * len(cols_with_na))))
    sns.heatmap(df[cols_with_na].isna().T, cbar=False, cmap='Reds')
    plt.title('Missingness pattern (rows vs. columns with missing)')
    plt.ylabel('Columns'); plt.xlabel('Rows')
    plt.tight_layout(); plt.show()

# =========================================
# Missingness Cleaning & Column Filtering
# =========================================
if 'number_of_reviews' in df.columns:
    mask_zero_reviews = df['number_of_reviews'].fillna(0).eq(0)
    for c in ['reviews_12m','reviews_per_month']:
        if c in df.columns:
            df.loc[mask_zero_reviews, c] = 0.0

MISSING_DROP_RATIO = 0.99
to_drop = [c for c in df.columns
           if (df[c].isna().mean() >= MISSING_DROP_RATIO) or (df[c].nunique(dropna=True) <= 1)]
if to_drop:
    df = df.drop(columns=to_drop)
    print("Dropped non-informative columns:", to_drop)

if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    out_mask = df['price'] > 5000
    df.loc[out_mask, 'price'] = np.nan
    print("Set price>5000 to NaN:", int(out_mask.sum()))

print("\nPreview after cleaning:")
display(df.head(10))
display(df.tail(10))

# =========================================
# Feature Engineering
# =========================================
df_new = df.copy()
df_new['price_category'] = pd.cut(
    df_new['price'],
    bins=[-1, 100, 300, float('inf')],
    labels=['זול', 'בינוני', 'יקר']
).astype('category')

if 'host_listings' in df_new.columns:
    df_new['host_experience'] = pd.cut(
        df_new['host_listings'],
        bins=[-1, 2, 5, float('inf')],
        labels=['חדש', 'בינוני', 'מנוסה']
    ).astype('category')

def availability_level(x):
    if pd.isna(x):
        return 'לא ידוע'
    elif x < 120:
        return 'נמוכה'
    elif 120 <= x <= 250:
        return 'בינונית'
    else:
        return 'גבוהה'

df_new['availability_level'] = df_new['availability_365'].apply(availability_level).astype('category')

print("עמודות חדשות נוספו:")
display(df_new[['price', 'price_category',
                'host_listings', 'host_experience',
                'availability_365', 'availability_level']].head(15))

# =========================================
# Save Cleaned Data
# =========================================
out_path = os.path.join(OUTPUTS_DIR, 'listingsCleaned.csv')
df.to_csv(out_path, index=False)
print("Saved cleaned sample to:", out_path)

# =========================================
# (Optional) Add your EDA and plotting code below
# =========================================
# Add your EDA and visualization code below