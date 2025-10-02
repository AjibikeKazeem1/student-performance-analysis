### Import the necessary python libraries  
import pandas as pd
import re
import difflib
import os
import numpy as np

## -------------------------------------
#### STEP 1: Load the Data 
## --------------------------------------
raw = pd.read_csv("C:/Users/HP/Desktop/student-performance-analysis/data/raw/StudentsPerformance.csv")
df = raw.copy() ## use a copy of the data for cleaning work-flow
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nPreview:")
print(df.head(),"\n")

##----------------------------------------------
### STEP 2: Clean Column Names of the Dataset
## ---------------------------------------------

# ---------- utility functions ----------
def normalize(s: str) -> str:
    """Normalized string used for fuzzy matching (letters+digits only, lowercase)."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def sanitize_for_column(s: str) -> str:
    """Produce a safe snake_case column name from display string."""
    t = re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')
    return t

# ---------- canonical expected columns (display form) ----------
canonical_display = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course',
    'math score',
    'reading score',
    'writing score'
]

# Precompute normalized forms
canon_norm = {normalize(c): c for c in canonical_display}
canon_norm_keys = list(canon_norm.keys())

# ---------- load your dataset ----------
inpath = "C:/Users/HP/Desktop/student-performance-analysis/data/raw/StudentsPerformance.csv"   
if not os.path.exists(inpath):
    raise FileNotFoundError(f"File not found: {inpath}")

df = pd.read_csv(inpath)
orig_cols = df.columns.tolist()
print("Original columns:")
print(orig_cols, "\n")

# ---------- build rename map ----------
rename_map = {}
for col in orig_cols:
    norm = normalize(col)
    # try to match to a canonical normalized key
    match = difflib.get_close_matches(norm, canon_norm_keys, n=1, cutoff=0.5)
    if match:
        display_name = canon_norm[match[0]]            # e.g. 'race/ethnicity'
else:
        # fallback: replace sequences of underscores/spaces with single space, strip
        display_name = re.sub(r'[_\s]+', ' ', str(col)).strip().lower()
    # final sanitized column name in snake_case
    final_col = sanitize_for_column(display_name)
    rename_map[col] = final_col

# ---------- show mapping for inspection ----------
print("Proposed renaming:")
for old, new in rename_map.items():
    print(f"'{old}' -> '{new}'")
print()

# ---------- apply rename ----------
df = df.rename(columns=rename_map)

# ---------- optional: reorder or ensure canonical order ----------
# If you want canonical order and to ensure all canonical columns exist, do:
desired_order = [sanitize_for_column(c) for c in canonical_display]
# keep only columns that actually exist in df
desired_present = [c for c in desired_order if c in df.columns]
# then add any other columns that were present but not in desired list
other_cols = [c for c in df.columns if c not in desired_present]
new_order = desired_present + other_cols
df = df[new_order]

print("Cleaned columns:")
print(df.columns.tolist())
print("\nPreview:")
print(df.head())

# ---------- save cleaned file ----------
outpath = "StudentsPerformance_cleaned.csv"
df.to_csv(outpath, index=False)
print(f"\nSaved cleaned dataset to: {outpath}")

## Columns expected in the datasets(for reference purpose)
## num_cols means numerical columns, cat_cols means categorical columns  
num_cols = ["math_score", "reading_score", "writing_score"]
cat_cols = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]


# ------------------------------------
### STEP3: inspect the data
# ------------------------------------
print("Initial shape:",df.shape)
print(df.head(6)) ## print the first 6 rows of the data 
print("\nInfo:")
print(df.info())
print("\nValue counts (categoricals):")
for c in df.columns:
    if df[c].dtype == "object" or c in cat_cols:
        print("\n", c, "->", df[c].value_counts(dropna=False).to_dict())
# -------------------------------------
### STEP 4: Missing Value diagnostics 
# --------------------------------------
missing = df.isnull().sum().sort_values(ascending= False)
missing_pct = (missing/ len(df)*100).round(2)
print("\nMissing values (count & %):")
print(pd.concat([missing, missing_pct], axis =1, keys = ["count(n)", "percent(%)"]))

# -------------------------------------------
### STEP 5: General cleaning 
# -------------------------------------------
# Friendly human labels 
if "lunch" in df.columns:
    df["lunch"] = df["lunch"].replace({
        "standard":"Standard", 
        "free/reduced":"Free/Reduced", 
        "free_reduced":"Free/Reduced"  # this line is to guard against variants
    })
# Capitalize gender 
if "gender" in df.columns:
    df["gender"] = df["gender"].map({"female": "Female", "male":"Male"}).fillna(df["gender"])

# Normalize test_preparation_course 
if "test_preparation_course" in df.columns:
    df["test_preparation_course"] = df["test_preparation_course"].replace({
        "none":"None", 
        "completed":"Completed"
    })

# Standardized parental education categories to ordered categories 
expected_edu = [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
]
if "parental_level_of_education" in df.columns:
    #map titles to a consistent lower-case form then back to title form 
    df["parental_level_of_education"] = df["parental_level_of_education"].str.lower().str.strip()
    # unify common variants manually if observed
    edu_map = {v:v.title() for v in expected_edu}
    df["parental_level_of_education"] = df["parental_level_of_education"].replace(edu_map).fillna(df["parental_level_of_education"])

print("\nAfter standardization (unique values):")

for c in ["gender", "lunch", "test_preparation_course", "parental_level_of_education", "race_ethnicity"]:
    if c in df.columns:
        print(c, ":", df[c].dropna().unique())

## Convert numeric columns and handle non-numeric entries----------
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
print("\nNumeric summary before imputation:")
print(df[num_cols].describe())

## --------------------------------------
### STEP 6: Handle missing values 
## --------------------------------------
# Stategy 
# -if numeric NaNs: impute median 
# -if categorical NaNs: fill with mode

# Numerical imputation 
for c in num_cols:
    if c in df.columns:
        n_miss = df[c].isna().sum()
        if n_miss > 0:
            med = df[c].median()
            df[c + "_missing_flag"] = df[c].isna().astype(int)
            df[c].fillna(med, inplace=True)
            print(f"Imputed {n_miss} missing in {c} with median {med}")

# Categorical imputation 
for c in cat_cols:
    if c in df.columns:
        n_miss = df[c].isna().sum()
        if n_miss > 0:
            mode = df[c].mode(dropna = True)
            fill = mode [0] if len(mode) > o else "Unknown"
            df[c + "_missing_flag"] = df[c].isna().astype(int)
            df[c].fillna(fill, inplace = True)
            print(f"Imputed {n_miss} missing in {c} with mode '{fill}'")

## ----------------------------------------
### STEP 7: Remove duplicates 
## ----------------------------------------
dup_count = df.duplicated().sum()
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"\nDropped {dup_count} duplicate rows.")

else:
    print("\nNo duplicate rows found.")

## ----------------------------------------------
### STEP 8: Fix invalid scores/ outliers
## ----------------------------------------------
valid_mask = df[num_cols].apply(lambda s: s.between(0, 100)).all(axis=1)
invalid_rows = (~valid_mask).sum()
if invalid_rows > 0:
    print(f"\nFound {invalid_rows} rows with invalid scores (outside 0-100). Removing them.")
    df = df[valid_mask].copy()
else:
    print("\nNo invalid score values found (all 0-100).")

## ----------------------------------------
### STEP 9: Feature Engineering
## ----------------------------------------
df['total_score'] = df[num_cols].sum(axis=1)
df['avg_score'] = df['total_score'] / len(num_cols)
df['pass_math'] = (df['math_score'] >= 50).astype(int)        # adjust threshold as needed
df['pass_reading'] = (df['reading_score'] >= 50).astype(int)
df['pass_writing'] = (df['writing_score'] >= 50).astype(int)

## ----------------------------------------
### STEP 10: Encoding for modeling 
## ----------------------------------------
# simple encoding 
if "gender" in df.columns:
    df["gender_bin"] = df["gender"].map({"Female":0, "Male":1})

# One-hot encoding 
ohe_cols = [c for c in ["race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"] if c in df.columns]
df = pd.get_dummies(df, columns = ohe_cols, drop_first = False)

## -------------------------------------
### Final Checks 
## ----------------------------------
print("\nFinal shape:", df.shape)
print(df.info())
print("\nMissing after cleaning (should be zero or only flags):")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# ----Save cleaned dataset ----------
df.to_csv(outpath, index=False)
print("\nSaved cleaned data to:", outpath)

# ----------move file to folder outside python environment-------
os.replace("StudentsPerformance_cleaned.csv", 
           r"C:/Users/HP/Desktop/student-performance-analysis/data/processed/StudentsPerformance_cleaned.csv")
print("File moved successfully!")