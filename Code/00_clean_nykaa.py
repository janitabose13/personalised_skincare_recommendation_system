"""
00_clean_nykaa.py
=================
Cleans raw Nykaa data and merges EWG Skin Deep + INCIDecoder ingredient data
into a single enriched master CSV.

Run after:
  python nykaa_full_scraper.py
  python ewg_scraper.py
  python incidecoder_scraper.py

Outputs:
  data/processed/skincare_master.csv    products with EWG + INCIDecoder features
  data/processed/user_ratings.csv       real per-user review ratings
  data/processed/ingredient_flags.csv   per-ingredient safety reference table
"""

import pandas as pd
import numpy as np
import os, re

print("=" * 60)
print("DATA CLEANING + EWG/INCIDecoder MERGE")
print("=" * 60)

os.makedirs("data/processed", exist_ok=True)

# ── 1. Load raw Nykaa ─────────────────────────────────────────────────────────
prod_df   = pd.read_csv("data/raw/nykaa_products.csv")
review_df = pd.read_csv("data/raw/nykaa_reviews.csv")
print(f"\nRaw products : {len(prod_df)}")
print(f"Raw reviews  : {len(review_df)}")

# ── 2. Remove non-skincare ────────────────────────────────────────────────────
NON_SKIN_KW = [
    "hair","shampoo","conditioner","nail ","lip ","mascara","foundation",
    "blush","eyeliner","eyeshadow","lipstick","body wash","shower",
    "deodorant","perfume","cologne","tooth","dental","shave","razor","beard",
    "hair oil","hair mask","hair serum","hair color","hair dye","dry shampoo",
    "heat protectant","shimmer pencil","kajal","kohl","bronzer","concealer",
    "highlighter","setting powder","brow gel","false lash","foot cream",
    "foot scrub","hand cream","hand wash","nail enamel","nail paint",
    "nail polish","nail lacquer","nail care","body spray","body mist",
    "body butter","body lotion","body oil","baby soap","baby wash",
    "baby powder","baby lotion","diaper","mosquito","wax heater",
    "pedicure tool","cotton ball","cotton bud","makeup brush","brush set",
    "palette","curling tong","straightener","hair dryer","trimmer","epilator",
]
SKIN_CATS = ["Moisturizer","Serum","Cleanser","Sunscreen","Toner",
             "Eye Cream","Face Oil","Night Cream","Exfoliator","Face Mask"]

def is_skincare(row):
    name = str(row.get("product_name","")).lower()
    cat  = str(row.get("category",""))
    if cat not in SKIN_CATS: return False
    if any(kw in name for kw in NON_SKIN_KW): return False
    return True

before   = len(prod_df)
prod_df  = prod_df[prod_df.apply(is_skincare, axis=1)].copy()
prod_df["rating"] = pd.to_numeric(prod_df["rating"], errors="coerce")
prod_df  = prod_df[prod_df["rating"].between(1.0, 5.0)].copy()
prod_df  = prod_df[
    (prod_df["num_ratings"].fillna(0) >= 5) |
    (prod_df["num_reviews"].fillna(0) >= 2)
].copy()
prod_df  = prod_df.drop_duplicates(subset=["product_name","brand"]).reset_index(drop=True)
print(f"After cleaning : {len(prod_df)} products (removed {before - len(prod_df)})")

# Ensure base feature columns exist
for col, default in [
    ("skin_oily",0),("skin_dry",0),("skin_combination",0),
    ("skin_sensitive",0),("skin_normal",0),
    ("ingredient_count",0),("num_actives",0),("key_actives",""),
    ("irritant_count",0),("comedogen_count",0),
    ("has_fragrance",0),("is_fragrance_free",1),
]:
    if col not in prod_df.columns:
        prod_df[col] = default

if "price_tier" not in prod_df.columns:
    prod_df["price_tier"] = prod_df["price_inr"].apply(
        lambda p: "Budget" if pd.notna(p) and p<500 else "Mid-Range" if pd.notna(p) and p<=1500 else "Luxury" if pd.notna(p) else "Unknown"
    )

if "brand_tier" not in prod_df.columns:
    LUX  = {"forest essentials","kama ayurveda","guerlain","tatcha","la mer","clinique","estee lauder","shiseido","lancome","dior","chanel","sulwhasoo"}
    DRUG = {"cetaphil","cerave","neutrogena","nivea","himalaya","biotique","ponds","lakme","garnier","mamaearth","plum","minimalist","dot & key","the ordinary","wow","mcaffeine"}
    def btier(b):
        bl = str(b).lower()
        if any(x in bl for x in LUX):  return "Luxury"
        if any(x in bl for x in DRUG): return "Drugstore"
        return "Prestige"
    prod_df["brand_tier"] = prod_df["brand"].apply(btier)

# ── 3. Load EWG ingredient data ───────────────────────────────────────────────
ewg_file = "data/raw/ewg_ingredients.csv"
ewg_df   = None
if os.path.exists(ewg_file):
    ewg_df = pd.read_csv(ewg_file)
    ewg_df = ewg_df[ewg_df["ewg_hazard_label"] != "not_found"].copy()
    print(f"\nEWG ingredients : {len(ewg_df)}")
    print(f"  High    : {(ewg_df['ewg_hazard_label']=='high').sum()}")
    print(f"  Moderate: {(ewg_df['ewg_hazard_label']=='moderate').sum()}")
    print(f"  Low     : {(ewg_df['ewg_hazard_label']=='low').sum()}")
else:
    print("\nWARNING: data/raw/ewg_ingredients.csv not found — run ewg_scraper.py")

# ── 4. Load INCIDecoder ingredient data ───────────────────────────────────────
inci_file = "data/raw/incidecoder_ingredients.csv"
inci_df   = None
if os.path.exists(inci_file):
    inci_df = pd.read_csv(inci_file)
    print(f"\nINCIDecoder ingredients : {len(inci_df)}")
else:
    print("\nWARNING: data/raw/incidecoder_ingredients.csv not found — run incidecoder_scraper.py")

# ── 5. Build ingredient lookup helpers ───────────────────────────────────────
def build_lookup(df, name_cols):
    """Build {lowercase_name: row_index} lookup from multiple name columns."""
    lookup = {}
    if df is None: return lookup
    for col in name_cols:
        if col not in df.columns: continue
        for i, val in enumerate(df[col].fillna("").str.lower()):
            if val and val not in lookup:
                lookup[val] = i
    return lookup

ewg_lookup  = build_lookup(ewg_df,  ["query_name","inci_name"])
inci_lookup = build_lookup(inci_df, ["ingredient_name","inci_name"])

def match_ingredient(ing_name, lookup):
    """Fuzzy match ingredient name against lookup dict."""
    ing = ing_name.strip().lower()
    if ing in lookup: return lookup[ing]
    # Partial match: check if any lookup key is contained in the ingredient or vice versa
    for key, idx in lookup.items():
        if len(key) > 4 and (key in ing or ing in key):
            return idx
    return None

# ── 6. Per-product EWG feature computation ───────────────────────────────────
def ewg_features(ings_text):
    if ewg_df is None or not ings_text or str(ings_text) == "nan":
        return dict(ewg_max_hazard=np.nan, ewg_mean_hazard=np.nan,
                    ewg_high_hazard_cnt=0, ewg_cancer_flags=0,
                    ewg_allergy_flags=0, ewg_endocrine_flags=0,
                    ewg_developmental_flags=0, ewg_irritation_flags=0,
                    ewg_restricted_flags=0, ewg_matched=0)
    ings   = [x.strip() for x in str(ings_text).split(",") if x.strip()]
    scores = []; cancer=allergy=endocrine=devel=irrit=restric=high_cnt=0
    for ing in ings:
        idx = match_ingredient(ing, ewg_lookup)
        if idx is None: continue
        row = ewg_df.iloc[idx]
        try:
            s = float(row.get("ewg_hazard_score") or 0)
            scores.append(s)
            if s >= 7: high_cnt += 1
        except: pass
        cancer  += int(row.get("concern_cancer",0) or 0)
        allergy += int(row.get("concern_allergy",0) or 0)
        endocrine+=int(row.get("concern_endocrine",0) or 0)
        devel   += int(row.get("concern_developmental",0) or 0)
        irrit   += int(row.get("concern_irritation",0) or 0)
        restric += int(row.get("concern_restricted",0) or 0)
    return dict(
        ewg_max_hazard      = max(scores) if scores else np.nan,
        ewg_mean_hazard     = round(np.mean(scores),2) if scores else np.nan,
        ewg_high_hazard_cnt = high_cnt,
        ewg_cancer_flags    = cancer,
        ewg_allergy_flags   = allergy,
        ewg_endocrine_flags = endocrine,
        ewg_developmental_flags = devel,
        ewg_irritation_flags= irrit,
        ewg_restricted_flags= restric,
        ewg_matched         = len(scores),
    )

# ── 7. Per-product INCIDecoder feature computation ────────────────────────────
def inci_features(ings_text):
    if inci_df is None or not ings_text or str(ings_text) == "nan":
        return dict(inci_humectant_cnt=0, inci_emollient_cnt=0,
                    inci_exfoliant_cnt=0, inci_preservative_cnt=0,
                    inci_high_irritant_cnt=0, inci_comedogen3plus_cnt=0,
                    inci_matched=0)
    ings = [x.strip() for x in str(ings_text).split(",") if x.strip()]
    hum=emol=exfol=pres=high_irr=com3=matched=0
    for ing in ings:
        idx = match_ingredient(ing, inci_lookup)
        if idx is None: continue
        row   = inci_df.iloc[idx]
        funcs = str(row.get("functions","")).lower()
        if "humectant"    in funcs: hum    += 1
        if "emollient"    in funcs: emol   += 1
        if "exfoliant"    in funcs: exfol  += 1
        if "preservative" in funcs: pres   += 1
        if str(row.get("irritancy_level","")).lower() == "high": high_irr += 1
        try:
            if float(row.get("comedogen_score",0) or 0) >= 3: com3 += 1
        except: pass
        matched += 1
    return dict(inci_humectant_cnt=hum, inci_emollient_cnt=emol,
                inci_exfoliant_cnt=exfol, inci_preservative_cnt=pres,
                inci_high_irritant_cnt=high_irr,
                inci_comedogen3plus_cnt=com3, inci_matched=matched)

print("\nComputing per-product EWG + INCIDecoder features...")
ewg_rows  = prod_df["ingredients"].apply(ewg_features)
inci_rows = prod_df["ingredients"].apply(inci_features)
prod_df   = pd.concat([
    prod_df.reset_index(drop=True),
    pd.DataFrame(ewg_rows.tolist()).reset_index(drop=True),
    pd.DataFrame(inci_rows.tolist()).reset_index(drop=True),
], axis=1)

pct = (prod_df["ewg_matched"] > 0).mean() * 100
print(f"  Products with >=1 EWG match    : {pct:.1f}%")
pct2 = (prod_df["inci_matched"] > 0).mean() * 100
print(f"  Products with >=1 INCIDecoder match : {pct2:.1f}%")

# ── 8. Save master ────────────────────────────────────────────────────────────
prod_df.to_csv("data/processed/skincare_master.csv", index=False)
print(f"\n✓ skincare_master.csv : {len(prod_df)} rows × {len(prod_df.columns)} cols")
print(prod_df["category"].value_counts().to_string())

# ── 9. Build user_ratings.csv ─────────────────────────────────────────────────
print("\nBuilding user_ratings.csv...")
valid_pids = set(prod_df["product_id"].astype(str))
review_df["product_id"] = review_df["product_id"].astype(str)
review_df  = review_df[review_df["product_id"].isin(valid_pids)].copy()
review_df["rating"] = pd.to_numeric(review_df["rating"], errors="coerce")
review_df  = review_df[review_df["rating"].between(1,5)].copy()

def infer_skin(text):
    t = str(text).lower()
    if any(k in t for k in ["oily","acne","pore","sebum"]):       return "oily"
    if any(k in t for k in ["dry skin","very dry","flaky","tight","hydrat"]): return "dry"
    if any(k in t for k in ["combination","combo","t-zone"]):     return "combination"
    if any(k in t for k in ["sensitive","reactive","irritat","redness"]): return "sensitive"
    return "normal"

if "skin_type" not in review_df.columns or review_df.get("skin_type","").eq("").all():
    review_df["skin_type"] = (
        review_df.get("review_text", pd.Series(dtype=str)).fillna("") + " " +
        review_df.get("review_title", pd.Series(dtype=str)).fillna("")
    ).apply(infer_skin)

ratings_out = review_df[["user_id","product_id","rating","skin_type"]].dropna(
    subset=["user_id","product_id","rating"])
ratings_out = ratings_out[ratings_out["user_id"].astype(str).str.strip() != ""]
ratings_out.to_csv("data/processed/user_ratings.csv", index=False)
print(f"✓ user_ratings.csv : {len(ratings_out)} rows, {ratings_out['user_id'].nunique()} users")

# ── 10. Save ingredient_flags reference table ─────────────────────────────────
parts = []
if ewg_df is not None:
    parts.append(ewg_df[["query_name","inci_name","ewg_hazard_score","ewg_hazard_label",
                          "ewg_concerns_raw","concern_cancer","concern_allergy",
                          "concern_endocrine","concern_irritation","concern_restricted",
                          "data_availability","function_tags"]].copy().rename(
                              columns={"query_name":"ingredient_name"}))
if inci_df is not None:
    slim = inci_df[["ingredient_name","functions","irritancy_level",
                    "comedogen_score","skin_suitability","description"]].copy()
    if parts:
        parts[0] = parts[0].merge(
            slim.rename(columns={"ingredient_name":"inci_name",
                                  "functions":"inci_functions",
                                  "irritancy_level":"inci_irritancy",
                                  "comedogen_score":"inci_comedogen_score",
                                  "skin_suitability":"inci_skin_suitability",
                                  "description":"inci_description"}),
            on="inci_name", how="outer")
    else:
        parts.append(slim)

if parts:
    flags = parts[0]
    flags.to_csv("data/processed/ingredient_flags.csv", index=False)
    print(f"✓ ingredient_flags.csv : {len(flags)} rows")

print("\n✓ Data preparation complete. Run: python run_all.py --from 1")
