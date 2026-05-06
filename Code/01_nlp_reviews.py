"""
01_nlp_reviews.py
=================
Natural Language Processing on Nykaa user review text.

Since skin_type is already captured in the review data during scraping,
this script focuses on two NLP tasks that add new information:

  Task 1 — Skin Concern Identification
           Detects which skin concerns (acne, pigmentation, dryness,
           sensitivity, anti-aging, pores, dark circles, uneven texture)
           the reviewer mentions. Produces binary flags per concern.
           These are the concerns the CUSTOMER had when purchasing.

  Task 2 — Customer Satisfaction Classification
           Classifies each review as Satisfied / Neutral / Dissatisfied
           using a valence lexicon + star rating signal.
           Produces:
             satisfaction_label  (Satisfied / Neutral / Dissatisfied)
             satisfaction_score  (continuous -1 to +1)
           Used as dependent variable in RQ1 and evaluation signal in RQ2.

  Also runs:
  - Aspect-level sentiment (hydration, texture, fragrance, breakouts,
    value, sensitivity) for EDA and RQ3 analysis
  - LDA topic modelling for literature validation

Outputs:
  data/processed/user_ratings.csv        enriched with all NLP columns
  results/nlp_concern_dist.png           skin concern frequency chart
  results/nlp_satisfaction_dist.png      satisfaction distribution
  results/nlp_satisfaction_by_skin.png   satisfaction by skin type
  results/nlp_concern_by_skin.png        concern heatmap by skin type
  results/nlp_concern_vs_satisfaction.png concern impact on satisfaction
  results/nlp_topic_keywords.csv         LDA topic top-words
  results/nlp_aspect_scores.csv          aspect sentiment summary

Run: python 01_nlp_reviews.py
"""

import pandas as pd
import numpy as np
import re, os, warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN = True
except ImportError:
    SKLEARN = False

try:
    from wordcloud import WordCloud
    WC = True
except ImportError:
    WC = False

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("NLP ANALYSIS — Skin Concerns + Customer Satisfaction")
print("=" * 60)

# Prefer the processed user_ratings.csv (has product_id, clean data, review_text)
# Fall back to raw nykaa_reviews.csv if processed doesn't exist yet.
processed_path = "data/processed/user_ratings.csv"
raw_path       = "data/raw/nykaa_reviews.csv"

if os.path.exists(processed_path):
    df = pd.read_csv(processed_path)
    print(f"\nLoaded from processed: {len(df):,} rows")
    print("  (Run 00_clean_nykaa.py first if this looks wrong)")
elif os.path.exists(raw_path):
    df = pd.read_csv(raw_path)
    print(f"\nLoaded from raw: {len(df):,} rows")
else:
    print(f"ERROR: neither {processed_path} nor {raw_path} found.")
    print("Run nykaa_full_scraper.py then 00_clean_nykaa.py first.")
    exit(1)

# Identify review text column
TEXT_COL = next(
    (c for c in ["review_text","review_description","description","review","comment","text"]
     if c in df.columns), None
)
if TEXT_COL is None:
    df["review_text"] = ""
    TEXT_COL = "review_text"
elif TEXT_COL != "review_text":
    df["review_text"] = df[TEXT_COL]

# skin_type is captured directly from Nykaa review metadata.
# Read the column as-is. Never infer from review text.
SKIN_COL = next(
    (c for c in ["skin_type", "skin_type_tag", "reviewer_skin_type", "skintype"]
     if c in df.columns), None
)
if SKIN_COL:
    if SKIN_COL != "skin_type":
        df["skin_type"] = df[SKIN_COL]
    # Normalise capitalisation
    df["skin_type"] = df["skin_type"].fillna("").str.strip().str.lower()
    n_with = (df["skin_type"].str.len() > 0).sum()
    print(f"Skin type column: '{SKIN_COL}'  |  {n_with:,} non-empty values")
    print(df["skin_type"].value_counts().head(10).to_string())
else:
    print("NOTE: No skin_type column found in data.")
    print("      skin_type should be scraped from Nykaa review metadata.")
    print("      Check that nykaa_full_scraper.py is saving the skinType field.")
    df["skin_type"] = ""

df["rating"]       = pd.to_numeric(df.get("rating", pd.Series(dtype=float)), errors="coerce")
df                 = df[df["rating"].between(1, 5)].copy()
df["review_text"]  = df["review_text"].fillna("").astype(str)
df["review_title"] = df.get("review_title", pd.Series(dtype=str)).fillna("").astype(str)
df["full_text"]    = (df["review_title"] + " " + df["review_text"]).str.strip()

print(f"Reviews with valid ratings : {len(df):,}")
print(f"Reviews with text content  : {(df['full_text'].str.len()>5).sum():,}")
if "skin_type" in df.columns:
    print(f"\nSkin type distribution (from scraped data):")
    print(df["skin_type"].value_counts().head(8).to_string())


# ── 2. Text cleaning ──────────────────────────────────────────────────────────
def clean_text(text):
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-z\s'-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["clean_text"] = df["full_text"].apply(clean_text)


# ── 3. TASK 1: Skin Concern Identification ────────────────────────────────────
# Identifies which skin concerns the reviewer was addressing when they
# purchased and reviewed the product. This is independent of skin type —
# a reviewer with oily skin may have concerns about acne AND pigmentation.
# Binary flag per concern: 1 = mentioned, 0 = not mentioned.

CONCERN_PATTERNS = {
    "concern_acne": [
        r"\bacne\b", r"\bpimple[s]?\b", r"\bbreakout[s]?\b", r"\bblemish",
        r"\bwhitehead", r"\bblackhead", r"\bcystic\b", r"\bzit[s]?\b",
        r"\bblackspot", r"\bspots\b",
    ],
    "concern_pigmentation": [
        r"\bhyperpigment", r"\bdark\s*spot[s]?\b", r"\bpigment",
        r"\buneven\s*tone", r"\bdull\s*skin", r"\bdiscolo",
        r"\btanning\b", r"\bdarkening", r"\bpost[\s-]acne",
        r"\bred\s*mark[s]?\b", r"\bmelanin",
    ],
    "concern_dryness": [
        r"\bdry\s*skin\b", r"\bdehydrat", r"\bflak", r"\btight\s*skin",
        r"\bpeel", r"\bparched\b", r"\brough\s*skin", r"\bcrack",
        r"\bdryness\b", r"\black\s*of\s*moisture",
    ],
    "concern_oiliness": [
        r"\boily\s*skin\b", r"\bexcess\s*oil", r"\bgreasy\b",
        r"\bsebum\b", r"\bmattif", r"\boil\s*control",
        r"\bshiny\s*face", r"\bt[\s-]zone",
    ],
    "concern_sensitivity": [
        r"\bsensitive\s*skin\b", r"\birritatr?", r"\breactive\b",
        r"\bredness\b", r"\bsting", r"\bburn", r"\bitch",
        r"\ballerg", r"\breaction\b", r"\binflam",
    ],
    "concern_aging": [
        r"\bwrinkle[s]?\b", r"\bfine\s*line[s]?\b", r"\baging\b",
        r"\bsagging\b", r"\bfirmness\b", r"\belastic",
        r"\banti[\s-]age", r"\bcrow[s]?\s*feet", r"\bjowl",
    ],
    "concern_pores": [
        r"\bpore[s]?\b", r"\bopen\s*pore", r"\blarge\s*pore",
        r"\bminimiz.*pore", r"\bpore[\s-]less",
    ],
    "concern_dark_circles": [
        r"\bdark\s*circle[s]?\b", r"\bunder[\s-]eye",
        r"\bpuffy\s*eye", r"\beye\s*bag[s]?\b", r"\beye\s*area",
    ],
    "concern_texture": [
        r"\buneven\s*texture", r"\brough\s*texture", r"\bbumpy\b",
        r"\btexture\b", r"\bskin\s*texture", r"\bsmooth.*skin",
    ],
}

def identify_concerns(text):
    t = str(text).lower()
    return {
        concern: int(any(re.search(p, t) for p in patterns))
        for concern, patterns in CONCERN_PATTERNS.items()
    }

print("\nIdentifying skin concerns from review text...")
concern_df = pd.DataFrame(df["clean_text"].apply(identify_concerns).tolist())
df = pd.concat([df, concern_df], axis=1)

concern_cols = list(CONCERN_PATTERNS.keys())
concern_counts = {c: int(df[c].sum()) for c in concern_cols}
print("\nSkin concern mentions:")
for c, n in sorted(concern_counts.items(), key=lambda x: -x[1]):
    print(f"  {c:<35}: {n:5,} ({n/len(df)*100:.1f}%)")


# ── 4. TASK 2: Customer Satisfaction Classification ───────────────────────────
# Combines review text sentiment with the star rating.
# satisfaction_label: Satisfied / Neutral / Dissatisfied
# satisfaction_score: -1 to +1 continuous (used in RQ1 regression and RQ2)

SATISFIED_WORDS = {
    "love","loved","great","excellent","amazing","perfect","best","fantastic",
    "wonderful","awesome","brilliant","superb","outstanding","recommend",
    "very happy","satisfied","worth","effective","works","worked","visible",
    "results","impressed","pleased","glad","repurchase","buy again",
    "favourite","favorite","definitely",
}
DISSATISFIED_WORDS = {
    "disappointed","disappointing","waste","terrible","horrible","awful",
    "worst","useless","doesn't work","didn't work","not work","no results",
    "bad","poor","regret","return","refund","broke out","breakout",
    "irritated","burned","stung","reaction","rash","avoid",
    "not worth","overpriced","do not buy","don't buy",
}

def compute_satisfaction(row):
    text      = str(row["clean_text"]).lower()
    pos_score = sum(1 for w in SATISFIED_WORDS    if w in text)
    neg_score = sum(1 for w in DISSATISFIED_WORDS if w in text)
    total     = pos_score + neg_score
    text_score = (pos_score - neg_score) / total if total > 0 else 0.0

    # Rating normalised to -1 to +1  (1→-1, 3→0, 5→+1)
    rating    = float(row.get("rating", 3))
    rat_score = (rating - 3) / 2

    # Weighted combination: 60% text signal, 40% rating signal
    combined  = round(0.6 * text_score + 0.4 * rat_score, 3)

    # Label: rating is primary, text breaks ties at 3 stars
    if   rating >= 4.0 or (rating == 3.0 and text_score > 0.2):
        label = "Satisfied"
    elif rating <= 2.0 or (rating == 3.0 and text_score < -0.2):
        label = "Dissatisfied"
    else:
        label = "Neutral"

    return pd.Series({"satisfaction_score": combined, "satisfaction_label": label})

print("\nClassifying customer satisfaction...")
sat_df = df.apply(compute_satisfaction, axis=1)
df     = pd.concat([df, sat_df], axis=1)

sat_counts = df["satisfaction_label"].value_counts()
print("\nSatisfaction distribution:")
print(sat_counts.to_string())
print(f"Mean satisfaction score: {df['satisfaction_score'].mean():.3f}")


# ── 5. Aspect-level sentiment (EDA + RQ3) ────────────────────────────────────
ASPECT_POS = {
    "hydration"  : ["hydrat","moistur","nourish","plump","dewy","soft","supple","glow"],
    "texture"    : ["smooth","light","absorb","silky","non-greasy","velvety","thin"],
    "fragrance"  : ["smell","scent","fragranc","aroma","pleasant","nice smell"],
    "breakouts"  : ["clear","no breakout","non-comedogenic","no acne","no pimple"],
    "value"      : ["worth","value","affordable","cheap","budget","good deal","reasonable"],
    "sensitivity": ["gentle","mild","soothing","calm","no irritat","no react","hypoallergen"],
}
ASPECT_NEG = {
    "hydration"  : ["dry","tight","flaky","peel","not hydrat","no moisture"],
    "texture"    : ["heavy","greasy","oily","thick","sticky","clogs","clog pore"],
    "fragrance"  : ["strong smell","bad smell","pungent","chemical smell","overwhelming"],
    "breakouts"  : ["breakout","acne","pimple","comedogenic","clog","blackhead"],
    "value"      : ["expensive","pricey","overpriced","not worth","waste","too costly"],
    "sensitivity": ["irritat","burn","sting","react","redness","allerg","rash"],
}

def aspect_score(text, aspect):
    t = str(text).lower()
    pos = sum(1 for kw in ASPECT_POS.get(aspect,[]) if kw in t)
    neg = sum(1 for kw in ASPECT_NEG.get(aspect,[]) if kw in t)
    return 1 if pos > neg else -1 if neg > pos else 0

for aspect in ASPECT_POS:
    df[f"aspect_{aspect}"] = df["clean_text"].apply(lambda t: aspect_score(t, aspect))


# ── 6. LDA Topic Modelling ────────────────────────────────────────────────────
STOP_WORDS = {
    "skin","product","use","like","good","great","really","just","get","very",
    "my","it","and","or","the","a","an","is","in","to","for","this","that",
    "with","have","been","so","on","of","am","was","i","me","its","you",
    "we","not","but","be","are","has","if","will","do","did","does","can",
}
if SKLEARN:
    texts_for_lda = df["clean_text"][df["clean_text"].str.len() > 20].tolist()
    if len(texts_for_lda) >= 100:
        print("\nRunning LDA topic modelling...")
        cv  = CountVectorizer(max_features=800, min_df=5, max_df=0.85,
                              stop_words=list(STOP_WORDS), ngram_range=(1,2))
        dtm = cv.fit_transform(texts_for_lda)
        lda = LatentDirichletAllocation(n_components=6, random_state=42,
                                        learning_method="batch", max_iter=20)
        lda.fit(dtm)
        feat_names   = cv.get_feature_names_out()
        topic_labels = ["Hydration & Moisture","Texture & Absorption",
                        "Acne & Oiliness","Fragrance & Sensory",
                        "Brightening & Anti-aging","Sensitivity & Reactions"]
        rows = []
        for i, comp in enumerate(lda.components_):
            kws   = [feat_names[j] for j in comp.argsort()[-12:][::-1]]
            label = topic_labels[i] if i < len(topic_labels) else f"Topic {i+1}"
            rows.append({"topic": label, "keywords": ", ".join(kws)})
            print(f"  Topic {i+1} ({label}): {', '.join(kws[:6])}")
        pd.DataFrame(rows).to_csv("results/nlp_topic_keywords.csv", index=False)


# ── 7. Figures ────────────────────────────────────────────────────────────────
PALETTE = ["#E74C3C","#E67E22","#F1C40F","#2ECC71","#27AE60",
           "#3498DB","#9B59B6","#1ABC9C","#E91E63"]

# Figure 1: Skin concern frequency
fig, ax = plt.subplots(figsize=(10, 5))
labels  = [c.replace("concern_","").replace("_"," ").title() for c in concern_cols]
values  = [df[c].sum() for c in concern_cols]
pairs   = sorted(zip(values, labels), reverse=True)
vals_s, lbs_s = zip(*pairs)
bars = ax.barh(lbs_s, vals_s, color=PALETTE[:len(lbs_s)], edgecolor="white")
ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
ax.set_xlabel("Number of Reviews Mentioning Concern")
ax.set_title("Figure 1. Skin Concern Distribution Identified from Review Text (NLP)\n"
             "(Preliminary — one review may mention multiple concerns)",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, max(vals_s) * 1.12)
plt.tight_layout()
plt.savefig("results/nlp_concern_dist.png", dpi=150)
plt.close()
print("\n→ Figure 1 saved: results/nlp_concern_dist.png")

# Figure 2: Satisfaction distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sat_order  = ["Satisfied","Neutral","Dissatisfied"]
sat_colors = ["#2ECC71","#F39C12","#E74C3C"]
sat_vals   = [sat_counts.get(k,0) for k in sat_order]
axes[0].bar(sat_order, sat_vals, color=sat_colors, edgecolor="white")
for i, v in enumerate(sat_vals):
    axes[0].text(i, v + len(df)*0.005, f"{v/len(df)*100:.1f}%", ha="center", fontsize=10)
axes[0].set_title("(a) Satisfaction Label Distribution", fontsize=11)
axes[0].set_ylabel("Number of Reviews")

axes[1].hist(df["satisfaction_score"], bins=30, color="#45B7D1", edgecolor="white")
axes[1].axvline(df["satisfaction_score"].mean(), color="red", linestyle="--",
                label=f"Mean={df['satisfaction_score'].mean():.2f}")
axes[1].set_title("(b) Satisfaction Score Distribution", fontsize=11)
axes[1].set_xlabel("Score (-1 = Dissatisfied, +1 = Satisfied)")
axes[1].legend()

fig.suptitle("Figure 2. Customer Satisfaction from NLP Analysis (Preliminary)\n"
             "(Text lexicon + star rating; 60/40 weighted combination)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("results/nlp_satisfaction_dist.png", dpi=150)
plt.close()
print("→ Figure 2 saved: results/nlp_satisfaction_dist.png")

# Figure 3: Satisfaction by skin type (uses scraped skin_type column)
skin_types = []
if "skin_type" in df.columns:
    skin_types = [s for s in ["oily","dry","combination","sensitive","normal"]
                  if df["skin_type"].eq(s).sum() > 10]

if skin_types:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    skin_df = df[df["skin_type"].isin(skin_types)]
    mean_s  = skin_df.groupby("skin_type")["satisfaction_score"].mean().reindex(skin_types)
    sat_pct = skin_df.groupby("skin_type")["satisfaction_label"].apply(
        lambda x: (x=="Satisfied").mean()*100).reindex(skin_types)

    axes[0].bar(skin_types, mean_s.values,
                color=PALETTE[:len(skin_types)], edgecolor="white")
    axes[0].set_title("(a) Mean Satisfaction Score by Skin Type", fontsize=11)
    axes[0].set_ylabel("Mean Score")

    axes[1].bar(skin_types, sat_pct.values,
                color=PALETTE[:len(skin_types)], edgecolor="white")
    for i, v in enumerate(sat_pct.values):
        axes[1].text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=9)
    axes[1].set_title("(b) % Satisfied by Skin Type", fontsize=11)
    axes[1].set_ylabel("% Satisfied"); axes[1].set_ylim(0, 110)

    fig.suptitle("Figure 3. Customer Satisfaction by Skin Type (Preliminary)\n"
                 "(Skin type from scraped review metadata — not NLP-inferred)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/nlp_satisfaction_by_skin.png", dpi=150)
    plt.close()
    print("→ Figure 3 saved: results/nlp_satisfaction_by_skin.png")

    # Figure 4: Concern heatmap by skin type
    fig, ax = plt.subplots(figsize=(12, 5))
    heat = (skin_df.groupby("skin_type")[concern_cols].mean().reindex(skin_types)
            .rename(columns={c: c.replace("concern_","").replace("_"," ").title()
                             for c in concern_cols}))
    sns.heatmap(heat.T, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Proportion of reviews mentioning concern"})
    ax.set_title("Figure 4. Skin Concerns by Skin Type (Preliminary)\n"
                 "(Concern from NLP; skin type from scraped metadata)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/nlp_concern_by_skin.png", dpi=150)
    plt.close()
    print("→ Figure 4 saved: results/nlp_concern_by_skin.png")

# Figure 5: Concern vs satisfaction
fig, ax = plt.subplots(figsize=(10, 5))
labels_c   = [c.replace("concern_","").replace("_"," ").title() for c in concern_cols]
with_vals  = [df[df[c]==1]["satisfaction_score"].mean() for c in concern_cols]
witho_vals = [df[df[c]==0]["satisfaction_score"].mean() for c in concern_cols]
x = np.arange(len(concern_cols))
ax.bar(x-0.2, with_vals,  0.4, label="Concern Mentioned",     color="#E74C3C", alpha=0.8)
ax.bar(x+0.2, witho_vals, 0.4, label="Concern Not Mentioned", color="#2ECC71", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels_c, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean Satisfaction Score")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.legend()
ax.set_title("Figure 5. Satisfaction Score: Concern Mentioned vs Not Mentioned (Preliminary)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("results/nlp_concern_vs_satisfaction.png", dpi=150)
plt.close()
print("→ Figure 5 saved: results/nlp_concern_vs_satisfaction.png")


# ── 8. Save enriched user_ratings.csv ────────────────────────────────────────
print("\nSaving enriched user_ratings.csv...")
keep_cols  = ["user_id","product_id","rating","review_text","review_title","skin_type",
              "satisfaction_label","satisfaction_score"]
keep_cols += concern_cols
keep_cols += [f"aspect_{a}" for a in ASPECT_POS]

out = df[[c for c in keep_cols if c in df.columns]].copy()
out = out.dropna(subset=["user_id","product_id","rating"])
out = out[out["user_id"].astype(str).str.strip() != ""]
out.to_csv("data/processed/user_ratings.csv", index=False)
print(f"✓ user_ratings.csv : {len(out):,} rows, {out['user_id'].nunique():,} users")
print(f"  Columns : {list(out.columns)}")

aspect_cols = [f"aspect_{a}" for a in ASPECT_POS]
if "skin_type" in out.columns:
    out.groupby("skin_type")[aspect_cols].mean().round(3).to_csv("results/nlp_aspect_scores.csv")
    print("✓ results/nlp_aspect_scores.csv saved")


# ── 9. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("NLP SUMMARY (PRELIMINARY)")
print(f"{'='*60}")
print(f"Total reviews           : {len(df):,}")
print(f"With text content       : {(df['clean_text'].str.len()>10).sum():,}")
print(f"\nSatisfaction breakdown:")
print(df["satisfaction_label"].value_counts().to_string())
print(f"\nTop concerns identified:")
for c, n in sorted(concern_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {c:<35}: {n:,}")
print(f"\nNote: Results are preliminary — full analysis pending")
print(f"      INCIDecoder scraping completion.")
print(f"\nNext: python 00_clean_nykaa.py")
