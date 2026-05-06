"""
01b_eda.py
==========
Exploratory Data Analysis for the Personalized Skincare Recommendation System.
Generates all figures and tables needed for the Interim Report Section 4.

Run after: python 01_nlp_reviews.py && python 00_clean_nykaa.py

Outputs (all to results/eda_*.png and results/eda_*.csv):
  Figure 1  — Rating distribution (histogram + KDE)
  Figure 2  — Product count by category
  Figure 3  — Price distribution by category (boxplot)
  Figure 4  — Rating vs price tier (boxplot)
  Figure 5  — Ingredient count distribution
  Figure 6  — Top 20 brands by product count
  Figure 7  — Skin type suitability heatmap
  Figure 8  — EWG/GHS hazard score distribution
  Figure 9  — INCIDecoder function tag frequency
  Figure 10 — User rating distribution (reviews)
  Figure 11 — Reviews per user distribution
  Figure 12 — Correlation matrix of numeric features
  Table 1   — Descriptive statistics CSV
  Table 2   — Missing value summary CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import os, warnings

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

sns.set_theme(style="whitegrid")
PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
           "#DDA0DD", "#98D8C8", "#F7DC6F"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

prod_path   = "data/processed/skincare_master.csv"
review_path = "data/processed/user_ratings.csv"

if not os.path.exists(prod_path):
    print(f"ERROR: {prod_path} not found. Run 00_clean_nykaa.py first.")
    exit(1)

prod = pd.read_csv(prod_path)
print(f"\nProducts loaded : {len(prod):,} rows × {len(prod.columns)} cols")

rev = pd.read_csv(review_path) if os.path.exists(review_path) else pd.DataFrame()
if len(rev):
    print(f"Reviews loaded  : {len(rev):,} rows, {rev['user_id'].nunique():,} users")

# ── Table 1: Descriptive statistics ──────────────────────────────────────────
# irritant_count, comedogen_count removed — now sourced from INCIDecoder only
# has_fragrance, is_fragrance_free, skin_oily/dry etc removed — not in product CSV
num_cols = ["rating", "num_ratings", "price_inr", "ingredient_count",
            "ewg_max_hazard", "ewg_mean_hazard",
            "inci_humectant_cnt", "inci_emollient_cnt",
            "inci_exfoliant_cnt", "inci_high_irritant_cnt",
            "inci_comedogen3plus_cnt"]
num_cols = [c for c in num_cols if c in prod.columns]

desc = prod[num_cols].describe().round(3)
desc.to_csv("results/eda_descriptive_stats.csv")
print(f"\nTable 1 — Descriptive statistics saved.")
print(desc.T.to_string())

# ── Table 2: Missing values ───────────────────────────────────────────────────
missing = pd.DataFrame({
    "column": prod.columns,
    "missing_count":  prod.isnull().sum().values,
    "missing_pct":    (prod.isnull().sum() / len(prod) * 100).round(2).values,
    "dtype": prod.dtypes.values,
}).query("missing_count > 0").sort_values("missing_pct", ascending=False)
missing.to_csv("results/eda_missing_values.csv", index=False)
print(f"\nTable 2 — Missing values: {len(missing)} columns have missing data")


# ── Figure 1: Rating distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Histogram
axes[0].hist(prod["rating"].dropna(), bins=20, color="#4ECDC4",
             edgecolor="white", linewidth=0.8)
axes[0].set_xlabel("Aggregate Rating (1–5)")
axes[0].set_ylabel("Number of Products")
axes[0].set_title("(a) Product Rating Distribution", fontsize=11)
axes[0].axvline(prod["rating"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {prod['rating'].mean():.2f}")
axes[0].legend()

# By category
cats = prod["category"].value_counts().index.tolist()[:4]
for cat, color in zip(cats, PALETTE[:4]):
    subset = prod[prod["category"] == cat]["rating"].dropna()
    axes[1].hist(subset, bins=15, alpha=0.6, label=cat, color=color, edgecolor="white")
axes[1].set_xlabel("Rating")
axes[1].set_ylabel("Count")
axes[1].set_title("(b) Rating Distribution by Category", fontsize=11)
axes[1].legend(fontsize=9)

fig.suptitle("Figure 1. Distribution of Nykaa Product Aggregate Ratings\n"
             "(Products with ≥5 ratings and valid 1.0–5.0 range)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("results/eda_fig1_rating_dist.png", dpi=150)
plt.close()
print("→ Figure 1 saved: eda_fig1_rating_dist.png")


# ── Figure 2: Product count by category ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
cat_counts = prod["category"].value_counts()
bars = ax.barh(cat_counts.index, cat_counts.values,
               color=PALETTE[:len(cat_counts)], edgecolor="white")
ax.bar_label(bars, fmt="%d", padding=4)
ax.set_xlabel("Number of Products")
ax.set_title("Figure 2. Product Count by Category\n(Nykaa.com scraped data)",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, cat_counts.max() * 1.15)
plt.tight_layout()
plt.savefig("results/eda_fig2_category_counts.png", dpi=150)
plt.close()
print("→ Figure 2 saved: eda_fig2_category_counts.png")


# ── Figure 3: Price distribution by category ─────────────────────────────────
if "price_inr" in prod.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = prod[prod["price_inr"].between(0, 5000)].copy()
    cats_order = prod["category"].value_counts().index.tolist()[:5]
    plot_sub = plot_df[plot_df["category"].isin(cats_order)]
    sns.boxplot(data=plot_sub, x="category", y="price_inr",
                order=cats_order, palette=PALETTE[:5], ax=ax,
                flierprops=dict(marker=".", alpha=0.3, markersize=3))
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Price (INR)")
    ax.set_title("Figure 3. Price Distribution by Product Category\n"
                 "(Outliers >₹5,000 excluded for readability)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/eda_fig3_price_by_category.png", dpi=150)
    plt.close()
    print("→ Figure 3 saved: eda_fig3_price_by_category.png")


# ── Figure 4: Rating by price tier ───────────────────────────────────────────
if "price_tier" in prod.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    tier_order = ["Budget", "Mid-Range", "Luxury"]
    tier_order = [t for t in tier_order if t in prod["price_tier"].values]

    sns.boxplot(data=prod, x="price_tier", y="rating",
                order=tier_order, palette=PALETTE[:3], ax=axes[0],
                flierprops=dict(marker=".", alpha=0.3, markersize=3))
    axes[0].set_title("(a) Rating by Price Tier", fontsize=11)
    axes[0].set_xlabel("Price Tier")
    axes[0].set_ylabel("Aggregate Rating")
    axes[0].set_ylim(0.5, 5.5)

    # Mean rating + CI
    tier_stats = prod.groupby("price_tier")["rating"].agg(["mean","sem"]).reindex(tier_order)
    axes[1].bar(tier_order, tier_stats["mean"],
                yerr=tier_stats["sem"] * 1.96,
                color=PALETTE[:3], edgecolor="white",
                capsize=5, error_kw=dict(linewidth=1.5))
    axes[1].set_title("(b) Mean Rating ± 95% CI by Price Tier", fontsize=11)
    axes[1].set_xlabel("Price Tier")
    axes[1].set_ylabel("Mean Rating")
    axes[1].set_ylim(0, 5.5)
    for i, (tier, row) in enumerate(tier_stats.iterrows()):
        axes[1].text(i, row["mean"] + row["sem"] * 2 + 0.1,
                     f"{row['mean']:.2f}", ha="center", fontsize=10)

    fig.suptitle("Figure 4. Relationship Between Price Tier and User Rating\n"
                 "(Budget: <₹500; Mid-Range: ₹500–1,500; Luxury: >₹1,500)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/eda_fig4_rating_by_price.png", dpi=150)
    plt.close()
    print("→ Figure 4 saved: eda_fig4_rating_by_price.png")


# ── Figure 5: Ingredient count distribution ──────────────────────────────────
if "ingredient_count" in prod.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ic = prod["ingredient_count"].dropna()
    axes[0].hist(ic, bins=30, color="#FF6B6B", edgecolor="white")
    axes[0].axvline(ic.mean(), color="navy", linestyle="--",
                    label=f"Mean = {ic.mean():.1f}")
    axes[0].axvline(ic.median(), color="green", linestyle=":",
                    label=f"Median = {ic.median():.1f}")
    axes[0].set_xlabel("Number of Ingredients")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) Ingredient Count per Product", fontsize=11)
    axes[0].legend()

    # Scatter: ingredient count vs rating
    scatter_df = prod[["ingredient_count", "rating"]].dropna()
    axes[1].scatter(scatter_df["ingredient_count"], scatter_df["rating"],
                    alpha=0.15, s=10, color="#4ECDC4")
    m, b, r, p, _ = stats.linregress(scatter_df["ingredient_count"], scatter_df["rating"])
    x_range = np.linspace(scatter_df["ingredient_count"].min(),
                          scatter_df["ingredient_count"].max(), 100)
    axes[1].plot(x_range, m * x_range + b, "r-", linewidth=2,
                 label=f"r = {r:.3f}, p = {p:.3f}")
    axes[1].set_xlabel("Ingredient Count")
    axes[1].set_ylabel("Rating")
    axes[1].set_title("(b) Ingredient Count vs. Rating", fontsize=11)
    axes[1].legend()

    fig.suptitle("Figure 5. Ingredient Count Distribution and Relationship with Rating",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/eda_fig5_ingredient_count.png", dpi=150)
    plt.close()
    print("→ Figure 5 saved: eda_fig5_ingredient_count.png")


# ── Figure 6: Top brands ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
top_brands = prod["brand"].value_counts().head(20)
ax.barh(top_brands.index[::-1], top_brands.values[::-1],
        color="#96CEB4", edgecolor="white")
ax.set_xlabel("Number of Products")
ax.set_title("Figure 6. Top 20 Brands by Product Count on Nykaa.com",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("results/eda_fig6_top_brands.png", dpi=150)
plt.close()
print("→ Figure 6 saved: eda_fig6_top_brands.png")


# ── Figure 7: Skin type distribution from reviewer metadata ──────────────────
# skin_oily/dry/combination/sensitive/normal columns removed from skincare_master.csv
# Skin type is now in user_ratings.csv from reviewer profile metadata
if len(rev) and "skin_type" in rev.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Fig 7a: skin type distribution (reviewer counts)
    skin_counts = (rev["skin_type"]
                   .str.strip().str.lower()
                   .replace("", np.nan)
                   .dropna()
                   .value_counts())
    skin_order = ["normal","oily","dry","sensitive","combination"]
    skin_order = [s for s in skin_order if s in skin_counts.index]
    sk_colors  = {"normal":"#F39C12","oily":"#E74C3C","dry":"#3498DB",
                  "sensitive":"#2ECC71","combination":"#9B59B6"}
    bars = axes[0].bar([s.capitalize() for s in skin_order],
                       [skin_counts[s] for s in skin_order],
                       color=[sk_colors.get(s,"#888") for s in skin_order],
                       edgecolor="white")
    axes[0].bar_label(bars, fmt="%d", padding=3, fontsize=8)
    axes[0].set_xlabel("Skin Type")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_title("(a) Skin Type Distribution\n(From scraped reviewer metadata)",
                      fontsize=11)

    # Fig 7b: mean satisfaction by skin type (if available)
    if "satisfaction_score" in rev.columns:
        rev_clean = rev[rev["skin_type"].str.strip().str.lower().isin(skin_order)].copy()
        rev_clean["skin_type"] = rev_clean["skin_type"].str.strip().str.lower()
        sat_by_skin = rev_clean.groupby("skin_type")["satisfaction_score"].mean().reindex(skin_order)
        axes[1].bar([s.capitalize() for s in skin_order],
                    sat_by_skin.values,
                    color=[sk_colors.get(s,"#888") for s in skin_order],
                    edgecolor="white")
        for i, v in enumerate(sat_by_skin.values):
            axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        axes[1].set_xlabel("Skin Type")
        axes[1].set_ylabel("Mean Satisfaction Score (-1 to +1)")
        axes[1].set_title("(b) Mean Satisfaction Score by Skin Type", fontsize=11)
        axes[1].set_ylim(0, sat_by_skin.max() * 1.15)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "satisfaction_score not yet computed\n(run 01_nlp_reviews.py)",
                     transform=axes[1].transAxes, ha="center", va="center", fontsize=10)

    fig.suptitle("Figure 7. Skin Type Distribution and Satisfaction by Skin Type\n"
                 "(Skin type from Nykaa reviewer profile metadata — not NLP-inferred)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/eda_fig7_skin_heatmap.png", dpi=150)
    plt.close()
    print("→ Figure 7 saved: eda_fig7_skin_heatmap.png")
elif "skin_type" not in rev.columns:
    print("→ Figure 7 skipped: skin_type column not in user_ratings.csv")


# ── Figure 8: GHS/EWG hazard distribution ────────────────────────────────────
ewg_col = next((c for c in ["ewg_mean_hazard","ewg_max_hazard","ghs_hazard_score"]
                if c in prod.columns and prod[c].notna().sum() > 50), None)
if ewg_col:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    hazard = prod[ewg_col].dropna()
    axes[0].hist(hazard, bins=20, color="#FF6B6B", edgecolor="white")
    axes[0].set_xlabel("Hazard Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"(a) Distribution of {ewg_col.replace('_',' ').title()}", fontsize=11)

    label_col = "ewg_hazard_label" if "ewg_hazard_label" in prod.columns else None
    if label_col:
        label_counts = prod[label_col].value_counts()
        axes[1].pie(label_counts.values,
                    labels=label_counts.index,
                    colors=["#2ECC71","#F39C12","#E74C3C","#BDC3C7"],
                    autopct="%1.1f%%", startangle=90)
        axes[1].set_title("(b) Hazard Label Distribution", fontsize=11)
    else:
        axes[1].axis("off")

    fig.suptitle("Figure 8. GHS/EWG Ingredient Hazard Score Distribution\n"
                 "(Based on PubChem GHS classifications aggregated per product)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/eda_fig8_hazard_dist.png", dpi=150)
    plt.close()
    print("→ Figure 8 saved: eda_fig8_hazard_dist.png")


# ── Figure 9: INCIDecoder function tag frequency ─────────────────────────────
inci_cols = {
    "inci_humectant_cnt":     "Humectant",
    "inci_emollient_cnt":     "Emollient",
    "inci_exfoliant_cnt":     "Exfoliant",
    "inci_preservative_cnt":  "Preservative",
    "inci_high_irritant_cnt": "High Irritant",
    "inci_comedogen3plus_cnt":"Comedogenic ≥3",
}
available_inci = {k: v for k, v in inci_cols.items() if k in prod.columns}
if available_inci:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Mean count per product
    means = {v: prod[k].mean() for k, v in available_inci.items()}
    axes[0].bar(means.keys(), means.values(),
                color=PALETTE[:len(means)], edgecolor="white")
    axes[0].set_xlabel("Ingredient Function Tag")
    axes[0].set_ylabel("Mean Count per Product")
    axes[0].set_title("(a) Mean INCIDecoder Function Tag Counts", fontsize=11)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha="right")

    # % products with ≥1 ingredient of each type
    pcts = {v: (prod[k] > 0).mean() * 100 for k, v in available_inci.items()}
    axes[1].bar(pcts.keys(), pcts.values(),
                color=PALETTE[:len(pcts)], edgecolor="white")
    axes[1].set_xlabel("Ingredient Function Tag")
    axes[1].set_ylabel("% Products with ≥1 Ingredient of This Type")
    axes[1].set_title("(b) Product Coverage by Function Tag", fontsize=11)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle("Figure 9. INCIDecoder Ingredient Function Tag Distribution\n"
                 "(Function tags from INCIDecoder; comedogenicity from Fulton 1984 scale)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/eda_fig9_inci_functions.png", dpi=150)
    plt.close()
    print("→ Figure 9 saved: eda_fig9_inci_functions.png")


# ── Figure 10 & 11: Review-level analysis ────────────────────────────────────
if len(rev) and "rating" in rev.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Rating distribution in reviews
    rev_ratings = rev["rating"].dropna()
    rating_counts = rev_ratings.value_counts().sort_index()
    axes[0].bar(rating_counts.index.astype(str), rating_counts.values,
                color=["#E74C3C","#E67E22","#F1C40F","#2ECC71","#27AE60"],
                edgecolor="white")
    for i, (k, v) in enumerate(rating_counts.items()):
        pct = v / len(rev_ratings) * 100
        axes[0].text(i, v + len(rev_ratings)*0.01, f"{pct:.1f}%",
                     ha="center", fontsize=9)
    axes[0].set_xlabel("Star Rating")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_title("(a) Distribution of Individual User Ratings", fontsize=11)

    # Reviews per user
    rpu = rev.groupby("user_id").size()
    axes[1].hist(rpu, bins=min(50, rpu.max()),
                 color="#45B7D1", edgecolor="white")
    axes[1].set_xlabel("Number of Reviews per User")
    axes[1].set_ylabel("Number of Users")
    axes[1].set_title(f"(b) Reviews per User\n"
                      f"(Mean = {rpu.mean():.1f}, Median = {rpu.median():.0f})",
                      fontsize=11)
    axes[1].set_xlim(0, min(rpu.max(), 100))

    fig.suptitle("Figure 10. Individual User Rating Distribution and User Activity\n"
                 "(Nykaa.com scraped reviews; ratings on 1–5 star scale)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/eda_fig10_review_dist.png", dpi=150)
    plt.close()
    print("→ Figure 10 saved: eda_fig10_review_dist.png")


# ── Figure 11: Correlation matrix ────────────────────────────────────────────
corr_cols = [c for c in num_cols if c in prod.columns and prod[c].notna().sum() > 100]
if len(corr_cols) >= 4:
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = prod[corr_cols].corr().round(3)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 8},
                cbar_kws={"label": "Pearson r"},
                xticklabels=[c.replace("_"," ").title() for c in corr_cols],
                yticklabels=[c.replace("_"," ").title() for c in corr_cols])
    ax.set_title("Figure 11. Correlation Matrix of Numeric Product Features\n"
                 "(Pearson r; full lower triangle)",
                 fontsize=11, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/eda_fig11_correlation.png", dpi=150)
    plt.close()
    print("→ Figure 11 saved: eda_fig11_correlation.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("EDA COMPLETE")
print(f"{'='*60}")
print(f"Products analysed     : {len(prod):,}")
print(f"Reviews analysed      : {len(rev):,}")
print(f"Category breakdown:")
print(prod["category"].value_counts().to_string())
print(f"\nAll figures saved to: results/eda_*.png")
print(f"Tables saved to     : results/eda_*.csv")
print(f"\nNext: python 02_rq1_regression.py")
