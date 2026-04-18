"""
incidecoder_scraper.py
======================
Scrapes ingredient function and safety data from INCIDecoder.com.

Strategy — same as EWG scraper:
  Step 1: Extract every unique ingredient from data/raw/nykaa_products.csv
  Step 2: For each ingredient, search INCIDecoder and scrape its detail page:
          https://incidecoder.com/search?query=INGREDIENT_NAME
          Then navigate to: https://incidecoder.com/ingredients/SLUG
          Extract:
            - Canonical INCI name
            - Function tags (what-it-does: humectant, emollient, etc.)
            - Irritancy level (low / medium / high)
            - Comedogenicity score (0-5 Fulton scale)
            - Skin type suitability
            - Brief description
  Step 3: Save incrementally — resume-safe

Outputs:
  data/raw/incidecoder_ingredients.csv   one row per ingredient

Runtime: ~5-12 hours for ~2,000-5,000 ingredients
Run: python incidecoder_scraper.py
"""

import re, time, os, random, subprocess
from urllib.parse import quote_plus
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

os.makedirs("data/raw", exist_ok=True)
BASE = "https://incidecoder.com"


def extract_all_ingredients():
    """
    Parse every product ingredient list from nykaa_products.csv.
    Returns sorted list of ALL unique lowercase INCI ingredient names
    that pass text quality filters — no frequency cutoff.
    """
    nykaa_file = "data/raw/nykaa_products.csv"
    if not os.path.exists(nykaa_file):
        print("ERROR: data/raw/nykaa_products.csv not found.")
        print("       Run nykaa_full_scraper.py first.")
        return []

    df   = pd.read_csv(nykaa_file)
    rows = df["ingredients"].dropna().tolist()
    print(f"Parsing ingredients from {len(rows):,} products...")

    SENTENCE_WORDS = {
        "use", "apply", "avoid", "keep", "store", "wash", "rinse",
        "contact", "eyes", "external", "only", "children", "reach",
        "consult", "doctor", "dermatologist", "tested", "patch",
        "test", "discontinue", "result", "stop", "using", "contains",
        "certified", "organic", "formula", "product", "ingredient",
        "made", "india", "imported", "manufactured", "distributed",
        "marketed", "registered", "trademark", "suitable",
        "recommended", "direction", "instruction", "warning",
        "caution", "note", "important", "please", "before", "after",
        "during", "first", "second", "third", "step", "percent",
        "weight", "volume", "quantity", "amount",
    }

    unique = set()
    for ing_str in rows:
        parts = re.split(r",", str(ing_str))
        for part in parts:
            cleaned = re.sub(r"[\[\]\(\)\*\+\d%°©®™]", "", part)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            cleaned = re.sub(r'^[-\'"._\s]+', '', cleaned).strip()
            cleaned = re.sub(r'[-\'"._\s]+$', '', cleaned).strip()
            if len(cleaned) < 3 or len(cleaned) > 60:
                continue
            if not re.search(r"[a-z]", cleaned):
                continue
            if re.search(r"[:;=&@#/\\<>]", cleaned):
                continue
            if re.search(r"\.\s", cleaned) or cleaned.endswith("."):
                continue
            if len(set(cleaned.split()) & SENTENCE_WORDS) >= 2:
                continue
            if re.search(r"^\d", cleaned):
                continue
            if re.search(r"\.com|www\.", cleaned):
                continue
            if len(cleaned.split()) > 6:
                continue
            unique.add(cleaned)

    result = sorted(unique)
    print(f"Unique ingredients extracted: {len(result):,}")
    return result


def build_driver():
    v = None
    for cmd in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "google-chrome",
    ]:
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
            v = r.stdout.strip().split()[-1]; break
        except: pass

    import shutil
    p = shutil.which("chromedriver")
    if not p:
        from webdriver_manager.chrome import ChromeDriverManager
        wdm = ChromeDriverManager(driver_version=v).install() if v else ChromeDriverManager().install()
        p   = wdm if os.access(wdm, os.X_OK) else None
        if not p:
            for root, _, files in os.walk(os.path.dirname(os.path.dirname(wdm))):
                for f in files:
                    c = os.path.join(root, f)
                    if f == "chromedriver" and os.access(c, os.X_OK):
                        p = c; break
                if p: break

    try: subprocess.run(["xattr", "-cr", p], capture_output=True, timeout=5)
    except: pass

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.add_argument("--log-level=3")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    d = webdriver.Chrome(service=Service(p), options=opts)
    d.execute_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    return d


# ── INCIDecoder-specific parsers ──────────────────────────────────────────────

# Known irritancy classifications (supplement page scraping)
KNOWN_IRRITANTS = {
    "high": [
        "fragrance", "parfum", "alcohol denat", "sodium lauryl sulfate",
        "methylisothiazolinone", "methylchloroisothiazolinone", "dmdm hydantoin",
        "formaldehyde", "linalool", "limonene", "geraniol", "citral", "eugenol",
        "isoeugenol", "cinnamal", "benzyl benzoate", "menthol", "camphor",
        "propylparaben", "butylparaben", "oxybenzone",
    ],
    "medium": [
        "benzyl alcohol", "phenoxyethanol", "chlorphenesin", "imidazolidinyl urea",
        "diazolidinyl urea", "sodium benzoate", "dehydroacetic acid",
        "cocamidopropyl betaine", "sodium laureth sulfate", "triethanolamine",
        "potassium hydroxide", "ammonium hydroxide", "isopropyl alcohol",
    ],
}

# Known comedogenicity scores (Fulton scale 0-5)
KNOWN_COMEDOGENS = {
    5: ["isopropyl myristate", "wheat germ oil", "linseed oil"],
    4: ["isopropyl palmitate", "isopropyl isostearate", "butyl stearate",
        "decyl oleate", "cocoa butter", "coconut oil", "palm oil",
        "octyl stearate", "acetylated lanolin"],
    3: ["cotton seed oil", "soybean oil", "lanolin alcohol", "isostearyl neopentanoate",
        "myristyl myristate", "octyl palmitate", "cetearyl alcohol"],
    2: ["shea butter", "jojoba oil", "almond oil", "avocado oil",
        "glyceryl stearate", "beeswax"],
    1: ["argan oil", "rosehip oil", "marula oil", "mineral oil", "dimethicone"],
}


def classify_irritancy(name):
    n = name.lower()
    for level in ["high", "medium"]:
        if any(k in n for k in KNOWN_IRRITANTS[level]):
            return level
    return "low"


def classify_comedogenicity(name):
    n = name.lower()
    for score in sorted(KNOWN_COMEDOGENS.keys(), reverse=True):
        if any(k in n for k in KNOWN_COMEDOGENS[score]):
            return score
    return 0


def parse_inci_page(soup, page_text, name):
    """Extract function tags, irritancy, comedogenicity from INCIDecoder page."""
    t = page_text.lower()

    # ── Function tags ─────────────────────────────────────────────────────────
    # INCIDecoder displays these as "what-it-does" tags
    functions = []
    func_patterns = {
        "humectant"    : ["humectant"],
        "emollient"    : ["emollient"],
        "skin-conditioning": ["skin conditioning", "skin conditioner"],
        "exfoliant"    : ["exfoliant", "exfoliating agent", "keratolytic"],
        "preservative" : ["preservative"],
        "antioxidant"  : ["antioxidant"],
        "surfactant"   : ["surfactant", "cleansing", "foaming"],
        "uv-filter"    : ["uv filter", "sunscreen", "uva", "uvb"],
        "emulsifier"   : ["emulsifier", "emulsifying"],
        "film-forming" : ["film-forming", "film former"],
        "fragrance"    : ["fragrance", "parfum", "masking", "scent"],
        "pH-adjuster"  : ["ph adjuster", "buffering", "neutralizer"],
        "occlusive"    : ["occlusive"],
        "chelating"    : ["chelating", "sequestering"],
        "soothing"     : ["soothing", "anti-irritant", "calming"],
        "brightening"  : ["brightening", "skin brightening", "lightening"],
        "anti-aging"   : ["anti-aging", "anti-wrinkle", "collagen stimulat"],
        "peptide"      : ["peptide", "tripeptide", "tetrapeptide"],
        "vitamin"      : ["vitamin", "ascorbic", "retinol", "tocopherol", "niacinamide"],
    }
    # Try to get from HTML tags/badges first
    for sel in ["[class*='what']", "[class*='function']", "[class*='tag']",
                "[class*='badge']", "[class*='category']", ".ingredient-function"]:
        for el in soup.select(sel):
            txt = el.get_text(strip=True).lower()
            if 2 < len(txt) < 40:
                for tag, kws in func_patterns.items():
                    if any(kw in txt for kw in kws):
                        if tag not in functions:
                            functions.append(tag)

    # Fallback: scan full page text
    for tag, kws in func_patterns.items():
        if tag not in functions:
            if any(kw in t for kw in kws):
                functions.append(tag)

    # ── Skin type suitability ─────────────────────────────────────────────────
    skin_suits = []
    if any(k in t for k in ["oily skin", "oil control", "mattif", "pore", "acne"]): skin_suits.append("oily")
    if any(k in t for k in ["dry skin", "hydrat", "moistur", "nourish"]): skin_suits.append("dry")
    if any(k in t for k in ["sensitive", "gentle", "sooth", "calm", "anti-irritant"]): skin_suits.append("sensitive")
    if any(k in t for k in ["all skin", "all types", "suitable for all"]): skin_suits.append("all")
    if not skin_suits: skin_suits.append("all")

    # ── Description ──────────────────────────────────────────────────────────
    description = ""
    for sel in ["p", ".description", "[class*='description']", "[class*='about']", "[class*='intro']"]:
        for el in soup.select(sel):
            txt = el.get_text(strip=True)
            if 30 < len(txt) < 500:
                description = txt; break
        if description: break

    # ── Irritancy ─────────────────────────────────────────────────────────────
    # INCIDecoder shows irritancy as a 0-5 number, same format as comedogenicity.
    # Displayed as "irr." value on ingredient page.
    irritancy_score = None

    # Method 1: dedicated irritancy HTML element
    for sel in ["[class*='irrit']","[class*='irritan']",".irrit",".irr-score"]:
        for el in soup.select(sel):
            m = re.search(r"(\d+(?:\.\d+)?)", el.get_text())
            if m:
                v = float(m.group(1))
                if 0 <= v <= 5:
                    irritancy_score = v; break

    # Method 2: "irr." text pattern — first number before com.
    if irritancy_score is None:
        for pat in [
            r"irr\.?\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"irritan[^\d\n]{0,20}?(\d+(?:\.\d+)?)",
        ]:
            m2 = re.search(pat, t, re.I)
            if m2:
                v = float(m2.group(1))
                if 0 <= v <= 5:
                    irritancy_score = v; break

    # Convert numeric score to label
    if irritancy_score is not None:
        irritancy = "high" if irritancy_score >= 3 else "medium" if irritancy_score >= 1 else "low"
    else:
        # Fallback to keyword classification
        irritancy = classify_irritancy(name)
        if "high irritancy" in t or "strong irritant" in t:  irritancy = "high"
        elif "moderate irritan" in t:                         irritancy = "medium" if irritancy != "high" else "high"

    # ── Comedogenicity ────────────────────────────────────────────────────────
    # INCIDecoder shows comedogenicity as a 0-5 number on the ingredient page.
    # It appears in several possible locations in the HTML:
    #   1. A span/div with class containing "comedogen" or "com"
    #   2. In a table cell near the text "com." or "comedogenic"
    #   3. As text pattern "X irr. / Y com." or "irr. X, com. Y"
    comedogen_score = classify_comedogenicity(name)  # fallback

    # Method 1: dedicated comedogenicity element
    for sel in ["[class*='comedo']", "[class*='comedogen']", "[id*='comedo']",
                ".comedo", ".comedogenic-score", ".com-score"]:
        els = soup.select(sel)
        for el in els:
            m = re.search(r"(\d+(?:\.\d+)?)", el.get_text())
            if m:
                v = float(m.group(1))
                if 0 <= v <= 5:
                    comedogen_score = int(round(v)); break

    # Method 2: look for "irr., com." pattern in page HTML
    # INCIDecoder renders it as e.g. "0, 3" or "1 | 3" after "irr., com."
    if comedogen_score == classify_comedogenicity(name):  # still at fallback
        # Pattern: irritancy then comedogenicity, separated by comma or pipe
        for pat in [
            r"irr\.?,?\s*com\.?\s*[:\-]?\s*\d+[,\s/|]+\s*(\d+)",
            r"irritan[^\d]*?(\d)\s*[,/|]\s*(\d)\s*com",
            r"com(?:edogen)?[^\d]*?(\d+(?:\.\d+)?)",
            r"(\d)\s*irr.*?(\d)\s*com",
        ]:
            m2 = re.search(pat, t, re.I)
            if m2:
                groups = [g for g in m2.groups() if g is not None]
                v = float(groups[-1])  # last group = comedogenicity
                if 0 <= v <= 5:
                    comedogen_score = int(round(v)); break

    # Method 3: look in table cells near "com" text
    if comedogen_score == classify_comedogenicity(name):
        for td in soup.find_all(["td", "th", "span", "div"]):
            if "com" in td.get_text().lower()[:10]:
                # Check next sibling or adjacent cell for number
                nxt = td.find_next_sibling()
                if nxt:
                    m3 = re.search(r"(\d+(?:\.\d+)?)", nxt.get_text())
                    if m3:
                        v = float(m3.group(1))
                        if 0 <= v <= 5:
                            comedogen_score = int(round(v)); break

    return {
        "functions"       : "|".join(functions) if functions else "",
        "irritancy_level" : irritancy,
        "comedogen_score" : comedogen_score,
        "skin_suitability": "|".join(skin_suits),
        "description"     : description[:300],
    }


def scrape_ingredient(driver, name):
    """
    Search INCIDecoder for one ingredient, navigate to its page, extract data.
    Returns dict or None if not found.
    """
    search_url = f"{BASE}/search?query={quote_plus(name)}"
    try:
        driver.get(search_url)
        time.sleep(random.uniform(2.5, 4.0))
    except Exception as e:
        return {"_error": str(e)}

    driver.execute_script("window.scrollTo(0, 400);")
    time.sleep(0.8)

    soup = BeautifulSoup(driver.page_source, "lxml")

    # Find first ingredient link: /ingredients/SLUG
    ingredient_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href: continue
        if not href.startswith("http"):
            href = BASE + href
        if "/ingredients/" in href and "search" not in href:
            ingredient_url = href; break

    if not ingredient_url:
        return None

    try:
        driver.get(ingredient_url)
        time.sleep(random.uniform(2.5, 4.0))
        for _ in range(4):
            driver.execute_script("window.scrollBy(0, 600);")
            time.sleep(0.4)
    except Exception as e:
        return {"_error": str(e)}

    detail_soup = BeautifulSoup(driver.page_source, "lxml")
    page_text   = detail_soup.get_text(" ")

    # Canonical name from h1
    inci_name = name
    h1 = detail_soup.find("h1")
    if h1:
        candidate = h1.get_text(strip=True)
        if 2 < len(candidate) < 120:
            inci_name = candidate

    parsed = parse_inci_page(detail_soup, page_text, name)

    return {
        "query_name"      : name,
        "inci_name"       : inci_name,
        "functions"       : parsed["functions"],
        "irritancy_level" : parsed["irritancy_level"],
        "comedogen_score" : parsed["comedogen_score"],
        "skin_suitability": parsed["skin_suitability"],
        "description"     : parsed["description"],
        "source_url"      : ingredient_url,
        "source"          : "incidecoder",
    }


def run():
    print("=" * 60)
    print("INCIDecoder — FULL INGREDIENT SCRAPER")
    print("=" * 60)

    OUT_FILE = "data/raw/incidecoder_ingredients.csv"

    # Step 1: Get all unique ingredients from Nykaa data
    all_ingredients = extract_all_ingredients()
    if not all_ingredients:
        return

    # Resume support
    done = set()
    if os.path.exists(OUT_FILE):
        ex = pd.read_csv(OUT_FILE)
        if "query_name" in ex.columns:
            done = set(ex["query_name"].dropna().str.lower().tolist())
            print(f"Resuming — {len(done):,} done, {len(all_ingredients)-len(done):,} remaining")
        elif "ingredient_name" in ex.columns:
            done = set(ex["ingredient_name"].dropna().str.lower().tolist())
            print(f"Resuming — {len(done):,} done (legacy column name)")
        else:
            print(f"Existing file columns: {list(ex.columns)} — backing up and starting fresh")
            os.rename(OUT_FILE, OUT_FILE + ".bak")

    todo = [i for i in all_ingredients if i not in done]
    est_low  = len(todo) * 3 / 3600
    est_high = len(todo) * 6 / 3600
    print(f"\nIngredients to scrape : {len(todo):,}")
    print(f"Estimated runtime     : {est_low:.1f}–{est_high:.1f} hours")
    print(f"Safe to Ctrl+C and restart — resumes automatically\n")

    driver      = build_driver()
    restart_ctr = 0
    not_found   = []
    items_this_session = 0

    try:
        for name in tqdm(todo, desc="INCIDecoder"):
            items_this_session += 1
            if items_this_session > 1 and items_this_session % 40 == 0:
                tqdm.write("  Restarting driver (memory management)...")
                try: driver.quit()
                except: pass
                time.sleep(3)
                driver = build_driver()

            try:
                rec = scrape_ingredient(driver, name)

                if rec is None:
                    not_found.append(name)
                    # Save placeholder using fallback classification
                    placeholder = {
                        "query_name"      : name,
                        "inci_name"       : name,
                        "functions"       : "",
                        "irritancy_level" : classify_irritancy(name),
                        "comedogen_score" : classify_comedogenicity(name),
                        "skin_suitability": "all",
                        "description"     : "",
                        "source_url"      : "",
                        "source"          : "incidecoder_notfound_fallback",
                    }
                    pd.DataFrame([placeholder]).to_csv(OUT_FILE, mode="a",
                        header=not os.path.exists(OUT_FILE), index=False)
                    done.add(name)

                elif "_error" in rec:
                    tqdm.write(f"  ✗ error [{name}]: {rec['_error']}")
                    # Don't mark done — retry on restart

                else:
                    pd.DataFrame([rec]).to_csv(OUT_FILE, mode="a",
                        header=not os.path.exists(OUT_FILE), index=False)
                    done.add(name)

                    tqdm.write(
                        f"  ✓ {rec['inci_name'][:35]:35s} | "
                        f"funcs: {rec['functions'][:35]:35s} | "
                        f"irrit={rec['irritancy_level']:6s} comedo={rec['comedogen_score']}"
                    )

            except Exception as e:
                tqdm.write(f"  ✗ {name}: {e}")

            time.sleep(random.uniform(2.0, 3.5))

    finally:
        try: driver.quit()
        except: pass

    # Summary
    print(f"\n{'='*60}")
    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        found = df[df["source"] == "incidecoder"]
        print(f"Total scraped     : {len(df):,}")
        print(f"Found on INCIdec  : {len(found):,}")
        print(f"Not found (fallback): {len(df) - len(found):,}")
        if len(found):
            print(f"\nTop function tags:")
            all_funcs = "|".join(found["functions"].dropna()).split("|")
            from collections import Counter
            print(Counter(f for f in all_funcs if f).most_common(15))
            print(f"\nIrritancy distribution:")
            print(found["irritancy_level"].value_counts().to_string())
            print(f"\nComedogenicity distribution:")
            print(found["comedogen_score"].value_counts().sort_index().to_string())

    print(f"\n✓ INCIDecoder scraping complete.")
    print(f"Next: python 00_clean_nykaa.py")


if __name__ == "__main__":
    run()
