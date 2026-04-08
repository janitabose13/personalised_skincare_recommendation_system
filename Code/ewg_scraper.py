"""
ewg_scraper.py
==============
Scrapes ingredient-level safety data from EWG Skin Deep.

Strategy — "all possible ingredients" means every unique ingredient
found in the scraped Nykaa product data. This scraper:

  Step 1: Parses every product's INCI ingredient list from
          data/raw/nykaa_products.csv → deduplicated unique ingredient names

  Step 2: For each unique ingredient, searches EWG Skin Deep via Selenium:
          https://www.ewg.org/skindeep/search/?search=INGREDIENT_NAME
          Navigates to the ingredient detail page and extracts:
            - Hazard score (1-10)
            - Hazard label (low / moderate / high)
            - Concern flags (cancer, developmental, allergy, endocrine, etc.)
            - Data availability (none / limited / fair / good / robust)
            - Function tags (humectant, emollient, etc.)

  Step 3: Saves incrementally — safe to stop and restart at any time

Outputs:
  data/raw/ewg_ingredients.csv          one row per ingredient
  data/raw/ewg_ingredient_concerns.csv  one row per concern per ingredient

Runtime: ~4-10 hours for ~2,000-5,000 unique ingredients
Run: python ewg_scraper.py
"""

import re, json, time, os, random, subprocess
from urllib.parse import quote_plus
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

os.makedirs("data/raw", exist_ok=True)
BASE = "https://www.ewg.org"


def extract_all_ingredients():
    """
    Parse every product ingredient list from nykaa_products.csv.
    Returns sorted list of unique lowercase ingredient names.
    """
    nykaa_file = "data/raw/nykaa_products.csv"
    if not os.path.exists(nykaa_file):
        print("ERROR: data/raw/nykaa_products.csv not found.")
        print("       Run nykaa_full_scraper.py first.")
        return []

    df   = pd.read_csv(nykaa_file)
    rows = df["ingredients"].dropna().tolist()
    print(f"Parsing ingredients from {len(rows):,} products...")

    unique = set()
    for ing_str in rows:
        parts = re.split(r",", str(ing_str))
        for part in parts:
            # Strip numbers, brackets, percent signs, asterisks
            cleaned = re.sub(r"[\[\]\(\)\*\+\d%°]", "", part)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            # Skip garbage
            if len(cleaned) < 3 or len(cleaned) > 80:
                continue
            if re.search(r"[:;=&]", cleaned):
                continue
            if any(w in cleaned for w in [
                "may contain", "contains", "ingredient", "formula",
                "free from", "does not", "without", "certified",
            ]):
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


def parse_hazard_score(soup, text):
    # Method 1: JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            for key in ["ratingValue", "score", "hazardScore", "hazard_score"]:
                s = data.get(key)
                if s:
                    v = float(s)
                    if 1 <= v <= 10: return v
        except: pass

    # Method 2: CSS elements
    for sel in [
        "[class*='score']", "[class*='hazard']", "[class*='rating']",
        "[class*='Score']", "[class*='Hazard']", "[data-score]",
    ]:
        for el in soup.select(sel):
            txt = el.get_text(strip=True)
            m   = re.match(r"^(\d+(?:\.\d+)?)$", txt)
            if m:
                v = float(m.group(1))
                if 1 <= v <= 10: return v

    # Method 3: Regex over page text
    for pat in [
        r"hazard\s+score[:\s]+(\d+(?:\.\d+)?)",
        r"ewg\s+score[:\s]+(\d+(?:\.\d+)?)",
        r"overall\s+hazard[:\s]+(\d+(?:\.\d+)?)",
        r"rated\s+(\d+(?:\.\d+)?)\s+(?:out\s+of\s+)?10",
        r'"score"\s*:\s*(\d+(?:\.\d+)?)',
        r'"ratingValue"\s*:\s*"(\d+(?:\.\d+)?)"',
    ]:
        m = re.search(pat, text, re.I)
        if m:
            v = float(m.group(1))
            if 1 <= v <= 10: return v

    return None


def parse_concerns(text):
    t = text.lower()
    concern_map = {
        "cancer"         : ["cancer", "carcinogen", "carcinogenic", "tumor"],
        "developmental"  : ["developmental", "reproductive toxicit", "birth defect", "teratogen", "fertility"],
        "allergy"        : ["allerg", "immunotox", "contact sensitiz", "skin sensitiz", "contact dermatit"],
        "endocrine"      : ["endocrin", "hormone disrupt", "estrogenic", "androgenic", "thyroid disrupt"],
        "organ_toxicity" : ["organ toxicit", "systemic toxic", "hepatotox", "nephrotox"],
        "irritation"     : ["skin irritat", "eye irritat", "irritat", "mucous membrane"],
        "restricted"     : ["restrict", "prohibited", "banned", "not permitted", "prop 65", "eu prohibited"],
        "contamination"  : ["contaminat", "impurit", "nitrosamine", "1,4-dioxane", "formaldehyde releas"],
        "neurotoxicity"  : ["neurotox", "nervous system harm", "brain"],
    }
    found = []
    for key, kws in concern_map.items():
        if any(kw in t for kw in kws):
            found.append(key)
    return found


def parse_data_availability(text):
    t = text.lower()
    for label in ["robust", "good", "fair", "limited"]:
        if f"data availability: {label}" in t or f"data availability\n{label}" in t:
            return label
    # Fallback
    for label in ["robust", "good", "fair", "limited"]:
        if label in t: return label
    return "unknown"


def parse_function_tags(text):
    t = text.lower()
    tag_map = {
        "humectant"   : ["humectant", "moisture-binding", "moisture retention"],
        "emollient"   : ["emollient", "skin conditioning", "skin softening"],
        "exfoliant"   : ["exfoliant", "exfoliating", "keratolytic"],
        "preservative": ["preservative", "antimicrobial preserv"],
        "antioxidant" : ["antioxidant"],
        "surfactant"  : ["surfactant", "cleansing agent", "foaming agent"],
        "sunscreen"   : ["uv filter", "sunscreen active", "sun protection active"],
        "emulsifier"  : ["emulsifier", "emulsifying agent"],
        "fragrance"   : ["fragrance ingredient", "parfum component"],
        "occlusive"   : ["occlusive", "barrier former"],
        "ph_adjuster" : ["ph adjuster", "buffering agent", "neutralizer"],
    }
    found = []
    for tag, kws in tag_map.items():
        if any(kw in t for kw in kws):
            found.append(tag)
    return found


def scrape_ingredient(driver, name):
    """
    Search EWG for one ingredient, navigate to its detail page, extract data.
    Returns a dict, or None if ingredient not found on EWG.
    """
    search_url = f"{BASE}/skindeep/search/?search={quote_plus(name)}"
    try:
        driver.get(search_url)
        time.sleep(random.uniform(2.5, 4.0))
    except Exception as e:
        return {"_error": str(e)}

    driver.execute_script("window.scrollTo(0, 600);")
    time.sleep(0.8)

    soup = BeautifulSoup(driver.page_source, "lxml")

    # Find first ingredient detail page link
    # EWG ingredient URLs: /skindeep/ingredients/XXXXXX-name/
    ingredient_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href: continue
        if not href.startswith("http"):
            href = BASE + href
        if re.search(r"/skindeep/ingredients?/\d+", href, re.I):
            ingredient_url = href; break

    if not ingredient_url:
        return None

    try:
        driver.get(ingredient_url)
        time.sleep(random.uniform(2.5, 4.0))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
        time.sleep(0.8)
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

    score      = parse_hazard_score(detail_soup, page_text)
    concerns   = parse_concerns(page_text)
    data_avail = parse_data_availability(page_text)
    func_tags  = parse_function_tags(page_text)

    def flag(k): return int(k in concerns)

    return {
        "query_name"            : name,
        "inci_name"             : inci_name,
        "ewg_hazard_score"      : score,
        "ewg_hazard_label"      : (
            "low"      if score is not None and score <= 2 else
            "moderate" if score is not None and score <= 6 else
            "high"     if score is not None else "unknown"
        ),
        "ewg_concerns_raw"      : "|".join(concerns),
        "concern_cancer"        : flag("cancer"),
        "concern_developmental" : flag("developmental"),
        "concern_allergy"       : flag("allergy"),
        "concern_endocrine"     : flag("endocrine"),
        "concern_organ_tox"     : flag("organ_toxicity"),
        "concern_irritation"    : flag("irritation"),
        "concern_restricted"    : flag("restricted"),
        "concern_contamination" : flag("contamination"),
        "concern_neurotoxicity" : flag("neurotoxicity"),
        "data_availability"     : data_avail,
        "function_tags"         : "|".join(func_tags),
        "source_url"            : ingredient_url,
        "source"                : "ewg_skindeep",
    }


def run():
    print("=" * 60)
    print("EWG SKIN DEEP — FULL INGREDIENT SCRAPER")
    print("=" * 60)

    ING_FILE  = "data/raw/ewg_ingredients.csv"
    CONC_FILE = "data/raw/ewg_ingredient_concerns.csv"

    # Step 1: Extract all unique ingredients from Nykaa data
    all_ingredients = extract_all_ingredients()
    if not all_ingredients:
        return

    # Resume: skip already done
    done = set()
    if os.path.exists(ING_FILE):
        ex   = pd.read_csv(ING_FILE)
        done = set(ex["query_name"].str.lower().tolist())
        remaining = len(all_ingredients) - len(done)
        print(f"Resuming — {len(done):,} done, {remaining:,} remaining")

    todo = [i for i in all_ingredients if i not in done]
    est_low  = len(todo) * 3 / 3600
    est_high = len(todo) * 5 / 3600
    print(f"\nIngredients to scrape : {len(todo):,}")
    print(f"Estimated runtime     : {est_low:.1f}–{est_high:.1f} hours")
    print(f"Safe to Ctrl+C and restart — resumes automatically\n")

    driver      = build_driver()
    restart_ctr = 0
    not_found   = []

    try:
        for name in tqdm(todo, desc="EWG"):
            restart_ctr += 1
            if restart_ctr > 1 and restart_ctr % 40 == 0:
                tqdm.write("  Restarting driver (memory management)...")
                try: driver.quit()
                except: pass
                time.sleep(3)
                driver = build_driver()

            try:
                rec = scrape_ingredient(driver, name)

                if rec is None:
                    not_found.append(name)
                    placeholder = {
                        "query_name": name, "inci_name": name,
                        "ewg_hazard_score": None, "ewg_hazard_label": "not_found",
                        "ewg_concerns_raw": "", "concern_cancer": 0,
                        "concern_developmental": 0, "concern_allergy": 0,
                        "concern_endocrine": 0, "concern_organ_tox": 0,
                        "concern_irritation": 0, "concern_restricted": 0,
                        "concern_contamination": 0, "concern_neurotoxicity": 0,
                        "data_availability": "", "function_tags": "",
                        "source_url": "", "source": "ewg_not_found",
                    }
                    pd.DataFrame([placeholder]).to_csv(ING_FILE, mode="a",
                        header=not os.path.exists(ING_FILE), index=False)
                    done.add(name)

                elif "_error" in rec:
                    tqdm.write(f"  ✗ error [{name}]: {rec['_error']}")
                    # Don't mark done — will retry on restart

                else:
                    pd.DataFrame([rec]).to_csv(ING_FILE, mode="a",
                        header=not os.path.exists(ING_FILE), index=False)
                    done.add(name)

                    # Expand concerns into detail rows
                    if rec["ewg_concerns_raw"]:
                        c_rows = [{
                            "inci_name"   : rec["inci_name"],
                            "query_name"  : name,
                            "concern_name": c,
                            "source_url"  : rec["source_url"],
                        } for c in rec["ewg_concerns_raw"].split("|") if c]
                        if c_rows:
                            pd.DataFrame(c_rows).to_csv(CONC_FILE, mode="a",
                                header=not os.path.exists(CONC_FILE), index=False)

                    score_str = f"{rec['ewg_hazard_score']:.1f}" if rec["ewg_hazard_score"] else "?"
                    tqdm.write(
                        f"  ✓ {rec['inci_name'][:35]:35s} | "
                        f"score={score_str:4s} ({rec['ewg_hazard_label']:8s}) | "
                        f"concerns: {rec['ewg_concerns_raw'][:40]}"
                    )

            except Exception as e:
                tqdm.write(f"  ✗ {name}: {e}")

            time.sleep(random.uniform(2.0, 3.5))

    finally:
        try: driver.quit()
        except: pass

    # Summary
    print(f"\n{'='*60}")
    if os.path.exists(ING_FILE):
        df    = pd.read_csv(ING_FILE)
        found = df[df["ewg_hazard_label"] != "not_found"]
        print(f"Total scraped : {len(df):,}")
        print(f"Found on EWG  : {len(found):,}")
        print(f"Not found     : {len(df) - len(found):,}")
        if len(found):
            print(f"\nHazard distribution:")
            print(found["ewg_hazard_label"].value_counts().to_string())
            print(f"\nMean hazard score: {found['ewg_hazard_score'].mean():.2f}")
            print(f"\nTop high-hazard ingredients:")
            high = found[found["ewg_hazard_label"]=="high"].sort_values(
                "ewg_hazard_score", ascending=False).head(10)
            if len(high):
                print(high[["inci_name","ewg_hazard_score","ewg_concerns_raw"]].to_string(index=False))
    print(f"\n✓ EWG scraping complete.")
    print(f"Next: python incidecoder_scraper.py")


if __name__ == "__main__":
    run()
