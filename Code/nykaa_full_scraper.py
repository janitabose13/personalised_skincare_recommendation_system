"""
nykaa_full_scraper.py
=====================
Scrapes all skincare products from Nykaa category pages.

Confirmed working categories:
  Moisturizer : https://www.nykaa.com/skin/moisturizers/c/8393   — 112 pages, 2235 products
  Serum       : https://www.nykaa.com/skin/serums/c/73006         — 88 pages,  1746 products
  Cleanser    : https://www.nykaa.com/skin/cleansers/c/8378       — 105 pages, 2100 products

Each page uses ?page_no=N param.
Products scraped from window.__PRELOADED_STATE__.productPage.product
Reviews from /gateway-api/products/{id}/reviews API

Outputs:
  data/raw/nykaa_products.csv   (appended incrementally)
  data/raw/nykaa_reviews.csv    (appended incrementally)
"""

import re, json, time, os, subprocess, random, requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import numpy as np

os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs",        exist_ok=True)

BASE = "https://www.nykaa.com"

# ── Exact category configs from your URLs ────────────────────────────────────
CATEGORIES = [
    {
        "name"     : "Moisturizer",
        "base_url" : f"{BASE}/skin/moisturizers/c/8393",
        "max_pages": 112,
    },
    {
        "name"     : "Serum",
        "base_url" : f"{BASE}/skin/serums/c/73006",
        "max_pages": 88,
    },
    {
        "name"     : "Cleanser",
        "base_url" : f"{BASE}/skin/cleansers/c/8378",
        "max_pages": 105,
    },
]

MAX_REVIEWS = 500   # per product (25 pages × 20 reviews)

REVIEW_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; ARM Mac OS X 13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Accept"    : "application/json",
    "Referer"   : "https://www.nykaa.com/",
}

SKIN_KW = {
    "oily"       : ["oily","oil control","mattif","pore","sebum","acne"],
    "dry"        : ["dry skin","hydrat","moistur","nourish","ceramide","barrier"],
    "combination": ["combination","t-zone"],
    "sensitive"  : ["sensitive","gentle","calm","fragrance-free","hypoaller"],
    "normal"     : ["all skin","normal skin","suitable for all"],
}
KEY_ACTIVES = ["niacinamide","retinol","vitamin c","ascorbic acid","hyaluronic acid",
               "sodium hyaluronate","salicylic acid","glycolic acid","lactic acid",
               "ceramide","peptide","zinc oxide","titanium dioxide","bakuchiol",
               "squalane","centella","kojic acid","azelaic acid","tranexamic acid"]
IRRITANTS   = ["fragrance","parfum","alcohol denat","methylisothiazolinone","linalool",
               "limonene","geraniol","menthol","benzyl alcohol"]
COMEDOGENS  = ["coconut oil","isopropyl myristate","isopropyl palmitate","wheat germ oil",
               "cocoa butter","palm oil","soybean oil"]


def build_driver():
    v = None
    for cmd in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome","google-chrome"]:
        try:
            r = subprocess.run([cmd,"--version"],capture_output=True,text=True,timeout=5)
            v = r.stdout.strip().split()[-1]; break
        except: pass
    print(f"Chrome: {v}")

    import shutil
    p = shutil.which("chromedriver")
    if p:
        try:
            r2 = subprocess.run([p,"--version"],capture_output=True,text=True,timeout=5)
            if r2.returncode != 0: p = None
        except: p = None

    if not p:
        from webdriver_manager.chrome import ChromeDriverManager
        wdm = ChromeDriverManager(driver_version=v).install() if v else ChromeDriverManager().install()
        if os.access(wdm, os.X_OK):
            p = wdm
        else:
            for root,_,files in os.walk(os.path.dirname(os.path.dirname(wdm))):
                for f in files:
                    c = os.path.join(root,f)
                    if f=="chromedriver" and os.access(c,os.X_OK): p=c; break
                if p: break

    try: subprocess.run(["xattr","-cr",p], capture_output=True, timeout=5)
    except: pass
    print(f"ChromeDriver: {p}")

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1280,800")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.add_argument("--disable-background-networking")
    opts.add_argument("--log-level=3")
    opts.add_argument("user-agent=Mozilla/5.0 (Macintosh; ARM Mac OS X 13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
    opts.add_experimental_option("excludeSwitches",["enable-automation"])
    opts.add_experimental_option("useAutomationExtension",False)
    d = webdriver.Chrome(service=Service(p), options=opts)
    d.execute_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    return d


def get_page_urls(driver, base_url, page_no):
    """Get all product URLs from a single category page."""
    # Nykaa uses page_no param — append correctly
    sep = "&" if "?" in base_url else "?"
    url = f"{base_url}{sep}page_no={page_no}&ptype=lst"
    driver.get(url)
    time.sleep(4)

    # Scroll fully to trigger lazy-loaded product cards
    last_h = 0
    for _ in range(12):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.7)
        h = driver.execute_script("return document.body.scrollHeight")
        if h == last_h:
            break
        last_h = h

    # Also check __PRELOADED_STATE__ for product list (faster than HTML parsing)
    try:
        state_prods = driver.execute_script("""
            var s = window.__PRELOADED_STATE__;
            if (!s) return null;
            var cl = s.categoryListing || s.searchListingPage;
            if (!cl) return null;
            // Try different structures
            var prods = [];
            if (cl.products) prods = cl.products;
            else if (cl.data && cl.data.products) prods = cl.data.products;
            else if (cl.productList) prods = cl.productList;
            return prods.length > 0 ? prods : null;
        """)
        if state_prods:
            urls = set()
            for p in state_prods:
                pid  = str(p.get("id") or p.get("productId") or "")
                slug = p.get("slug") or p.get("url") or ""
                if pid:
                    if slug and not slug.startswith("http"):
                        slug = f"{BASE}/{slug.lstrip('/')}"
                    url_full = f"{slug}?productId={pid}" if slug else f"{BASE}/p/{pid}?productId={pid}"
                    urls.add((url_full, pid))
            if urls:
                return list(urls)
    except:
        pass

    # Fallback: parse HTML
    soup = BeautifulSoup(driver.page_source, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"(/[^?#]+/p/(\d+))", a["href"])
        if m:
            slug = m.group(1)
            pid  = m.group(2)
            urls.add((f"{BASE}{slug}?productId={pid}", pid))
    return list(urls)


def scrape_product(driver, url, pid, category):
    """Load product page and extract all fields."""
    driver.get(url)
    prod = None
    for _ in range(12):
        time.sleep(1)
        prod = driver.execute_script(
            "var s=window.__PRELOADED_STATE__;"
            "return s&&s.productPage&&s.productPage.product||null"
        )
        if prod: break

    if not prod:
        return None

    name  = (prod.get("name") or prod.get("title") or "").strip()
    if not name: return None

    brand = prod.get("brandName","") or ""
    mrp   = prod.get("mrp")
    offer = prod.get("offerPrice")
    price = offer or mrp
    try:    price = float(price)
    except: price = None
    try:    mrp   = float(mrp)
    except: mrp   = None

    rating = prod.get("rating")
    try:    rating = round(float(rating),1)
    except: rating = None

    try:    rating_count = int(prod.get("ratingCount") or 0)
    except: rating_count = None
    try:    review_count = int(prod.get("reviewCount") or 0)
    except: review_count = None

    ings = prod.get("ingredients") or ""
    if not ings:
        desc = re.sub(r"<[^>]+>"," ", str(prod.get("description","")))
        m2   = re.search(r"(?:key ingredient|ingredient)[s]?[:\s]+(.{20,600})",desc,re.I)
        if m2: ings = m2.group(1).strip()
    ings = re.sub(r"<[^>]+>"," ", str(ings)).strip() if ings else ""
    il   = ings.lower()

    ing_count     = len([x for x in ings.split(",") if x.strip()]) if ings else 0
    key_actives   = "|".join([a for a in KEY_ACTIVES if a in il])
    irritant_cnt  = sum(1 for i in IRRITANTS  if i in il)
    comedogen_cnt = sum(1 for c in COMEDOGENS if c in il)

    combined   = f"{name} {ings} {re.sub(r'<[^>]+>','',str(prod.get('description','')))}".lower()
    skin_types = [s for s,kws in SKIN_KW.items() if any(k in combined for k in kws)]

    LUX  = {"forest essentials","kama ayurveda","guerlain","tatcha","la mer","clinique","estee lauder","shiseido","lancome","dior","chanel","sulwhasoo"}
    DRUG = {"cetaphil","cerave","neutrogena","nivea","himalaya","biotique","ponds","lakme","garnier","mamaearth","plum","minimalist","dot & key","the ordinary","wow","mama earth","mcaffeine","re'equil","derma co","acne star"}
    bl   = brand.lower()
    bt   = "Luxury" if any(x in bl for x in LUX) else "Drugstore" if any(x in bl for x in DRUG) else "Prestige"

    def ptier(p):
        if p is None: return "Unknown"
        if p < 500:   return "Budget"
        if p <= 1500: return "Mid-Range"
        return "Luxury"

    return {
        "product_id"       : str(prod.get("id","") or pid),
        "product_name"     : name,
        "brand"            : brand,
        "brand_tier"       : bt,
        "category"         : category,
        "price_inr"        : price,
        "mrp_inr"          : mrp,
        "discount_pct"     : prod.get("discount"),
        "price_tier"       : ptier(price),
        "rating"           : rating,
        "num_ratings"      : rating_count,
        "num_reviews"      : review_count,
        "in_stock"         : prod.get("inStock"),
        "skin_type_tags"   : "|".join(skin_types) if skin_types else "all",
        "skin_oily"        : int("oily" in skin_types),
        "skin_dry"         : int("dry" in skin_types),
        "skin_combination" : int("combination" in skin_types),
        "skin_sensitive"   : int("sensitive" in skin_types),
        "skin_normal"      : int("normal" in skin_types),
        "ingredients"      : ings,
        "ingredient_count" : ing_count,
        "key_actives"      : key_actives,
        "num_actives"      : len([a for a in key_actives.split("|") if a]),
        "irritant_count"   : irritant_cnt,
        "comedogen_count"  : comedogen_cnt,
        "has_fragrance"    : int("fragrance" in il or "parfum" in il),
        "is_fragrance_free": int("fragrance" not in il and "parfum" not in il),
        "source_url"       : url,
    }


def fetch_reviews(product_id, max_reviews=MAX_REVIEWS):
    """Fetch real reviews from Nykaa review API."""
    reviews = []; page = 1
    while len(reviews) < max_reviews:
        url = f"{BASE}/gateway-api/products/{product_id}/reviews?domain=nykaa&size=20&source=react&page={page}"
        try:
            r = requests.get(url, headers=REVIEW_HEADERS, timeout=10)
            if r.status_code != 200: break
            batch = r.json().get("response",{}).get("reviewData",[])
            if not batch: break
            reviews.extend(batch); page += 1
            time.sleep(random.uniform(0.3, 0.7))
        except: break
    return reviews[:max_reviews]


def infer_skin_from_text(text):
    t = str(text).lower()
    if any(k in t for k in ["oily","oil","acne","pore","sebum"]):      return "oily"
    if any(k in t for k in ["dry","hydrat","moistur","flak","tight"]): return "dry"
    if any(k in t for k in ["combination","combo","t-zone"]):          return "combination"
    if any(k in t for k in ["sensitive","reacti","irritat","redness"]): return "sensitive"
    return "normal"


# ── Load existing data to resume ──────────────────────────────────────────────
PROD_FILE   = "data/raw/nykaa_products.csv"
REV_FILE    = "data/raw/nykaa_reviews.csv"

existing_pids = set()
if os.path.exists(PROD_FILE):
    try:
        ex = pd.read_csv(PROD_FILE)
        existing_pids = set(ex["product_id"].astype(str).tolist())
        print(f"Resuming — {len(existing_pids)} products already scraped")
    except: pass

# ── Main loop ─────────────────────────────────────────────────────────────────
print("="*60)
print("NYKAA FULL SCRAPER")
print("="*60)

driver       = build_driver()
restart_count= 0

try:
    for cat in CATEGORIES:
        cat_name  = cat["name"]
        base_url  = cat["base_url"]
        max_pages = cat["max_pages"]

        print(f"\n{'='*60}")
        print(f"Category: {cat_name}  ({max_pages} pages)")
        print(f"{'='*60}")

        for page_no in range(1, max_pages + 1):
            print(f"\n  Page {page_no}/{max_pages}")

            # Restart driver every 50 pages to prevent memory buildup
            restart_count += 1
            if restart_count > 1 and restart_count % 50 == 0:
                print("  Restarting driver (memory management)...")
                try: driver.quit()
                except: pass
                time.sleep(3)
                driver = build_driver()

            try:
                page_items = get_page_urls(driver, base_url, page_no)
            except Exception as e:
                print(f"  Page {page_no} failed: {e}")
                continue

            if not page_items:
                print(f"  Page {page_no}: 0 products found (already scraped or load issue) — continuing")
                continue

            new_items = [(u,p) for u,p in page_items if p not in existing_pids]
            print(f"  Found {len(page_items)} products ({len(new_items)} new)")

            for url, pid in tqdm(new_items, desc=f"p{page_no}"):
                if pid in existing_pids: continue

                try:
                    rec = scrape_product(driver, url, pid, cat_name)
                    if rec:
                        # Save product immediately
                        prod_row = pd.DataFrame([rec])
                        prod_row.to_csv(PROD_FILE, mode="a",
                                       header=not os.path.exists(PROD_FILE),
                                       index=False)
                        existing_pids.add(pid)

                        # Fetch + save reviews immediately
                        reviews = fetch_reviews(pid)
                        if reviews:
                            rev_rows = []
                            for rv in reviews:
                                if rv.get("rating"):
                                    txt = str(rv.get("description","")) + " " + str(rv.get("title",""))
                                    rev_rows.append({
                                        "product_id"  : pid,
                                        "product_name": rec["product_name"],
                                        "brand"       : rec["brand"],
                                        "category"    : cat_name,
                                        "user_id"     : rv.get("encryptedUserId","") or rv.get("name",""),
                                        "rating"      : float(rv["rating"]),
                                        "review_title": rv.get("title",""),
                                        "review_text" : rv.get("description",""),
                                        "is_buyer"    : rv.get("isBuyer",False),
                                        "skin_type"   : infer_skin_from_text(txt),
                                    })
                            if rev_rows:
                                pd.DataFrame(rev_rows).to_csv(
                                    REV_FILE, mode="a",
                                    header=not os.path.exists(REV_FILE),
                                    index=False)

                        tqdm.write(f"  ✓ {rec['product_name'][:50]} | {rec['brand']} | ₹{rec['price_inr']} | ★{rec['rating']} ({rec['num_ratings']})")
                        if reviews: tqdm.write(f"    → {len(reviews)} reviews")

                except Exception as e:
                    tqdm.write(f"  ✗ {url[:60]}: {e}")

                time.sleep(random.uniform(1.0, 2.0))

            time.sleep(random.uniform(1.0, 2.0))

finally:
    try: driver.quit()
    except: pass

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
if os.path.exists(PROD_FILE):
    df = pd.read_csv(PROD_FILE)
    print(f"Total products: {len(df)}")
    print(df["category"].value_counts().to_string())
    print(f"\nRating stats:")
    print(df["rating"].describe().round(2).to_string())

if os.path.exists(REV_FILE):
    rv = pd.read_csv(REV_FILE)
    print(f"\nTotal reviews: {len(rv)}")
    print(f"Unique users:  {rv['user_id'].nunique()}")

print(f"\nNext step: python 00_clean_nykaa.py")
