"""
ewg_scraper.py  (v4 — open data replacement)
=============================================
EWG Skin Deep is blocked by Cloudflare and cannot be scraped.

Uses two FREE open databases instead — no login, no scraping, pure API:

  1. CosIng  — EU Cosmetics Ingredient Database
     https://ec.europa.eu/growth/tools-databases/cosing/
     Official INCI functions, EU restriction/ban status (Annex II-VI)
     28,000+ ingredients

  2. PubChem — NIH/NLM open chemistry database
     https://pubchem.ncbi.nlm.nih.gov/
     GHS hazard classifications (cancer, reproductive, sensitisation, etc.)
     100M+ compounds

These are MORE rigorous than EWG and fully citable in academic work.
The output schema is identical to the old EWG scraper — all downstream
scripts (00_clean_nykaa.py, RQ1-RQ4) work without any changes.

Run: python ewg_scraper.py
"""

import re, time, os, random, requests
from urllib.parse import quote
from tqdm import tqdm
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

COSING_S  = requests.Session()
PUBCHEM_S = requests.Session()
PUBCHEM_S.headers.update({"Accept": "application/json"})


def extract_all_ingredients():
    nykaa_file = "data/raw/nykaa_products.csv"
    if not os.path.exists(nykaa_file):
        print("ERROR: data/raw/nykaa_products.csv not found.")
        return []
    df   = pd.read_csv(nykaa_file)
    rows = df["ingredients"].dropna().tolist()
    print(f"Parsing ingredients from {len(rows):,} products...")
    SENTENCE_WORDS = {
        "use","apply","avoid","keep","store","wash","rinse","contact","eyes",
        "external","only","children","reach","consult","doctor","dermatologist",
        "tested","patch","test","discontinue","result","stop","using","contains",
        "certified","organic","formula","product","ingredient","made","india",
        "imported","manufactured","distributed","marketed","registered","trademark",
        "suitable","recommended","direction","instruction","warning","caution",
        "note","important","please","before","after","during","first","second",
        "third","step","percent","weight","volume","quantity","amount",
    }
    unique = set()
    for ing_str in rows:
        for part in re.split(r",", str(ing_str)):
            c = re.sub(r"[\[\]\(\)\*\+\d%°©®™]", "", part)
            c = re.sub(r"\s+", " ", c).strip().lower()
            c = re.sub(r'^[\-\'\"\._\s]+', '', c).strip()
            c = re.sub(r'[\-\'\"\._\s]+$', '', c).strip()
            if len(c) < 3 or len(c) > 60:               continue
            if not re.search(r"[a-z]", c):               continue
            if re.search(r"[:;=&@#/\\<>]", c):           continue
            if re.search(r"\.\s", c) or c.endswith("."): continue
            if len(set(c.split()) & SENTENCE_WORDS) >= 2: continue
            if re.search(r"^\d|\.com|www\.", c):          continue
            if len(c.split()) > 6:                        continue
            unique.add(c)
    result = sorted(unique)
    print(f"Unique ingredients: {len(result):,}")
    return result


def lookup_cosing(name):
    """Query CosIng EU API for official INCI functions and restriction status."""
    try:
        r = COSING_S.get(
            "https://ec.europa.eu/growth/tools-databases/cosing/rest/cosing/ingredients/search",
            params={"name": name, "status": "all", "pageSize": 3},
            timeout=10
        )
        if r.status_code != 200: return {}
        data  = r.json()
        items = data.get("ingredients") or data.get("results") or data.get("data") or []
        if not items: return {}
        item  = items[0]
        funcs = item.get("functions") or item.get("function") or []
        if isinstance(funcs, list):
            fnames = [f.get("name","") if isinstance(f,dict) else str(f) for f in funcs]
        else:
            fnames = [str(funcs)]
        annexes    = item.get("annexes") or item.get("restrictions") or []
        ann_str    = " ".join([str(a) for a in (annexes if isinstance(annexes,list) else [annexes])]).lower()
        return {
            "cosing_inci"      : item.get("inciName") or item.get("name") or name,
            "cosing_functions" : "|".join(f for f in fnames if f),
            "cosing_restricted": int(bool(annexes)),
            "cosing_banned"    : int("ii" in ann_str or "prohibit" in ann_str),
            "cosing_cas"       : item.get("casNumber") or item.get("cas") or "",
        }
    except: return {}


def lookup_pubchem(name, cas=""):
    """Query PubChem for GHS hazard classifications."""
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    cid  = None
    for query in ([f"{base}/compound/name/{quote(cas)}/cids/JSON"] if cas else []) + \
                 [f"{base}/compound/name/{quote(name)}/cids/JSON"]:
        try:
            r = PUBCHEM_S.get(query, timeout=10)
            if r.status_code == 200:
                cids = r.json().get("IdentifierList",{}).get("CID",[])
                if cids: cid = cids[0]; break
        except: pass
        time.sleep(0.2)

    if not cid: return {}

    # Fetch GHS hazard data
    hazards = []
    try:
        r2 = PUBCHEM_S.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification",
            timeout=15
        )
        if r2.status_code == 200:
            for sec in r2.json().get("Record",{}).get("Section",[]):
                for subsec in sec.get("Section",[]):
                    for info in subsec.get("Information",[]):
                        for sv in info.get("Value",{}).get("StringWithMarkup",[]):
                            t = sv.get("String","")
                            if t: hazards.append(t)
    except: pass

    ht = " ".join(hazards).lower()

    cancer   = int(bool(re.search(r"h35[01]|carcinogen|category\s*1[ab]?", ht)))
    devel    = int(bool(re.search(r"h36[01]|reproductive|developmental toxicit", ht)))
    allergy  = int(bool(re.search(r"h31[47]|h334|sensitiz|allergen", ht)))
    irritat  = int(bool(re.search(r"h31[456]|skin irritat|eye irritat", ht)))
    endocrin = int(bool(re.search(r"endocrin|hormone disrupt|estrogenic", ht)))
    organ    = int(bool(re.search(r"h37[01]|organ toxicit|systemic toxic", ht)))
    neuro    = int(bool(re.search(r"neurotox|nervous system", ht)))

    score = None
    if cancer or devel:    score = 8.0 if "category 1" in ht else 6.0
    elif allergy or organ: score = 5.0
    elif irritat:          score = 3.0
    elif ht.strip():       score = 2.0

    return {
        "pubchem_cid"          : cid,
        "ghs_hazards_raw"      : " | ".join(hazards[:5]),
        "concern_cancer"       : cancer,
        "concern_developmental": devel,
        "concern_allergy"      : allergy,
        "concern_irritation"   : irritat,
        "concern_endocrine"    : endocrin,
        "concern_organ_tox"    : organ,
        "concern_neurotoxicity": neuro,
        "derived_hazard_score" : score,
    }


def fallback_functions(name):
    """Keyword-based function classification when CosIng has no result."""
    n = name.lower()
    tags = []
    if any(k in n for k in ["glycerin","glycerol","sorbitol","hyaluronic","sodium pca",
                              "urea","propanediol","butylene glycol","trehalose","inositol"]):
        tags.append("humectant")
    if any(k in n for k in ["dimethicone","silicone","cyclomethicone","petrolatum","mineral oil",
                              "lanolin","shea","jojoba","squalane","ceramide","caprylic",
                              "isopropyl myristate","cetyl","stearyl","beeswax"]):
        tags.append("emollient")
    if any(k in n for k in ["glycolic","salicylic","lactic","mandelic","azelaic","tartaric",
                              "malic","citric acid","retinol","retinoic","gluconolactone"]):
        tags.append("exfoliant")
    if any(k in n for k in ["phenoxyethanol","paraben","methylisothiazolinone","benzoate",
                              "sorbate","dehydroacetic","chlorphenesin","dmdm","bronopol"]):
        tags.append("preservative")
    if any(k in n for k in ["tocopherol","ascorbic","vitamin c","vitamin e","resveratrol",
                              "ferulic","niacinamide","green tea","astaxanthin","coenzyme"]):
        tags.append("antioxidant")
    if any(k in n for k in ["sulfate","sulfonate","betaine","glucoside","polysorbate",
                              "laureth","cocamide","sles","sls"]):
        tags.append("surfactant")
    if any(k in n for k in ["zinc oxide","titanium dioxide","avobenzone","oxybenzone",
                              "octinoxate","octocrylene","homosalate","octisalate"]):
        tags.append("uv-filter")
    if any(k in n for k in ["fragrance","parfum","linalool","limonene","geraniol","citral",
                              "eugenol","coumarin","musk"]):
        tags.append("fragrance")
    if any(k in n for k in ["carbomer","xanthan","cellulose","acrylate","carrageenan",
                              "gelatin","guar","locust","pectin"]):
        tags.append("viscosity-agent")
    if any(k in n for k in ["centella","allantoin","panthenol","bisabolol","aloe","chamomile",
                              "calendula","madecassoside","asiaticoside"]):
        tags.append("soothing")
    if any(k in n for k in ["peptide","tripeptide","tetrapeptide","hexapeptide","palmitoyl"]):
        tags.append("peptide")
    if any(k in n for k in ["sodium hydroxide","triethanolamine","citric acid","lactic acid",
                              "potassium hydroxide","arginine"]):
        tags.append("ph-adjuster")
    return "|".join(tags)


def scrape_ingredient(name):
    cosing  = lookup_cosing(name)
    time.sleep(random.uniform(0.2, 0.5))
    pubchem = lookup_pubchem(name, cas=cosing.get("cosing_cas",""))
    time.sleep(random.uniform(0.2, 0.5))

    func_tags = cosing.get("cosing_functions","") or fallback_functions(name)
    score     = pubchem.get("derived_hazard_score")

    concerns = [k for k,flag in [
        ("cancer",        pubchem.get("concern_cancer",0)),
        ("developmental", pubchem.get("concern_developmental",0)),
        ("allergy",       pubchem.get("concern_allergy",0)),
        ("endocrine",     pubchem.get("concern_endocrine",0)),
        ("organ_toxicity",pubchem.get("concern_organ_tox",0)),
        ("irritation",    pubchem.get("concern_irritation",0)),
        ("restricted",    cosing.get("cosing_banned",0)),
        ("neurotoxicity", pubchem.get("concern_neurotoxicity",0)),
    ] if flag]

    return {
        "query_name"            : name,
        "inci_name"             : cosing.get("cosing_inci") or name,
        "ewg_hazard_score"      : score,
        "ewg_hazard_label"      : ("low"      if score is not None and score<=2 else
                                   "moderate" if score is not None and score<=6 else
                                   "high"     if score is not None else "unknown"),
        "ewg_concerns_raw"      : "|".join(concerns),
        "concern_cancer"        : pubchem.get("concern_cancer",0),
        "concern_developmental" : pubchem.get("concern_developmental",0),
        "concern_allergy"       : pubchem.get("concern_allergy",0),
        "concern_endocrine"     : pubchem.get("concern_endocrine",0),
        "concern_organ_tox"     : pubchem.get("concern_organ_tox",0),
        "concern_irritation"    : pubchem.get("concern_irritation",0),
        "concern_restricted"    : int(cosing.get("cosing_banned",0) or cosing.get("cosing_restricted",0)),
        "concern_contamination" : 0,
        "concern_neurotoxicity" : pubchem.get("concern_neurotoxicity",0),
        "data_availability"     : "good" if pubchem.get("pubchem_cid") else "limited",
        "function_tags"         : func_tags,
        "cosing_functions"      : cosing.get("cosing_functions",""),
        "pubchem_cid"           : str(pubchem.get("pubchem_cid","")),
        "ghs_hazards_raw"       : pubchem.get("ghs_hazards_raw",""),
        "source_url"            : f"https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem.get('pubchem_cid','')}" if pubchem.get("pubchem_cid") else "",
        "source"                : "cosing+pubchem",
    }


def run():
    print("=" * 60)
    print("INGREDIENT SAFETY LOOKUP — CosIng (EU) + PubChem (NIH)")
    print("=" * 60)

    ING_FILE  = "data/raw/ewg_ingredients.csv"
    CONC_FILE = "data/raw/ewg_ingredient_concerns.csv"

    all_ingredients = extract_all_ingredients()
    if not all_ingredients: return

    # Resume — skip already done, ignore old ewg_not_found placeholders
    done = set()
    if os.path.exists(ING_FILE):
        ex   = pd.read_csv(ING_FILE)
        real = ex[ex["source"] == "cosing+pubchem"]
        done = set(real["query_name"].str.lower().tolist())
        if len(done) > 0:
            print(f"Resuming — {len(done):,} already scraped")
        # Rewrite file without old not_found garbage
        real.to_csv(ING_FILE, index=False)

    todo = [i for i in all_ingredients if i not in done]
    print(f"\nIngredients to look up : {len(todo):,}")
    print(f"Estimated runtime      : {len(todo)*1.2/3600:.1f}–{len(todo)*2/3600:.1f} hours")
    print(f"No browser needed — pure API calls, no Cloudflare issues\n")

    for name in tqdm(todo, desc="Ingredients"):
        try:
            rec = scrape_ingredient(name)
            pd.DataFrame([rec]).to_csv(ING_FILE, mode="a",
                header=not os.path.exists(ING_FILE), index=False)
            done.add(name)

            if rec["ewg_concerns_raw"] and rec["ewg_concerns_raw"] != "":
                c_rows = [{"inci_name": rec["inci_name"], "query_name": name,
                            "concern_name": c, "source": "cosing+pubchem"}
                           for c in rec["ewg_concerns_raw"].split("|") if c]
                if c_rows:
                    pd.DataFrame(c_rows).to_csv(CONC_FILE, mode="a",
                        header=not os.path.exists(CONC_FILE), index=False)

            if rec["ewg_hazard_label"] != "unknown" or rec["function_tags"]:
                score_str = f"{rec['ewg_hazard_score']:.1f}" if rec["ewg_hazard_score"] else "?"
                tqdm.write(f"  ✓ {rec['inci_name'][:35]:35s} | "
                           f"score={score_str:4s} | funcs: {rec['function_tags'][:35]}")

        except Exception as e:
            tqdm.write(f"  ✗ {name}: {e}")

        time.sleep(random.uniform(0.8, 1.2))

    print(f"\n{'='*60}")
    if os.path.exists(ING_FILE):
        df    = pd.read_csv(ING_FILE)
        found = df[df["source"]=="cosing+pubchem"]
        print(f"Total scraped        : {len(df):,}")
        print(f"With PubChem data    : {(found['pubchem_cid'].astype(str).str.len()>0).sum():,}")
        print(f"With CosIng functions: {(found['cosing_functions'].astype(str).str.len()>0).sum():,}")
        if len(found):
            print(f"\nHazard distribution:")
            print(found["ewg_hazard_label"].value_counts().to_string())
    print(f"\n✓ Done. Next: python incidecoder_scraper.py")


if __name__ == "__main__":
    run()
