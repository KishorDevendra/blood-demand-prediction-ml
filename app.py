from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier as RFC
import pickle
import math
import random
import os
import threading
import time
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
# OPTIONAL IMPORTS FOR ERAKTOSH LIVE FETCH
# ─────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPE_AVAILABLE = True
except ImportError:
    SCRAPE_AVAILABLE = False
    print("⚠️  requests/bs4 not installed – live eRaktosh fetch disabled.")

# ═══════════════════════════════════════════
# LOAD TRAINED BLOOD DEMAND MODEL
# ═══════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blood_demand_model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        blood_demand_model = pickle.load(f)
    print(f"✅ Loaded blood_demand_model.pkl  [{type(blood_demand_model).__name__}]")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Could not load model: {e}. Falling back to synthetic.")
    blood_demand_model = None
    MODEL_LOADED = False


def predict_demand(target_date: pd.Timestamp) -> float:
    if MODEL_LOADED and blood_demand_model is not None:
        features = pd.DataFrame([{
            "Month":      target_date.month,
            "DayOfWeek":  target_date.dayofweek,
            "WeekOfYear": int(target_date.isocalendar()[1]),
        }])
        try:
            if hasattr(blood_demand_model, "predict_proba"):
                proba  = blood_demand_model.predict_proba(features)[0]
                demand = float(5 + proba.max() * 35)
            else:
                demand = float(blood_demand_model.predict(features)[0])
        except Exception:
            demand = float(np.random.normal(15, 4))
    else:
        demand = float(np.random.normal(15, 4))
    return max(1.0, round(demand, 1))


# ═══════════════════════════════════════════
# BLOOD GROUPS & WEIGHTS
# ═══════════════════════════════════════════
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']

GROUP_MULTIPLIER = {
    'O+': 1.30, 'A+': 1.20, 'B+': 1.10, 'AB+': 0.85,
    'O-': 1.05, 'A-': 0.90, 'B-': 0.80, 'AB-': 0.70,
}

# ═══════════════════════════════════════════
# REAL DONOR DATA — MUMBAI ONLY
# ═══════════════════════════════════════════
DONOR_CSV = os.path.join(os.path.dirname(__file__), "blood_donation.csv")

try:
    raw_df = pd.read_csv(DONOR_CSV)
    print(f"✅ Loaded blood_donation.csv — {len(raw_df)} total rows")

    # ── STRICT MUMBAI FILTER ─────────────────────────────────────
    raw_df = raw_df[raw_df['City'].str.strip().str.lower() == 'mumbai'].reset_index(drop=True)
    print(f"✅ After Mumbai filter: {len(raw_df)} donors")

    donor_df = pd.DataFrame()
    donor_df['id']               = raw_df['Donor_ID'].astype(str)
    donor_df['name']             = raw_df['Full_Name'].fillna('Unknown')
    donor_df['blood_group']      = raw_df['Blood_Group']
    donor_df['age']              = pd.to_numeric(raw_df['Age'], errors='coerce').fillna(25).astype(int)
    donor_df['total_donations']  = pd.to_numeric(raw_df['Total_Donations'], errors='coerce').fillna(0).astype(int)
    donor_df['last_donation_dt'] = pd.to_datetime(raw_df['Last_Donation_Date'], dayfirst=True, errors='coerce')
    donor_df['phone']            = raw_df['Contact_Number'].astype(str)
    donor_df['eligible']         = raw_df['Eligible_for_Donation'].str.strip().str.lower() == 'yes'
    donor_df['city']             = 'Mumbai'
    donor_df['gender']           = raw_df.get('Gender', pd.Series([''] * len(raw_df))).fillna('')
    donor_df['medical']          = raw_df.get('Medical_Condition', pd.Series(['None'] * len(raw_df))).fillna('None')
    donor_df['center']           = raw_df.get('Donation_Center', pd.Series([''] * len(raw_df))).fillna('')
    donor_df['weight']           = pd.to_numeric(raw_df.get('Weight_kg',   pd.Series([np.nan]*len(raw_df))), errors='coerce').round(1)
    donor_df['hemoglobin']       = pd.to_numeric(raw_df.get('Hemoglobin_g_dL', pd.Series([np.nan]*len(raw_df))), errors='coerce').round(1)
    donor_df['last_donation']    = donor_df['last_donation_dt'].dt.strftime('%Y-%m-%d').fillna('N/A')
    donor_df = donor_df[donor_df['blood_group'].isin(BLOOD_GROUPS)].reset_index(drop=True)

except Exception as e:
    print(f"⚠️  CSV error: {e}. Empty donor list.")
    donor_df = pd.DataFrame(columns=[
        'id','name','blood_group','age','total_donations','last_donation',
        'last_donation_dt','phone','eligible','city','gender','medical',
        'center','weight','hemoglobin'
    ])
    donor_df['last_donation_dt'] = pd.NaT

# ═══════════════════════════════════════════
# RETENTION SCORE  (proper formula)
#
#  score = 70% × (donations / max_donations)     ← frequency
#        + 30% × (1 − days_since / 730)          ← recency  (2-year window)
#
#  Higher donations + donated recently = highest score.
#  Donors ranked by score descending → rank 1 = best donor to call first.
# ═══════════════════════════════════════════
TODAY   = pd.Timestamp.today().normalize()
MAX_DON = max(int(donor_df['total_donations'].max()), 1)

donor_df['days_since'] = (
    (TODAY - donor_df['last_donation_dt']).dt.days
    .fillna(730).clip(lower=0, upper=730)
)

donor_df['retention_score'] = (
    (donor_df['total_donations'] / MAX_DON) * 70
    + (1.0 - donor_df['days_since'] / 730) * 30
).clip(0, 100).round(1)

donor_df.sort_values('retention_score', ascending=False, inplace=True)
donor_df.reset_index(drop=True, inplace=True)
donor_df['rank'] = donor_df.index + 1

print(f"✅ Retention scores ready. Top: {donor_df.iloc[0]['name']} ({donor_df.iloc[0]['retention_score']})")

# ═══════════════════════════════════════════
# CURRENT STOCK
# ═══════════════════════════════════════════
_elig_counts = donor_df[donor_df['eligible']]['blood_group'].value_counts().to_dict()
current_stock = {bg: int(_elig_counts.get(bg, 0) // 5) + random.randint(1, 4) for bg in BLOOD_GROUPS}

# ═══════════════════════════════════════════
# MUMBAI BLOOD BANKS
# ═══════════════════════════════════════════
MUMBAI_BANKS = [
    {"name": "KEM Hospital Blood Bank",         "area": "Parel",          "phone": "022-24107000", "hours": "24/7"},
    {"name": "Nair Hospital Blood Bank",        "area": "Mumbai Central", "phone": "022-23027620", "hours": "24/7"},
    {"name": "Sion Hospital Blood Bank",        "area": "Sion",           "phone": "022-24076381", "hours": "24/7"},
    {"name": "Hinduja Hospital Blood Bank",     "area": "Mahim",          "phone": "022-24452222", "hours": "8AM-8PM"},
    {"name": "Lilavati Hospital Blood Bank",    "area": "Bandra",         "phone": "022-26751000", "hours": "24/7"},
    {"name": "Jaslok Hospital Blood Bank",      "area": "Pedder Road",    "phone": "022-66573333", "hours": "24/7"},
    {"name": "Bombay Hospital Blood Bank",      "area": "Marine Lines",   "phone": "022-22067676", "hours": "8AM-6PM"},
    {"name": "Kokilaben Hospital Blood Bank",   "area": "Andheri",        "phone": "022-42696969", "hours": "24/7"},
    {"name": "Global Hospital Blood Bank",      "area": "Parel",          "phone": "022-67670101", "hours": "24/7"},
    {"name": "Tata Memorial Blood Bank",        "area": "Parel",          "phone": "022-24177000", "hours": "8AM-5PM"},
    {"name": "Wockhardt Hospital Blood Bank",   "area": "Mira Road",      "phone": "022-28118989", "hours": "24/7"},
    {"name": "Holy Family Hospital Blood Bank", "area": "Bandra",         "phone": "022-26511111", "hours": "24/7"},
]

_eraktosh_cache = {"data": None, "fetched_at": None}
_CACHE_TTL      = 1800   # 30 minutes


def _fetch_eraktosh():
    """Scrape live blood availability from eRaktosh for Mumbai hospitals."""
    if not SCRAPE_AVAILABLE:
        return None
    try:
        url  = "https://eraktosh.nic.in/BB_Rpt.aspx"
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122",
            "Accept":     "text/html,application/xhtml+xml",
        })
        # GET page to harvest ASP.NET form tokens
        r = sess.get(url, timeout=12, verify=False)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        def _fv(name):
            el = soup.find("input", {"id": name})
            return el["value"] if el else ""

        payload = {
            "__VIEWSTATE":          _fv("__VIEWSTATE"),
            "__EVENTVALIDATION":    _fv("__EVENTVALIDATION"),
            "__VIEWSTATEGENERATOR": _fv("__VIEWSTATEGENERATOR"),
            "ctl00$ContentPlaceHolder1$ddlState": "27",
            "ctl00$ContentPlaceHolder1$btnSearch": "Search",
        }
        r2 = sess.post(url, data=payload, timeout=15, verify=False)
        r2.raise_for_status()
        soup2 = BeautifulSoup(r2.text, "html.parser")

        # Find results table
        table = soup2.find("table", {"id": "ctl00_ContentPlaceHolder1_GridView1"})
        if not table:
            for t in soup2.find_all("table"):
                hdrs = " ".join(th.get_text(strip=True).lower() for th in t.find_all("th"))
                if "blood bank" in hdrs or "a+" in hdrs:
                    table = t
                    break

        live = {}
        if table:
            def si(v):
                try: return max(0, int(str(v).replace(",","").strip()))
                except: return 0
            for row in table.find_all("tr")[1:]:
                cols = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cols) < 10:
                    continue
                city = cols[2].strip().lower() if len(cols) > 2 else ""
                if "mumbai" not in city and "bombay" not in city:
                    continue
                live[cols[0]] = {
                    "A+": si(cols[3]), "A-": si(cols[4]),
                    "B+": si(cols[5]), "B-": si(cols[6]),
                    "O+": si(cols[7]), "O-": si(cols[8]),
                    "AB+": si(cols[9]), "AB-": si(cols[10]) if len(cols) > 10 else 0,
                    "updated": datetime.now().strftime("%d %b %Y %H:%M"),
                }
        return live if live else None
    except Exception as ex:
        print(f"eRaktosh fetch error: {ex}")
        return None


def _get_banks():
    now = datetime.now()
    c   = _eraktosh_cache
    if c["data"] is None or (c["fetched_at"] and (now - c["fetched_at"]).seconds > _CACHE_TTL):
        live = _fetch_eraktosh()
        if live:
            c["data"] = live
            c["fetched_at"] = now

    out = []
    for bank in MUMBAI_BANKS:
        b = bank.copy()
        matched = False
        if c["data"]:
            for lname, avail in c["data"].items():
                kws = [w for w in bank["name"].lower().split() if len(w) > 4]
                if any(k in lname.lower() for k in kws):
                    b.update({bg: avail.get(bg, 0) for bg in BLOOD_GROUPS})
                    b["last_updated"] = avail.get("updated", "—")
                    b["source"] = "eRaktosh Live"
                    matched = True
                    break
        if not matched:
            rng = random.Random(abs(hash(bank["name"])) + int(time.time() // 3600))
            for bg in BLOOD_GROUPS:
                b[bg] = max(0, int(current_stock.get(bg, 5) * rng.uniform(0.3, 2.8)))
            b["last_updated"] = datetime.now().strftime("%d %b %Y %H:%M")
            b["source"] = "Simulated"
        out.append(b)
    return out


def _warm():
    time.sleep(3)
    live = _fetch_eraktosh()
    if live:
        _eraktosh_cache["data"] = live
        _eraktosh_cache["fetched_at"] = datetime.now()

threading.Thread(target=_warm, daemon=True).start()


def _clean(v):
    if isinstance(v, float) and math.isnan(v): return ""
    if isinstance(v, np.integer):               return int(v)
    if isinstance(v, np.floating):              return round(float(v), 2)
    return v

def _cr(row): return {k: _clean(v) for k, v in row.items()}


# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/dashboard")
def dashboard_data():
    tomorrow  = pd.Timestamp.today() + pd.Timedelta(days=1)
    forecasts, shortages = {}, []
    for bg in BLOOD_GROUPS:
        pred = round(predict_demand(tomorrow) * GROUP_MULTIPLIER.get(bg, 1.0), 1)
        forecasts[bg] = pred
        if pred > current_stock.get(bg, 0): shortages.append(bg)

    monthly_trend = []
    for i in range(6, 0, -1):
        d = pd.Timestamp.today() - pd.Timedelta(days=30*i)
        monthly_trend.append({"month": d.strftime("%b %Y"),
                               "units": int(predict_demand(d) * sum(GROUP_MULTIPLIER.values()))})

    bg_stats, wastage = [], {}
    for bg in BLOOD_GROUPS:
        s = current_stock.get(bg, 0)
        f = forecasts[bg]
        bg_stats.append({"group": bg, "stock": s, "forecast": f, "shortage": bg in shortages})
        wastage[bg] = round(max(0.0, s - f), 1)

    top5 = donor_df.nlargest(5, "retention_score")[
        ["name","blood_group","total_donations","last_donation","phone","eligible","retention_score","rank"]
    ].to_dict("records")

    return jsonify({
        "forecasts": forecasts, "shortages": shortages,
        "current_stock": current_stock, "monthly_trend": monthly_trend,
        "bg_stats": bg_stats, "top_donors": [_cr(d) for d in top5],
        "wastage": wastage,
        "total_donors": len(donor_df),
        "eligible_donors": int(donor_df["eligible"].sum()),
        "total_units_in_stock": sum(current_stock.values()),
        "model_active": MODEL_LOADED,
        "bg_distribution": donor_df["blood_group"].value_counts().to_dict(),
        "gender_dist": donor_df["gender"].value_counts().to_dict(),
        "medical_dist": donor_df["medical"].value_counts().head(6).to_dict(),
        "avg_donations": round(float(donor_df["total_donations"].mean()), 2),
        "max_donations": int(donor_df["total_donations"].max()),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data     = request.json or {}
    bg       = data.get("blood_group", "O+")
    stock    = int(data.get("stock", 0))
    date_str = data.get("date", "")
    td       = pd.Timestamp(date_str) if date_str else pd.Timestamp.today() + pd.Timedelta(days=1)
    base     = predict_demand(td)
    dp       = round(base * GROUP_MULTIPLIER.get(bg, 1.0), 1)
    shortage = dp > stock

    wf = [{"date": (pd.Timestamp.today()+pd.Timedelta(days=i)).strftime("%a %d %b"),
            "demand": round(predict_demand(pd.Timestamp.today()+pd.Timedelta(days=i))*GROUP_MULTIPLIER.get(bg,1.0), 1)}
          for i in range(1, 8)]

    return jsonify({
        "predicted_demand": dp, "current_stock": stock,
        "shortage": bool(shortage),
        "shortage_amount": round(max(0.0, dp-stock), 1),
        "excess": round(max(0.0, stock-dp), 1),
        "blood_group": bg,
        "model_used": "blood_demand_model.pkl" if MODEL_LOADED else "fallback_synthetic",
        "week_forecast": wf,
    })


@app.route("/api/donors")
def get_donors():
    bg    = request.args.get("blood_group", "all")
    elig  = request.args.get("eligible_only", "false") == "true"
    srch  = request.args.get("search", "").strip().lower()
    page  = int(request.args.get("page", 1))
    pp    = int(request.args.get("per_page", 50))

    mask = np.ones(len(donor_df), dtype=bool)
    if bg != "all":   mask &= (donor_df["blood_group"].values == bg)
    if elig:          mask &= (donor_df["eligible"].values == True)
    if srch:          mask &= donor_df["name"].str.lower().str.contains(srch, na=False).values

    idxs  = np.where(mask)[0]
    total = len(idxs)
    paged = idxs[(page-1)*pp : page*pp]
    cols  = [c for c in ["id","name","blood_group","age","total_donations","last_donation",
                          "phone","eligible","retention_score","rank","gender","medical",
                          "city","weight","hemoglobin"] if c in donor_df.columns]
    recs  = [_cr(r) for r in donor_df.iloc[paged][cols].to_dict("records")]

    return jsonify({"donors": recs, "total": total, "page": page,
                    "per_page": pp, "pages": max(1, math.ceil(total/pp))})


@app.route("/api/blood_banks")
def get_blood_banks():
    bg    = request.args.get("blood_group", "")
    hrs   = request.args.get("hours", "")
    banks = _get_banks()
    if hrs == "24/7": banks = [b for b in banks if b.get("hours") == "24/7"]
    if bg and bg in BLOOD_GROUPS:
        banks = sorted(banks, key=lambda b: b.get(bg, 0), reverse=True)
    return jsonify(banks)


@app.route("/api/blood_banks/refresh")
def refresh_banks():
    live = _fetch_eraktosh()
    if live:
        _eraktosh_cache["data"] = live
        _eraktosh_cache["fetched_at"] = datetime.now()
        return jsonify({"status": "live", "banks": len(live)})
    return jsonify({"status": "unavailable", "banks": 0})


@app.route("/api/wastage_hospitals")
def wastage_hospitals():
    """Returns redistribution plan: which hospitals can give/receive per blood group."""
    tomorrow  = pd.Timestamp.today() + pd.Timedelta(days=1)
    forecasts, shortages = {}, []
    for bg in BLOOD_GROUPS:
        pred = round(predict_demand(tomorrow) * GROUP_MULTIPLIER.get(bg, 1.0), 1)
        forecasts[bg] = pred
        if pred > current_stock.get(bg, 0): shortages.append(bg)

    surplus_groups = {
        bg: round(current_stock.get(bg, 0) - forecasts[bg], 1)
        for bg in BLOOD_GROUPS if current_stock.get(bg, 0) > forecasts[bg]
    }

    banks = _get_banks()

    recommendations = []
    for sbg in shortages:
        need = round(forecasts[sbg] - current_stock.get(sbg, 0), 1)
        givers = sorted(
            [{"name": b["name"], "area": b["area"], "phone": b["phone"], "units": b.get(sbg, 0)}
             for b in banks if b.get(sbg, 0) >= 8],
            key=lambda x: x["units"], reverse=True
        )[:4]
        receivers = sorted(
            [{"name": b["name"], "area": b["area"], "phone": b["phone"], "units": b.get(sbg, 0)}
             for b in banks],
            key=lambda x: x["units"]
        )[:3]
        recommendations.append({
            "blood_group": sbg, "need_units": need,
            "donor_hospitals": givers, "receiver_hospitals": receivers,
        })

    surplus_detail = []
    for bg, excess in surplus_groups.items():
        top = sorted(
            [{"name": b["name"], "area": b["area"], "phone": b["phone"], "units": b.get(bg, 0)}
             for b in banks], key=lambda x: x["units"], reverse=True
        )[:4]
        surplus_detail.append({
            "blood_group": bg, "excess_units": excess,
            "stock": current_stock.get(bg, 0), "forecast": forecasts[bg],
            "top_holders": top,
        })

    total_excess   = sum(surplus_groups.values())
    saveable       = min(total_excess, sum(r["need_units"] for r in recommendations))

    return jsonify({
        "recommendations": recommendations,
        "surplus_detail":  surplus_detail,
        "shortages":       shortages,
        "total_excess":    round(total_excess, 1),
        "saveable_units":  round(saveable, 1),
    })


@app.route("/api/model_info")
def model_info():
    if MODEL_LOADED and blood_demand_model is not None:
        info = {"status": "loaded", "model_type": type(blood_demand_model).__name__,
                "n_features": int(blood_demand_model.n_features_in_),
                "features": ["Month", "DayOfWeek", "WeekOfYear"]}
        try: info["classes"] = blood_demand_model.classes_.tolist()
        except: pass
    else:
        info = {"status": "not_loaded"}
    return jsonify(info)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)