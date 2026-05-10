"""
Food Donation Prediction Model
================================
Predicts number of people expected and food required for a given shelter and date.
 
Uses the exact same data and logic as the HTML widget:
  - Per-shelter, per-month historical averages + std devs (600 records, 2022-2024)
  - Weekend uplift  : +21.9%  (derived from dataset)
  - Festival uplift : +66.4%  (derived from dataset)
  - Food per person : per-shelter avg kg/person (~0.21-0.22 kg)
 
Usage:
    python food_donation_predictor.py
 
No external libraries needed beyond the Python standard library.
"""
 
from datetime import date, datetime
 
# ─── SHELTER DATA ─────────────────────────────────────────────────────────────
# Source: food_donation_dataset_v2.csv  (600 records, 2022–2024)
# fpp : average food-per-person in kg
# m   : month (1–12) -> { 'u': mean people served, 's': std dev }
 
SHELTER_DATA = {
    "Aashraya Shelter": {
        "fpp": 0.2183,
        "m": {
            1:  {"u": 923.7,  "s": 917.1},   2:  {"u": 657.7,  "s": 421.0},
            3:  {"u": 692.2,  "s": 608.6},   4:  {"u": 1077.0, "s": 468.2},
            5:  {"u": 474.0,  "s": 282.8},   6:  {"u": 711.5,  "s": 414.9},
            7:  {"u": 685.2,  "s": 435.0},   8:  {"u": 1021.6, "s": 669.0},
            9:  {"u": 1013.4, "s": 779.3},   10: {"u": 1563.8, "s": 391.7},
            11: {"u": 836.6,  "s": 536.9},   12: {"u": 1174.2, "s": 967.6},
        },
    },
    "Annapurna Shelter": {
        "fpp": 0.2105,
        "m": {
            1:  {"u": 712.5,  "s": 681.8},   2:  {"u": 975.3,  "s": 356.0},
            3:  {"u": 1179.4, "s": 642.1},   4:  {"u": 2750.0, "s": 1650.0},
            5:  {"u": 1185.0, "s": 811.3},   6:  {"u": 770.3,  "s": 333.6},
            7:  {"u": 1283.5, "s": 713.8},   8:  {"u": 946.2,  "s": 453.4},
            9:  {"u": 1557.0, "s": 934.2},   10: {"u": 905.8,  "s": 610.3},
            11: {"u": 890.3,  "s": 96.2},    12: {"u": 959.9,  "s": 737.5},
        },
    },
    "Asha Griha": {
        "fpp": 0.2149,
        "m": {
            1:  {"u": 1897.8, "s": 728.2},   2:  {"u": 1314.3, "s": 807.9},
            3:  {"u": 735.1,  "s": 444.6},   4:  {"u": 1174.0, "s": 923.6},
            5:  {"u": 666.5,  "s": 354.5},   6:  {"u": 714.2,  "s": 812.2},
            7:  {"u": 516.0,  "s": 316.2},   8:  {"u": 1139.5, "s": 553.4},
            9:  {"u": 715.9,  "s": 299.0},   10: {"u": 1437.7, "s": 555.6},
            11: {"u": 1295.2, "s": 739.5},   12: {"u": 1081.0, "s": 315.1},
        },
    },
    "Daya Niwas": {
        "fpp": 0.2205,
        "m": {
            1:  {"u": 1680.8, "s": 1409.9},  2:  {"u": 785.6,  "s": 419.8},
            3:  {"u": 479.0,  "s": 350.7},   4:  {"u": 1647.2, "s": 1127.4},
            5:  {"u": 595.0,  "s": 357.0},   6:  {"u": 838.0,  "s": 251.9},
            7:  {"u": 808.7,  "s": 567.8},   8:  {"u": 716.3,  "s": 719.3},
            9:  {"u": 940.6,  "s": 653.4},   10: {"u": 1474.5, "s": 799.2},
            11: {"u": 1061.0, "s": 672.7},   12: {"u": 583.2,  "s": 404.7},
        },
    },
    "Jeevan Kendra": {
        "fpp": 0.2173,
        "m": {
            1:  {"u": 1740.7, "s": 1771.3},  2:  {"u": 1137.0, "s": 780.4},
            3:  {"u": 1088.0, "s": 596.4},   4:  {"u": 1269.1, "s": 949.3},
            5:  {"u": 787.3,  "s": 562.7},   6:  {"u": 489.0,  "s": 358.9},
            7:  {"u": 701.3,  "s": 540.2},   8:  {"u": 1090.6, "s": 801.2},
            9:  {"u": 891.0,  "s": 567.1},   10: {"u": 1409.5, "s": 1118.4},
            11: {"u": 1379.1, "s": 451.2},   12: {"u": 1058.2, "s": 595.6},
        },
    },
    "Karuna Ashram": {
        "fpp": 0.2180,
        "m": {
            1:  {"u": 1773.6, "s": 1461.1},  2:  {"u": 1484.3, "s": 513.2},
            3:  {"u": 1220.3, "s": 1039.0},  4:  {"u": 839.8,  "s": 503.3},
            5:  {"u": 534.9,  "s": 295.3},   6:  {"u": 724.4,  "s": 510.7},
            7:  {"u": 665.0,  "s": 737.2},   8:  {"u": 1268.0, "s": 988.7},
            9:  {"u": 1078.0, "s": 667.9},   10: {"u": 817.0,  "s": 472.6},
            11: {"u": 1417.0, "s": 802.6},   12: {"u": 1755.6, "s": 1012.6},
        },
    },
    "Prerna Niwas": {
        "fpp": 0.2186,
        "m": {
            1:  {"u": 1330.1, "s": 936.5},   2:  {"u": 1223.7, "s": 543.8},
            3:  {"u": 1087.5, "s": 741.7},   4:  {"u": 934.6,  "s": 407.8},
            5:  {"u": 849.4,  "s": 458.8},   6:  {"u": 924.0,  "s": 547.7},
            7:  {"u": 549.5,  "s": 460.7},   8:  {"u": 979.0,  "s": 575.7},
            9:  {"u": 1130.3, "s": 1177.3},  10: {"u": 1429.5, "s": 638.1},
            11: {"u": 1515.3, "s": 1064.1},  12: {"u": 1311.1, "s": 560.9},
        },
    },
    "Sahara Home": {
        "fpp": 0.2195,
        "m": {
            1:  {"u": 601.4,  "s": 359.1},   2:  {"u": 945.1,  "s": 480.1},
            3:  {"u": 1184.5, "s": 814.8},   4:  {"u": 1006.8, "s": 591.2},
            5:  {"u": 749.4,  "s": 479.3},   6:  {"u": 1105.8, "s": 187.4},
            7:  {"u": 496.0,  "s": 357.6},   8:  {"u": 421.0,  "s": 252.6},
            9:  {"u": 821.4,  "s": 673.9},   10: {"u": 1178.5, "s": 623.7},
            11: {"u": 1363.7, "s": 968.5},   12: {"u": 1529.3, "s": 934.0},
        },
    },
    "Seva Sadan": {
        "fpp": 0.2136,
        "m": {
            1:  {"u": 1357.7, "s": 1219.8},  2:  {"u": 1882.8, "s": 1105.7},
            3:  {"u": 1145.8, "s": 1217.5},  4:  {"u": 520.3,  "s": 310.0},
            5:  {"u": 822.0,  "s": 493.2},   6:  {"u": 490.6,  "s": 427.0},
            7:  {"u": 574.8,  "s": 172.7},   8:  {"u": 959.0,  "s": 512.0},
            9:  {"u": 1024.8, "s": 597.3},   10: {"u": 1172.0, "s": 378.0},
            11: {"u": 808.8,  "s": 620.4},   12: {"u": 743.7,  "s": 624.7},
        },
    },
    "Umeed Niwas": {
        "fpp": 0.2145,
        "m": {
            1:  {"u": 1665.6, "s": 1093.8},  2:  {"u": 1266.3, "s": 914.3},
            3:  {"u": 355.0,  "s": 213.0},   4:  {"u": 1641.2, "s": 967.3},
            5:  {"u": 851.9,  "s": 731.9},   6:  {"u": 1454.4, "s": 478.6},
            7:  {"u": 669.0,  "s": 401.4},   8:  {"u": 1500.8, "s": 1172.2},
            9:  {"u": 1214.2, "s": 661.0},   10: {"u": 686.2,  "s": 297.8},
            11: {"u": 928.5,  "s": 785.6},   12: {"u": 1474.0, "s": 1027.3},
        },
    },
}
 
# ─── MULTIPLIERS ──────────────────────────────────────────────────────────────
# Both derived from dataset (Day_Type and Festival_Flag regression)
WEEKEND_MULT  = 1.219   # weekends see 21.9% more people than weekdays
FESTIVAL_MULT = 1.664   # festival days see 66.4% more people
 
# Major Indian festival / national holiday dates as (month, day)
FESTIVAL_DATES = {
    (1, 14), (1, 15), (1, 26),    # Makar Sankranti, Republic Day
    (3, 25), (3, 26),              # Holi
    (4, 14),                       # Baisakhi / Ambedkar Jayanti
    (8, 15),                       # Independence Day
    (10, 2), (10, 12),             # Gandhi Jayanti, Dussehra
    (10, 23), (10, 24),            # Diwali
    (11, 1), (11, 12), (11, 15),   # Regional festivals
    (12, 25),                      # Christmas
}
 
SHELTERS = sorted(SHELTER_DATA.keys())
 
 
# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
 
def get_season(month: int) -> str:
    if month in [12, 1, 2]:    return "Winter"
    if month in [3, 4, 5]:     return "Summer"
    if month in [6, 7, 8, 9]:  return "Monsoon"
    return "Post-Monsoon"
 
 
def is_festival(month: int, day: int) -> bool:
    return (month, day) in FESTIVAL_DATES
 
 
# ─── PREDICTION FUNCTION ──────────────────────────────────────────────────────
 
def predict(shelter_name: str, pred_date: date) -> dict:
    """
    Predict people expected and food required for a shelter on a given date.
 
    Parameters
    ----------
    shelter_name : str   — one of the 10 shelters
    pred_date    : date  — any future (or past) date
 
    Returns
    -------
    dict with:
        shelter, date, weekday, season, is_weekend, is_festival,
        predicted_people, low_people, high_people,
        food_kg, low_food_kg, high_food_kg,
        food_per_person_kg, packs_500g
    """
    if shelter_name not in SHELTER_DATA:
        raise ValueError(
            f"Unknown shelter '{shelter_name}'.\n"
            f"Valid options: {', '.join(SHELTERS)}"
        )
 
    s       = SHELTER_DATA[shelter_name]
    month   = pred_date.month
    day     = pred_date.day
    weekday = pred_date.weekday()      # 0 = Monday … 6 = Sunday
    is_we   = weekday >= 5
    is_fest = is_festival(month, day)
 
    base_u = s["m"][month]["u"]        # historical mean for this shelter+month
    base_s = s["m"][month]["s"]        # historical std dev
 
    # ── Point estimate ─────────────────────────────────────────────────────
    people = base_u
    if is_we:   people *= WEEKEND_MULT
    if is_fest: people *= FESTIVAL_MULT
    people = round(people)
 
    # ── Confidence range (mean ± 1 std dev, with same uplifts applied) ────
    low_p  = max(0, round((base_u - base_s) * (WEEKEND_MULT if is_we else 1)
                                             * (FESTIVAL_MULT if is_fest else 1)))
    high_p = round((base_u + base_s) * (WEEKEND_MULT if is_we else 1)
                                      * (FESTIVAL_MULT if is_fest else 1))
 
    # ── Food calculation ────────────────────────────────────────────────────
    fpp       = s["fpp"]
    food      = round(people  * fpp, 1)
    low_food  = round(low_p   * fpp, 1)
    high_food = round(high_p  * fpp, 1)
    packs_500g = int(-(-food // 0.5))  # ceiling: how many 500g packs needed
 
    return {
        "shelter":            shelter_name,
        "date":               pred_date.strftime("%d %b %Y"),
        "weekday":            pred_date.strftime("%A"),
        "season":             get_season(month),
        "is_weekend":         is_we,
        "is_festival":        is_fest,
        "predicted_people":   people,
        "low_people":         low_p,
        "high_people":        high_p,
        "food_kg":            food,
        "low_food_kg":        low_food,
        "high_food_kg":       high_food,
        "food_per_person_kg": fpp,
        "packs_500g":         packs_500g,
    }
 
 
# ─── DISPLAY ──────────────────────────────────────────────────────────────────
 
def print_banner():
    print("\n" + "═" * 56)
    print("   FOOD DONATION PREDICTOR")
    print("   Trained on 600 records  |  2022 – 2024")
    print("═" * 56)
 
 
def print_result(r: dict):
    print("\n" + "─" * 56)
    print(f"  Shelter  : {r['shelter']}")
    print(f"  Date     : {r['date']}  ({r['weekday']}, {r['season']})")
    if r["is_festival"]:
        print("  Note     : Festival day  → +66% uplift applied")
    if r["is_weekend"]:
        print("  Note     : Weekend       → +22% uplift applied")
    print("─" * 56)
    print(f"  {'People expected':<28}: {r['predicted_people']:>7,}")
    print(f"  {'  realistic range':<28}: {r['low_people']:>7,}  –  {r['high_people']:,}")
    print(f"  {'Food required (kg)':<28}: {r['food_kg']:>7.1f} kg")
    print(f"  {'  realistic range':<28}: {r['low_food_kg']:>7.1f}  –  {r['high_food_kg']:.1f} kg")
    print(f"  {'Food per person':<28}: {r['food_per_person_kg']:>7.3f} kg / person")
    print(f"  {'500g packs needed':<28}: {r['packs_500g']:>7,} packs")
    print("─" * 56 + "\n")
 
 
# ─── INPUT HELPERS ────────────────────────────────────────────────────────────
 
def choose_shelter() -> str:
    print("\n  Select a shelter:")
    for i, name in enumerate(SHELTERS, 1):
        print(f"    [{i:2}]  {name}")
    while True:
        raw = input("\n  Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(SHELTERS):
            return SHELTERS[int(raw) - 1]
        print("  Invalid — enter a number shown above.")
 
 
def choose_date() -> date:
    while True:
        raw = input("  Enter date (YYYY-MM-DD): ").strip()
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            print("  Invalid format. Example: 2025-12-25")
 
 
# ─── MAIN ─────────────────────────────────────────────────────────────────────
 
def main():
    print_banner()
 
    while True:
        print("\n" + "═" * 56)
        shelter   = choose_shelter()
        pred_date = choose_date()
 
        result = predict(shelter, pred_date)
        print_result(result)
 
        again = input("  Run another prediction? (y / n): ").strip().lower()
        if again != "y":
            print("\n  Goodbye!\n")
            break
 
 
if __name__ == "__main__":
    main()