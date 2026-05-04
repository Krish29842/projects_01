"""
=============================================================
  SHELTER ARRIVAL PREDICTION AGENT
  Predicts number of hungry people arriving at a shelter
  and calculates food quantity required.
=============================================================
Usage:
    python shelter_agent.py
    python shelter_agent.py --date 2026-06-15 --day Monday --shelter "Pahadi Niwas, Haridwar"
    python shelter_agent.py --list-shelters
=============================================================
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from datetime import datetime

# ── Color codes for terminal output ──────────────────────────────────────────
class Colors:
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    END       = '\033[0m'

def c(text, color): return f"{color}{text}{Colors.END}"

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║         SHELTER ARRIVAL PREDICTION AGENT                     ║
║         Food Security & Resource Planning Tool               ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}"""

# ── CSV path (auto-detects same directory or upload path) ────────────────────
POSSIBLE_PATHS = [
    "increasing_trend_donations_with_shelter.csv",
    "/mnt/user-data/uploads/increasing_trend_donations_with_shelter.csv",
    os.path.join(os.path.dirname(__file__), "increasing_trend_donations_with_shelter.csv"),
]

DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── Day-of-week and month seasonality factors (learned from domain knowledge) ─
DAY_FACTORS   = {0:0.95, 1:0.97, 2:1.00, 3:1.02, 4:1.05, 5:1.08, 6:1.10}
MONTH_FACTORS = {1:1.10, 2:1.05, 3:1.00, 4:0.95, 5:0.92, 6:0.90,
                 7:0.93, 8:0.95, 9:1.00,10:1.05,11:1.08,12:1.12}

# ── Food distribution ratios ──────────────────────────────────────────────────
FOOD_RATIOS = {
    "Rice / Grains":       0.45,
    "Dal / Lentils":       0.25,
    "Vegetables":          0.20,
    "Oil & Spices":        0.05,
    "Bread / Roti (kg)":   0.05,
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the CSV file."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Shelter Name'] = df['Shelter Name'].str.strip()
    df['No. of People Served'] = pd.to_numeric(df['No. of People Served'], errors='coerce')
    df['Quantity Delivered (kg)'] = pd.to_numeric(df['Quantity Delivered (kg)'], errors='coerce')
    df = df.dropna(subset=['No. of People Served', 'Shelter Name'])
    return df


def find_csv() -> str:
    """Find the CSV file from known paths."""
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    return None


def get_all_shelters(df: pd.DataFrame) -> list:
    return sorted(df['Shelter Name'].unique().tolist())


def validate_inputs(date_str: str, day_str: str, shelter_str: str,
                    df: pd.DataFrame) -> tuple:
    """
    Validates all three inputs.
    Returns (is_valid: bool, error_message: str, matched_shelter: str)
    """
    errors = []

    # ── Validate date ──────────────────────────────────────────────────────────
    try:
        target_date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        errors.append(f"Invalid date format '{date_str}'. Use YYYY-MM-DD (e.g. 2026-06-15).")
        target_date = None

    # ── Validate day ──────────────────────────────────────────────────────────
    day_str_clean = day_str.strip().capitalize()
    if day_str_clean not in DAY_NAMES:
        errors.append(f"Invalid day '{day_str}'. Choose from: {', '.join(DAY_NAMES)}.")
        day_valid = False
    else:
        day_valid = True

    # ── Cross-check date vs day ───────────────────────────────────────────────
    if target_date and day_valid:
        actual_day = target_date.strftime("%A")
        if actual_day != day_str_clean:
            errors.append(
                f"Date mismatch: {date_str} is a {c(actual_day, Colors.YELLOW)}, "
                f"not {c(day_str_clean, Colors.RED)}."
            )

    # ── Validate shelter ──────────────────────────────────────────────────────
    all_shelters = get_all_shelters(df)
    norm_input   = shelter_str.strip().lower()
    norm_map     = {s.lower(): s for s in all_shelters}

    if norm_input in norm_map:
        matched_shelter = norm_map[norm_input]
    else:
        # Fuzzy partial match
        candidates = [s for s in all_shelters if norm_input in s.lower()]
        if len(candidates) == 1:
            matched_shelter = candidates[0]
        elif len(candidates) > 1:
            errors.append(
                f"Ambiguous shelter '{shelter_str}'. Did you mean one of:\n"
                + "\n".join(f"  • {s}" for s in candidates[:5])
            )
            matched_shelter = None
        else:
            errors.append(
                f"Shelter '{shelter_str}' not found in dataset.\n"
                f"  Run with --list-shelters to see all valid names."
            )
            matched_shelter = None

    if errors:
        return False, "\n".join(errors), None

    return True, "", matched_shelter


def linear_regression_predict(x: np.ndarray, y: np.ndarray, x_pred: float) -> float:
    """Simple OLS linear regression: returns predicted y at x_pred."""
    n = len(x)
    if n < 2:
        return float(y[-1])
    sx, sy  = x.sum(), y.sum()
    sxy     = (x * y).sum()
    sx2     = (x ** 2).sum()
    denom   = n * sx2 - sx ** 2
    if denom == 0:
        return float(y.mean())
    slope     = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return intercept + slope * x_pred


def predict_arrivals(df: pd.DataFrame, shelter: str,
                     target_date: datetime) -> dict:
    """
    Predicts number of people arriving at a shelter on a given date.
    Uses linear regression on historical data + day/month adjustments.
    """
    shelter_df = df[df['Shelter Name'] == shelter].copy()
    shelter_df = shelter_df.sort_values('Date')

    # Aggregate by date (sum per day in case of multiple donations)
    daily = (shelter_df.groupby('Date')
             .agg({'No. of People Served': 'sum',
                   'Quantity Delivered (kg)': 'sum'})
             .reset_index())

    n = len(daily)

    # Build numeric index (days from first record)
    ref_date     = daily['Date'].min()
    daily['idx'] = (daily['Date'] - ref_date).dt.days
    x = daily['idx'].values.astype(float)
    y = daily['No. of People Served'].values.astype(float)

    # Predict at target index
    target_idx = (target_date - ref_date).days
    base_pred  = linear_regression_predict(x, y, float(max(target_idx, x[-1])))
    base_pred  = max(base_pred, 5.0)

    # Apply day-of-week and month seasonality
    dow        = target_date.weekday()   # Monday=0, Sunday=6
    month      = target_date.month
    day_f      = DAY_FACTORS.get(dow, 1.0)
    month_f    = MONTH_FACTORS.get(month, 1.0)
    predicted  = round(base_pred * day_f * month_f)

    # Average kg per person from history
    valid_rows    = daily[daily['No. of People Served'] > 0].copy()
    valid_rows['kgpp'] = (valid_rows['Quantity Delivered (kg)'] /
                          valid_rows['No. of People Served'])
    avg_kgpp      = valid_rows['kgpp'].mean() if len(valid_rows) > 0 else 0.95

    # Confidence score
    confidence    = min(97, 55 + (n // 5) * 3)

    return {
        "shelter":       shelter,
        "date":          target_date.strftime("%Y-%m-%d"),
        "day":           target_date.strftime("%A"),
        "predicted":     int(predicted),
        "avg_kgpp":      round(avg_kgpp, 3),
        "total_food_kg": round(predicted * avg_kgpp, 2),
        "confidence":    confidence,
        "data_points":   n,
        "trend_slope":   round(linear_regression_predict(x, y, float(x[-1]+1)) -
                               linear_regression_predict(x, y, float(x[-1])), 3),
    }


def compute_food_breakdown(total_kg: float) -> dict:
    """Break total food kg into categories."""
    return {cat: round(total_kg * ratio, 2) for cat, ratio in FOOD_RATIOS.items()}


def print_result(result: dict, food: dict):
    """Pretty-print the prediction result."""
    w = 62
    sep = "─" * w

    print(f"\n{c(sep, Colors.CYAN)}")
    print(c(f"  PREDICTION RESULT", Colors.BOLD + Colors.CYAN))
    print(c(sep, Colors.CYAN))

    print(f"  {c('Shelter :', Colors.BLUE)}  {result['shelter']}")
    print(f"  {c('Date    :', Colors.BLUE)}  {result['date']}  ({result['day']})")
    print(f"  {c('Records :', Colors.BLUE)}  {result['data_points']} historical data points used")

    trend_dir = "↑ increasing" if result['trend_slope'] > 0 else "↓ decreasing"
    trend_col = Colors.YELLOW if result['trend_slope'] > 0 else Colors.GREEN
    print(f"  {c('Trend   :', Colors.BLUE)}  {c(trend_dir, trend_col)}")
    print()

    # Main metrics
    print(c(f"  {'─'*58}", Colors.CYAN))
    print(f"  {c('ARRIVALS FORECAST', Colors.BOLD)}")
    print(c(f"  {'─'*58}", Colors.CYAN))
    print(f"\n  {c('Predicted people arriving :', Colors.YELLOW)}  "
          f"{c(str(result['predicted']), Colors.BOLD + Colors.GREEN)} people")
    print(f"  {c('Total food required       :', Colors.YELLOW)}  "
          f"{c(str(result['total_food_kg']) + ' kg', Colors.BOLD + Colors.GREEN)}")
    print(f"  {c('Avg food per person       :', Colors.YELLOW)}  "
          f"{result['avg_kgpp']} kg/person")

    # Confidence bar
    conf  = result['confidence']
    filled = int(conf / 5)
    bar   = "█" * filled + "░" * (20 - filled)
    col   = Colors.GREEN if conf >= 75 else Colors.YELLOW
    print(f"  {c('Confidence                :', Colors.YELLOW)}  "
          f"{c(bar, col)}  {c(str(conf)+'%', col)}")

    # Food breakdown
    print()
    print(c(f"  {'─'*58}", Colors.CYAN))
    print(f"  {c('FOOD BREAKDOWN (kg)', Colors.BOLD)}")
    print(c(f"  {'─'*58}", Colors.CYAN))
    max_val = max(food.values()) if food else 1
    for item, kg in food.items():
        bar_len   = int((kg / max_val) * 30)
        bar_str   = "▓" * bar_len + "░" * (30 - bar_len)
        print(f"  {item:<22} {c(bar_str, Colors.BLUE)}  {c(str(kg)+' kg', Colors.BOLD)}")

    # Contextual notes
    print()
    print(c(f"  {'─'*58}", Colors.CYAN))
    print(f"  {c('NOTES', Colors.BOLD)}")
    print(c(f"  {'─'*58}", Colors.CYAN))
    d = datetime.strptime(result['date'], "%Y-%m-%d")
    if d.weekday() >= 5:
        print(f"  {c('⚑', Colors.YELLOW)}  Weekend: demand typically 5–10% higher than weekdays.")
    if d.month in [11, 12, 1, 2]:
        print(f"  {c('⚑', Colors.CYAN)}  Winter month: expect elevated footfall at this shelter.")
    if result['trend_slope'] > 0.5:
        print(f"  {c('⚑', Colors.YELLOW)}  Strong upward trend detected — consider extra buffer stock.")
    print(f"  {c('⚑', Colors.GREEN)}  Prepare {int(result['predicted'] * 1.1)} meals for 10% safety buffer.")
    print(f"\n{c(sep, Colors.CYAN)}\n")


def print_shelter_list(df: pd.DataFrame):
    """Print all shelter names neatly."""
    shelters = get_all_shelters(df)
    print(f"\n{c('All shelters in dataset:', Colors.BOLD + Colors.CYAN)} "
          f"({len(shelters)} total)\n")
    for i, s in enumerate(shelters, 1):
        print(f"  {c(str(i).rjust(3)+'.', Colors.BLUE)} {s}")
    print()


def interactive_mode(df: pd.DataFrame):
    """Run the agent interactively (no CLI args)."""
    print(BANNER)
    print(c("  Interactive mode — type 'list' to see all shelters, 'quit' to exit.\n",
            Colors.CYAN))

    while True:
        print(c("─" * 62, Colors.CYAN))

        # Date input
        date_str = input(c("  Enter date (YYYY-MM-DD): ", Colors.YELLOW)).strip()
        if date_str.lower() in ('quit', 'exit', 'q'):
            print(c("\n  Goodbye!\n", Colors.GREEN)); break
        if date_str.lower() == 'list':
            print_shelter_list(df); continue

        # Day input
        day_str = input(c("  Enter day of week     : ", Colors.YELLOW)).strip()

        # Shelter input
        shelter_str = input(c("  Enter shelter name    : ", Colors.YELLOW)).strip()
        if shelter_str.lower() == 'list':
            print_shelter_list(df); continue

        # Validate
        valid, err, shelter = validate_inputs(date_str, day_str, shelter_str, df)
        if not valid:
            print(f"\n  {c('✗  INVALID', Colors.BOLD + Colors.RED)}\n")
            for line in err.split('\n'):
                print(f"  {c('→', Colors.RED)} {line}")
            print()
            continue

        # Predict
        target_date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        result = predict_arrivals(df, shelter, target_date)
        food   = compute_food_breakdown(result['total_food_kg'])
        print_result(result, food)

        again = input(c("  Make another prediction? (y/n): ", Colors.CYAN)).strip().lower()
        if again != 'y':
            print(c("\n  Goodbye!\n", Colors.GREEN)); break


def main():
    parser = argparse.ArgumentParser(
        description="Shelter Arrival Prediction Agent",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--date',     type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('--day',      type=str, help='Day of week (Monday–Sunday)')
    parser.add_argument('--shelter',  type=str, help='Shelter name (use quotes for names with commas)')
    parser.add_argument('--csv',      type=str, help='Path to CSV file (optional)')
    parser.add_argument('--list-shelters', action='store_true', help='List all valid shelter names')
    args = parser.parse_args()

    # ── Find CSV ──────────────────────────────────────────────────────────────
    csv_path = args.csv if args.csv else find_csv()
    if not csv_path:
        print(c("\n  ✗ CSV file not found. Place 'increasing_trend_donations_with_shelter.csv'"
                "\n    in the same directory or pass --csv <path>\n", Colors.RED))
        sys.exit(1)

    try:
        df = load_data(csv_path)
    except Exception as e:
        print(c(f"\n  ✗ Failed to load CSV: {e}\n", Colors.RED))
        sys.exit(1)

    print(c(f"\n  ✓ Loaded {len(df):,} records from {os.path.basename(csv_path)}", Colors.GREEN))
    print(c(f"    Shelters: {df['Shelter Name'].nunique()}  |  "
            f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}\n", Colors.CYAN))

    # ── List shelters mode ────────────────────────────────────────────────────
    if args.list_shelters:
        print_shelter_list(df)
        sys.exit(0)

    # ── CLI mode: all three args provided ─────────────────────────────────────
    if args.date and args.day and args.shelter:
        valid, err, shelter = validate_inputs(args.date, args.day, args.shelter, df)
        if not valid:
            print(f"\n  {c('✗  INVALID', Colors.BOLD + Colors.RED)}\n")
            for line in err.split('\n'):
                print(f"  {c('→', Colors.RED)} {line}")
            print()
            sys.exit(1)

        target_date = datetime.strptime(args.date.strip(), "%Y-%m-%d")
        result = predict_arrivals(df, shelter, target_date)
        food   = compute_food_breakdown(result['total_food_kg'])
        print(BANNER)
        print_result(result, food)

    # ── Interactive mode ──────────────────────────────────────────────────────
    else:
        interactive_mode(df)


if __name__ == "__main__":
    main()
