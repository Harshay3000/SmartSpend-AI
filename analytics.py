from datetime import datetime, timedelta
import pandas as pd
import os


def load_expense_data(filepath="expenses.csv"):
    """
    Loads expense data, validates it, and returns
    a clean DataFrame ready for analysis.
    """

    if not os.path.exists(filepath):
        return pd.DataFrame()

    if os.path.getsize(filepath) == 0:
        return pd.DataFrame()

    df = pd.read_csv(filepath)

    if df.empty:
        return pd.DataFrame()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["amount", "date", "category"])

    return df

def get_total_spending(filepath="expenses.csv"):
    """
    Returns the total amount spent.
    """

    df = load_expense_data(filepath)

    if df.empty:
        return 0

    return float(df["amount"].sum())

def get_transaction_count(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return 0

    return len(df)

def get_highest_expense(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return None

    return df.loc[df["amount"].idxmax()]

def get_lowest_expense(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return None

    return df.loc[df["amount"].idxmin()]

def get_category_summary(filepath="expenses.csv"):
    """
    Returns total spending for each category.
    """

    df = load_expense_data(filepath)

    if df.empty:
        return {}

    category_summary = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    return category_summary.to_dict()

def get_monthly_spending(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return {}

    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["amount"]
        .sum()
    )

    return monthly.astype(float).to_dict()

def get_daily_spending(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return {}

    daily = (
        df.groupby("date")["amount"]
        .sum()
    )

    return daily.astype(float).to_dict()

def get_average_daily_spending(filepath="expenses.csv"):

    daily = get_daily_spending(filepath)

    if not daily:
        return 0

    return sum(daily.values()) / len(daily)

def get_top_categories(n=5, filepath="expenses.csv"):

    summary = get_category_summary(filepath)

    if not summary:
        return {}

    return dict(list(summary.items())[:n])

def get_spending_trend(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return pd.DataFrame()

    trend = (
        df.groupby("date")["amount"]
        .sum()
        .reset_index()
    )

    return trend

def get_this_week_spending(filepath="expenses.csv"):
    """
    Returns the total spending for the current week.
    """

    df = load_expense_data(filepath)

    if df.empty:
        return 0

    today = pd.Timestamp.today().normalize()

    start_of_week = today - pd.Timedelta(days=today.weekday())

    weekly_df = df[df["date"] >= start_of_week]

    return float(weekly_df["amount"].sum())

def get_highest_category(filepath="expenses.csv"):

    summary = get_category_summary(filepath)

    if not summary:
        return None

    category = max(summary, key=summary.get)

    return category, summary[category]

def get_largest_transaction(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return None

    row = df.loc[df["amount"].idxmax()]

    return {
        "amount": float(row["amount"]),
        "category": row["category"],
        "date": row["date"].strftime("%Y-%m-%d"),
        "note": row["note"]
    }

def get_weekday_spending(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return {}

    df["weekday"] = df["date"].dt.day_name()

    weekday_summary = (
        df.groupby("weekday")["amount"]
        .sum()
    )

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_summary = weekday_summary.reindex(weekday_order, fill_value=0)

    return weekday_summary.to_dict()

def get_cumulative_spending(filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("date")

    daily = (
        df.groupby("date")["amount"]
        .sum()
        .reset_index()
    )

    daily["cumulative"] = daily["amount"].cumsum()

    return daily

def get_daily_growth(filepath="expenses.csv"):

    daily = get_spending_trend(filepath)

    if len(daily) < 2:
        return None

    previous = daily.iloc[-2]["amount"]
    current = daily.iloc[-1]["amount"]

    if previous == 0:
        return None

    growth = ((current - previous) / previous) * 100

    return round(growth, 2)

def get_recent_transactions(n=5, filepath="expenses.csv"):

    df = load_expense_data(filepath)

    if df.empty:

        return df

    return df.sort_values(
        "date",
        ascending=False
    ).head(n)