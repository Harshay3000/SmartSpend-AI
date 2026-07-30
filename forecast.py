import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from analytics import load_expense_data
from analytics import (
    load_expense_data,
    get_total_spending,
)

# ==========================================================
# CONFIGURATION
# ==========================================================

MIN_TRAINING_DAYS = 14


# ==========================================================
# PREPARE DAILY DATA
# ==========================================================

def prepare_daily_data(filepath="expenses.csv"):
    """
    Loads expenses and creates one row for every calendar day.

    Days with no expenses are assigned ₹0.
    """

    df = load_expense_data(filepath)

    if df.empty:
        return pd.DataFrame()

    daily = (
        df.groupby("date")["amount"]
        .sum()
        .reset_index()
    )

    daily = daily.sort_values("date")

    full_dates = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D",
    )

    daily = (
        daily.set_index("date")
        .reindex(full_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    daily["amount"] = pd.to_numeric(
        daily["amount"],
        errors="coerce",
    ).fillna(0)

    return daily


# ==========================================================
# BASELINE FEATURES
# ==========================================================

def create_baseline_features(daily):
    """
    Creates the simple baseline feature:
    day_number
    """

    if daily.empty:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
        )

    data = daily.copy()

    data["day_number"] = range(
        len(data)
    )

    X = data[
        ["day_number"]
    ]

    y = data["amount"]

    return X, y


# ==========================================================
# ADVANCED FEATURE ENGINEERING
# ==========================================================

def create_forecasting_features(daily):
    """
    Creates time and lag-based features.

    Features:
    - day_number
    - day_of_week
    - is_weekend
    - day_of_month
    - month
    - lag_1
    - lag_7
    - rolling_7_mean
    """

    if daily.empty:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
        )

    data = daily.copy()

    # ------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------

    data["day_number"] = range(
        len(data)
    )

    data["day_of_week"] = (
        data["date"].dt.dayofweek
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["day_of_month"] = (
        data["date"].dt.day
    )

    data["month"] = (
        data["date"].dt.month
    )

    # ------------------------------------------------------
    # Lag features
    # ------------------------------------------------------

    data["lag_1"] = (
        data["amount"].shift(1)
    )

    data["lag_7"] = (
        data["amount"].shift(7)
    )

    # ------------------------------------------------------
    # Rolling average
    # ------------------------------------------------------

    data["rolling_7_mean"] = (
        data["amount"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    # ------------------------------------------------------
    # Remove rows without enough history
    # ------------------------------------------------------

    data = data.dropna()

    if data.empty:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
        )

    features = [
        "day_number",
        "day_of_week",
        "is_weekend",
        "day_of_month",
        "month",
        "lag_1",
        "lag_7",
        "rolling_7_mean",
    ]

    X = data[features]

    y = data["amount"]

    return X, y


# ==========================================================
# TRAIN BASELINE MODEL
# ==========================================================

def train_baseline_model(
    filepath="expenses.csv"
):
    """
    Trains the simple day-number baseline model.
    """

    daily = prepare_daily_data(
        filepath
    )

    if len(daily) < MIN_TRAINING_DAYS:
        return None, daily

    X, y = create_baseline_features(
        daily
    )

    if X.empty:
        return None, daily

    model = LinearRegression()

    model.fit(
        X,
        y,
    )

    return model, daily


# ==========================================================
# TRAIN IMPROVED MODEL
# ==========================================================

def train_forecasting_model(
    filepath="expenses.csv"
):
    """
    Trains the feature-engineered forecasting model.
    """

    daily = prepare_daily_data(
        filepath
    )

    if len(daily) < MIN_TRAINING_DAYS:
        return None, daily

    X, y = create_forecasting_features(
        daily
    )

    if X.empty:
        return None, daily

    model = LinearRegression()

    model.fit(
        X,
        y,
    )

    return model, daily


# ==========================================================
# EVALUATION HELPER
# ==========================================================

def _calculate_metrics(
    y_true,
    predictions,
):
    """
    Calculates MAE, RMSE and R².
    """

    predictions = np.maximum(
        predictions,
        0,
    )

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    if len(y_true) >= 2:

        r2 = r2_score(
            y_true,
            predictions,
        )

    else:

        r2 = None

    return mae, rmse, r2


# ==========================================================
# EVALUATE BASELINE MODEL
# ==========================================================

def evaluate_baseline_model(
    filepath="expenses.csv"
):
    """
    Evaluates the baseline model using TimeSeriesSplit.
    """

    daily = prepare_daily_data(
        filepath
    )

    if daily.empty:

        return {
            "status": "insufficient_data",
            "message": "No expense data available.",
        }

    if len(daily) < MIN_TRAINING_DAYS:

        return {
            "status": "insufficient_data",
            "message": (
                f"At least {MIN_TRAINING_DAYS} "
                "days are recommended."
            ),
        }

    X, y = create_baseline_features(
        daily
    )

    tscv = TimeSeriesSplit(
        n_splits=3
    )

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    fold_results = []

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        tscv.split(X),
        start=1,
    ):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]

        model = LinearRegression()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        mae, rmse, r2 = _calculate_metrics(
            y_test,
            predictions,
        )

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        if r2 is not None:
            r2_scores.append(r2)

        fold_results.append({
            "fold": fold_number,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": (
                round(r2, 3)
                if r2 is not None
                else None
            ),
        })

    return {
        "status": "success",
        "mae": round(
            np.mean(mae_scores),
            2,
        ),
        "rmse": round(
            np.mean(rmse_scores),
            2,
        ),
        "r2": (
            round(
                np.mean(r2_scores),
                3,
            )
            if r2_scores
            else None
        ),
        "folds": fold_results,
        "training_days": len(daily),
    }


# ==========================================================
# EVALUATE IMPROVED MODEL
# ==========================================================

def evaluate_forecasting_model(
    filepath="expenses.csv"
):
    """
    Evaluates the improved feature-engineered model.
    """

    daily = prepare_daily_data(
        filepath
    )

    if daily.empty:

        return {
            "status": "insufficient_data",
            "message": "No expense data available.",
        }

    if len(daily) < MIN_TRAINING_DAYS:

        return {
            "status": "insufficient_data",
            "message": (
                f"At least {MIN_TRAINING_DAYS} "
                "days are recommended for the "
                "improved model."
            ),
        }

    X, y = create_forecasting_features(
        daily
    )

    if X.empty or len(X) < 8:

        return {
            "status": "insufficient_data",
            "message": (
                "Not enough usable rows remain "
                "after feature engineering."
            ),
        }

    n_splits = 3

    if len(X) < 12:
        n_splits = 2

    tscv = TimeSeriesSplit(
        n_splits=n_splits
    )

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    fold_results = []

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        tscv.split(X),
        start=1,
    ):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]

        model = LinearRegression()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        mae, rmse, r2 = _calculate_metrics(
            y_test,
            predictions,
        )

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        if r2 is not None:
            r2_scores.append(r2)

        fold_results.append({
            "fold": fold_number,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": (
                round(r2, 3)
                if r2 is not None
                else None
            ),
        })

    return {
        "status": "success",
        "mae": round(
            np.mean(mae_scores),
            2,
        ),
        "rmse": round(
            np.mean(rmse_scores),
            2,
        ),
        "r2": (
            round(
                np.mean(r2_scores),
                3,
            )
            if r2_scores
            else None
        ),
        "folds": fold_results,
        "training_days": len(daily),
        "usable_rows": len(X),
        "features": list(X.columns),
    }


# ==========================================================
# MODEL COMPARISON
# ==========================================================

def compare_models(
    filepath="expenses.csv"
):
    """
    Compares the baseline and improved models.

    Lower MAE and RMSE are better.
    Higher R² is generally better.
    """

    baseline = evaluate_baseline_model(
        filepath
    )

    improved = evaluate_forecasting_model(
        filepath
    )

    if (
        baseline["status"] != "success"
        or improved["status"] != "success"
    ):

        return {
            "status": "insufficient_data",
            "baseline": baseline,
            "improved": improved,
        }

    mae_improvement = (
        (
            baseline["mae"]
            - improved["mae"]
        )
        / baseline["mae"]
        * 100
        if baseline["mae"] != 0
        else 0
    )

    rmse_improvement = (
        (
            baseline["rmse"]
            - improved["rmse"]
        )
        / baseline["rmse"]
        * 100
        if baseline["rmse"] != 0
        else 0
    )

    return {
        "status": "success",

        "baseline": baseline,

        "improved": improved,

        "mae_improvement_percent":
            round(
                mae_improvement,
                2,
            ),

        "rmse_improvement_percent":
            round(
                rmse_improvement,
                2,
            ),
    }


# ==========================================================
# RECURSIVE IMPROVED FORECAST
# ==========================================================

def forecast_next_days_improved(
    days=7,
    filepath="expenses.csv",
):
    """
    Generates a multi-day forecast using the improved
    feature-engineered model.

    Future predictions are fed back as lag values for
    later prediction steps.
    """

    model, daily = train_forecasting_model(
        filepath
    )

    if model is None:

        return pd.DataFrame()

    if daily.empty:

        return pd.DataFrame()

    # ------------------------------------------------------
    # Historical amount series
    # ------------------------------------------------------

    history = (
        daily["amount"]
        .astype(float)
        .tolist()
    )

    last_date = daily["date"].max()

    future_rows = []

    # ------------------------------------------------------
    # Generate one day at a time
    # ------------------------------------------------------

    for step in range(
        1,
        days + 1,
    ):

        future_date = (
            last_date
            + pd.Timedelta(days=step)
        )

        day_number = (
            len(history)
        )

        day_of_week = (
            future_date.dayofweek
        )

        is_weekend = (
            1
            if day_of_week >= 5
            else 0
        )

        day_of_month = (
            future_date.day
        )

        month = (
            future_date.month
        )

        # ----------------------------------------------
        # Lag 1
        # ----------------------------------------------

        lag_1 = history[-1]

        # ----------------------------------------------
        # Lag 7
        # ----------------------------------------------

        if len(history) >= 7:

            lag_7 = history[-7]

        else:

            lag_7 = history[0]

        # ----------------------------------------------
        # Previous 7-day average
        # ----------------------------------------------

        recent_history = history[-7:]

        rolling_7_mean = (
            sum(recent_history)
            / len(recent_history)
        )

        # ----------------------------------------------
        # Create feature row
        # ----------------------------------------------

        X_future = pd.DataFrame({
            "day_number": [
                day_number
            ],

            "day_of_week": [
                day_of_week
            ],

            "is_weekend": [
                is_weekend
            ],

            "day_of_month": [
                day_of_month
            ],

            "month": [
                month
            ],

            "lag_1": [
                lag_1
            ],

            "lag_7": [
                lag_7
            ],

            "rolling_7_mean": [
                rolling_7_mean
            ],
        })

        prediction = model.predict(
            X_future
        )[0]

        prediction = max(
            float(prediction),
            0.0,
        )

        # ----------------------------------------------
        # Add prediction to history so it can become
        # a lag feature for future days.
        # ----------------------------------------------

        history.append(
            prediction
        )

        future_rows.append({
            "date": future_date,
            "predicted_amount":
                round(
                    prediction,
                    2,
                ),
        })

    return pd.DataFrame(
        future_rows
    )


# ==========================================================
# BASELINE FORECAST
# ==========================================================

def forecast_next_days(
    days=7,
    filepath="expenses.csv",
):
    """
    Generates a simple baseline forecast.
    """

    model, daily = train_baseline_model(
        filepath
    )

    if model is None:

        return pd.DataFrame()

    last_day_number = (
        len(daily) - 1
    )

    future_day_numbers = [
        last_day_number + i
        for i in range(
            1,
            days + 1,
        )
    ]

    future_X = pd.DataFrame({
        "day_number":
            future_day_numbers
    })

    predictions = model.predict(
        future_X
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    future_dates = pd.date_range(
        start=(
            daily["date"].max()
            + pd.Timedelta(days=1)
        ),
        periods=days,
        freq="D",
    )

    return pd.DataFrame({
        "date": future_dates,
        "predicted_amount":
            predictions,
    })


# ==========================================================
# BASELINE TOTAL
# ==========================================================

def forecast_total(
    days=7,
    filepath="expenses.csv",
):
    """
    Returns the total predicted spending from
    the baseline model.
    """

    forecast = forecast_next_days(
        days=days,
        filepath=filepath,
    )

    if forecast.empty:

        return None

    return float(
        forecast[
            "predicted_amount"
        ].sum()
    )


# ==========================================================
# IMPROVED TOTAL
# ==========================================================

def forecast_total_improved(
    days=7,
    filepath="expenses.csv",
):
    """
    Returns the total predicted spending from
    the improved recursive model.
    """

    forecast = forecast_next_days_improved(
        days=days,
        filepath=filepath,
    )

    if forecast.empty:

        return None

    return float(
        forecast[
            "predicted_amount"
        ].sum()
    )

from budget import load_budgets


def get_total_monthly_budget():
    """
    Returns the combined value of all configured category
    budgets.

    Example:
        groceries = 5000
        fuel = 2500
        books = 1500

        total monthly budget = 9000
    """

    budgets = load_budgets()

    if not budgets:
        return 0.0

    return float(
        sum(
            float(amount)
            for amount in budgets.values()
        )
    )

def get_current_month_spending(filepath="expenses.csv"):
    """
    Returns spending for the current calendar month.
    """

    df = load_expense_data(filepath)

    if df.empty:
        return 0.0

    today = pd.Timestamp.today()

    current_month = df[
        (df["date"].dt.year == today.year)
        & (df["date"].dt.month == today.month)
    ]

    return float(
        current_month["amount"].sum()
    )

def get_days_remaining_in_month():
    """
    Returns the number of calendar days remaining
    after today in the current month.
    """

    today = pd.Timestamp.today()

    next_month = (
        today.replace(day=1)
        + pd.offsets.MonthBegin(1)
    )

    last_day = (
        next_month
        - pd.Timedelta(days=1)
    )

    return int(
        (last_day - today.normalize()).days
    )

def forecast_month_end(filepath="expenses.csv"):
    """
    Estimates total spending by the end of the current month.

    Returns a dictionary containing:
    - current spending
    - days remaining
    - forecasted remaining spending
    - projected month-end spending
    - monthly budget
    - projected difference
    - status
    """

    current_spending = get_current_month_spending(
        filepath
    )

    monthly_budget = get_total_monthly_budget()

    days_remaining = get_days_remaining_in_month()

    # ------------------------------------------------------
    # No future days
    # ------------------------------------------------------

    if days_remaining <= 0:

        projected_total = current_spending

        remaining_forecast = 0.0

    else:

        forecast = forecast_next_days_improved(
            days=days_remaining,
            filepath=filepath,
        )

        if forecast.empty:

            return {
                "status": "insufficient_data",
                "message": (
                    "Not enough historical data "
                    "to forecast the rest of the month."
                ),
            }

        remaining_forecast = float(
            forecast["predicted_amount"].sum()
        )

        projected_total = (
            current_spending
            + remaining_forecast
        )

    # ------------------------------------------------------
    # No budget
    # ------------------------------------------------------

    if monthly_budget <= 0:

        return {
            "status": "no_budget",
            "current_spending":
                current_spending,
            "days_remaining":
                days_remaining,
            "forecast_remaining":
                remaining_forecast,
            "projected_total":
                projected_total,
            "monthly_budget":
                None,
        }

    # ------------------------------------------------------
    # Compare forecast with budget
    # ------------------------------------------------------

    difference = (
        projected_total
        - monthly_budget
    )

    if difference > 0:

        budget_status = "AT_RISK"

    elif projected_total >= (
        monthly_budget * 0.9
    ):

        budget_status = "NEAR_LIMIT"

    else:

        budget_status = "WITHIN_BUDGET"

    return {
        "status": "success",

        "current_spending":
            round(
                current_spending,
                2,
            ),

        "days_remaining":
            days_remaining,

        "forecast_remaining":
            round(
                remaining_forecast,
                2,
            ),

        "projected_total":
            round(
                projected_total,
                2,
            ),

        "monthly_budget":
            round(
                monthly_budget,
                2,
            ),

        "difference":
            round(
                difference,
                2,
            ),

        "budget_status":
            budget_status,
    }

def get_forecast_confidence(filepath="expenses.csv"):
    """
    Gives a simple confidence label based on the amount
    of historical data available.

    This is a project heuristic, not a statistical
    confidence interval.
    """

    daily = prepare_daily_data(
        filepath
    )

    days = len(daily)

    if days < 14:
        return "Low"

    elif days < 30:
        return "Medium"

    else:
        return "Higher"