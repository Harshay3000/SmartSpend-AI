from analytics import get_category_summary

from database import (
    save_budget,
    get_budget as db_get_budget,
    get_all_budgets,
    delete_budget as db_delete_budget
)


def load_budgets():
    """
    Returns all budgets as a dictionary.

    Example:

    {
        "Fuel": 2500,
        "Books": 1500
    }
    """

    df = get_all_budgets()

    if df.empty:
        return {}

    return dict(
        zip(
            df["category"],
            df["amount"]
        )
    )


def get_budget(category):

    row = db_get_budget(category)

    if row is None:
        return 0

    return row["amount"]


def set_budget(category, amount):

    save_budget(
        category,
        float(amount)
    )


def get_budget_status():
    """
    Combines budgets with actual spending.
    """

    budgets = load_budgets()

    spending = get_category_summary()

    report = []

    for category, budget in budgets.items():

        spent = spending.get(category, 0)

        remaining = budget - spent

        used = (
            (spent / budget) * 100
            if budget > 0
            else 0
        )

        report.append({

            "category": category,

            "budget": budget,

            "spent": spent,

            "remaining": remaining,

            "used": round(used, 2)

        })

    return report


def delete_budget(category):

    db_delete_budget(category)


def get_over_budget_categories():
    """
    Returns all categories that have exceeded their budget.
    """

    report = get_budget_status()

    return [
        item
        for item in report
        if item["used"] >= 100
    ]

