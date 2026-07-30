import json
import os

from analytics import get_category_summary

BUDGET_FILE = "budgets.json"

def load_budgets():
    """
    Loads all budgets from budgets.json.
    Returns an empty dictionary if the file doesn't exist.
    """

    if not os.path.exists(BUDGET_FILE):
        return {}

    with open(BUDGET_FILE, "r") as file:
        return json.load(file)

def save_budgets(budgets):
    """
    Saves the budget dictionary to budgets.json.
    """

    with open(BUDGET_FILE, "w") as file:
        json.dump(
            budgets,
            file,
            indent=4
        )

def set_budget(category, amount):
    """
    Creates or updates a budget for a category.
    """

    budgets = load_budgets()

    budgets[category.lower()] = float(amount)

    save_budgets(budgets)

def get_budget(category):
    """
    Returns the budget for a category.
    """

    budgets = load_budgets()

    return budgets.get(category.lower(), 0)

##engine

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

        used = (spent / budget) * 100 if budget > 0 else 0

        report.append({

            "category": category,

            "budget": budget,

            "spent": spent,

            "remaining": remaining,

            "used": round(used,2)

        })

    return report

def delete_budget(category):
    """
    Deletes a budget for a category.
    """

    budgets = load_budgets()

    if category.lower() in budgets:
        del budgets[category.lower()]
        save_budgets(budgets)

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