from llm_parser import _get_client

from analytics import (
    get_total_spending,
    get_category_summary,
    get_average_daily_spending,
    get_largest_transaction,
    get_monthly_spending,
)

from budget import get_budget_status

# FINANCIAL CONTEXT BUILDER
def build_financial_context():
    """
    Builds a structured, deterministic financial context for
    the LLM.

    Important calculations are performed in Python first.
    The LLM is responsible for explanation and reasoning,
    not for calculating the underlying financial facts.
    """

    # Basic analytics
    total = get_total_spending()

    categories = get_category_summary()

    average_daily = get_average_daily_spending()

    largest_transaction = get_largest_transaction()

    monthly = get_monthly_spending()

    # Budget information

    budget_report = get_budget_status()

    # Find highest spending category

    highest_category = None
    highest_category_amount = 0.0

    if categories:

        highest_category = max(
            categories,
            key=categories.get,
        )

        highest_category_amount = float(
            categories[highest_category]
        )

    # Create lookup table for budgets

    budget_lookup = {}

    for item in budget_report:

        category = str(
            item["category"]
        ).strip().lower()

        budget_lookup[category] = item

    # Build category-budget cross references

    category_budget_lines = []

    for category, amount in categories.items():

        normalized_category = str(
            category
        ).strip().lower()

        budget_item = budget_lookup.get(
            normalized_category
        )

        if budget_item:

            budget = float(
                budget_item["budget"]
            )

            spent = float(
                budget_item["spent"]
            )

            remaining = float(
                budget_item["remaining"]
            )

            used = float(
                budget_item["used"]
            )

            if used >= 100:

                status = "OVER BUDGET"

            elif used >= 80:

                status = "NEAR BUDGET LIMIT"

            else:

                status = "WITHIN BUDGET"

            category_budget_lines.append(
                (
                    f"- {category}: "
                    f"spent ₹{spent:,.2f}, "
                    f"budget ₹{budget:,.2f}, "
                    f"remaining ₹{remaining:,.2f}, "
                    f"used {used:.1f}%, "
                    f"status = {status}"
                )
            )

        else:

            category_budget_lines.append(
                (
                    f"- {category}: "
                    f"spent ₹{float(amount):,.2f}, "
                    f"budget = NOT SET"
                )
            )

    # Largest transaction budget relationship

    largest_transaction_context = (
        "No transaction data available."
    )

    if largest_transaction:

        largest_category = str(
            largest_transaction["category"]
        ).strip().lower()

        largest_amount = float(
            largest_transaction["amount"]
        )

        largest_date = largest_transaction.get(
            "date",
            "Unknown",
        )

        largest_note = largest_transaction.get(
            "note",
            "",
        )

        largest_budget_item = budget_lookup.get(
            largest_category
        )

        if largest_budget_item:

            largest_budget = float(
                largest_budget_item["budget"]
            )

            largest_spent = float(
                largest_budget_item["spent"]
            )

            largest_remaining = float(
                largest_budget_item["remaining"]
            )

            largest_used = float(
                largest_budget_item["used"]
            )

            if largest_used >= 100:

                largest_status = "OVER BUDGET"

            elif largest_used >= 80:

                largest_status = "NEAR BUDGET LIMIT"

            else:

                largest_status = "WITHIN BUDGET"

            largest_transaction_context = (
                f"Amount: ₹{largest_amount:,.2f}\n"
                f"Category: {largest_category}\n"
                f"Date: {largest_date}\n"
                f"Note: {largest_note}\n"
                f"Category budget: ₹{largest_budget:,.2f}\n"
                f"Category total spent: ₹{largest_spent:,.2f}\n"
                f"Category remaining budget: "
                f"₹{largest_remaining:,.2f}\n"
                f"Category budget usage: "
                f"{largest_used:.1f}%\n"
                f"Budget status: {largest_status}"
            )

        else:

            largest_transaction_context = (
                f"Amount: ₹{largest_amount:,.2f}\n"
                f"Category: {largest_category}\n"
                f"Date: {largest_date}\n"
                f"Note: {largest_note}\n"
                f"Category budget: NOT SET"
            )

    # Highest category budget relationship

    highest_category_context = (
        "No category data available."
    )

    if highest_category:

        highest_budget_item = budget_lookup.get(
            highest_category.lower()
        )

        if highest_budget_item:

            highest_budget = float(
                highest_budget_item["budget"]
            )

            highest_spent = float(
                highest_budget_item["spent"]
            )

            highest_remaining = float(
                highest_budget_item["remaining"]
            )

            highest_used = float(
                highest_budget_item["used"]
            )

            if highest_used >= 100:

                highest_status = "OVER BUDGET"

            elif highest_used >= 80:

                highest_status = "NEAR BUDGET LIMIT"

            else:

                highest_status = "WITHIN BUDGET"

            highest_category_context = (
                f"Category: {highest_category}\n"
                f"Spent: ₹{highest_category_amount:,.2f}\n"
                f"Budget: ₹{highest_budget:,.2f}\n"
                f"Remaining: ₹{highest_remaining:,.2f}\n"
                f"Budget usage: {highest_used:.1f}%\n"
                f"Status: {highest_status}"
            )

        else:

            highest_category_context = (
                f"Category: {highest_category}\n"
                f"Spent: ₹{highest_category_amount:,.2f}\n"
                f"Budget: NOT SET"
            )

    # Build final context

    context = f"""
SMARTSPEND FINANCIAL DATA
=========================

TOTAL SPENDING
--------------
₹{total:,.2f}


HIGHEST SPENDING CATEGORY
-------------------------
{highest_category_context}


AVERAGE DAILY SPENDING
----------------------
₹{average_daily:,.2f}


LARGEST TRANSACTION
-------------------
{largest_transaction_context}


CATEGORY SPENDING
-----------------
{categories}


CATEGORY-BUDGET CROSS REFERENCE
--------------------------------
{chr(10).join(category_budget_lines)
if category_budget_lines
else "No category data available."}


MONTHLY SPENDING
----------------
{monthly}


ALL BUDGET STATUS
-----------------
{budget_report}
"""

    return context


# AI FINANCIAL ADVISOR

def get_financial_advice(
    question,
    chat_history=None,
):
    """
    Answers financial questions using:

    1. Deterministic analytics
    2. Deterministic budget calculations
    3. Conversation history
    4. Groq for explanation and reasoning
    """

    # Initialize history

    if chat_history is None:

        chat_history = []

    # Get Groq client

    client = _get_client()

    # Build financial context

    context = build_financial_context()

    # Build current question prompt

    prompt = f"""
You are SmartSpend's AI Financial Advisor.

You are answering questions about the user's personal
financial data.

The application has already calculated the important
financial facts for you.

Your job is to:

- understand the user's question,
- use the supplied financial facts,
- use conversation history when interpreting follow-up
  questions,
- explain the answer clearly,
- provide practical suggestions when appropriate.

IMPORTANT RULES:

1. Treat the financial data below as the source of truth.

2. Do NOT invent amounts, categories, budgets, dates,
   transactions, or percentages.

3. Do NOT perform financial calculations if the necessary
   factual values are not available.

4. If a category has "Budget: NOT SET", explicitly say that
   no budget has been configured for that category.

5. If a user asks whether something was within budget,
   use the category-budget information provided below.

6. If the user refers to something with words such as
   "that", "it", "this", or "the previous one", use the
   conversation history to determine what they mean.

7. If the conversation history and current financial data
   are insufficient to determine the answer, say so instead
   of guessing.

8. Keep the answer concise, normally under 200 words.

9. Give practical recommendations, but do not present
   yourself as a licensed financial advisor.

FINANCIAL DATA
==============

{context}

CURRENT USER QUESTION
=====================

{question}
"""

    # Create Groq messages

    messages = [
        {
            "role": "system",
            "content": (
                "You are SmartSpend, a grounded personal "
                "finance assistant. "
                "The application calculates financial facts "
                "before calling you. "
                "Never invent missing financial information."
            ),}
    ]

    # Add conversation history

    messages.extend(
        chat_history
    )

    # Add current question

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Call Groq

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2,
    )

    # Return answer

    return response.choices[0].message.content


# PROACTIVE FINANCIAL INSIGHTS

def get_proactive_insights():
    """
    Detects important financial situations automatically.

    Python performs the factual detection first.
    Groq is then used to explain those findings.
    """

    categories = get_category_summary()
    budget_report = get_budget_status()
    total = get_total_spending()

    insights = []

    # No data

    if total <= 0 or not categories:
        return []

    # Budget-based insights

    for item in budget_report:

        category = str(
            item["category"]
        ).strip().lower()

        budget = float(
            item["budget"]
        )

        spent = float(
            item["spent"]
        )

        remaining = float(
            item["remaining"]
        )

        used = float(
            item["used"]
        )

        # Over budget
        if used >= 100:

            insights.append({
                "type": "danger",
                "category": category,
                "message": (
                    f"{category.title()} is over budget. "
                    f"Spent ₹{spent:,.2f} against a "
                    f"₹{budget:,.2f} budget."
                ),
            })

        # Near budget
        elif used >= 80:

            insights.append({
                "type": "warning",
                "category": category,
                "message": (
                    f"{category.title()} has used "
                    f"{used:.1f}% of its budget. "
                    f"₹{remaining:,.2f} remains."
                ),
            })

        # Very low usage
        elif used <= 40 and budget > 0:

            insights.append({
                "type": "positive",
                "category": category,
                "message": (
                    f"{category.title()} is well within budget. "
                    f"Only {used:.1f}% has been used."
                ),
            })

    # Highest spending category

    if categories:

        highest_category = max(
            categories,
            key=categories.get,
        )

        highest_amount = float(
            categories[highest_category]
        )

        percentage = (
            highest_amount / total
        ) * 100

        insights.append({
            "type": "info",
            "category": highest_category,
            "message": (
                f"{highest_category.title()} is your "
                f"highest spending category at "
                f"₹{highest_amount:,.2f}, representing "
                f"{percentage:.1f}% of total spending."
            ),
        })

    # ------------------------------------------------------
    # Largest transaction
    # ------------------------------------------------------

    largest = get_largest_transaction()

    if largest:

        insights.append({
            "type": "info",
            "category": largest["category"],
            "message": (
                f"Your largest individual transaction was "
                f"₹{largest['amount']:,.2f} in "
                f"{largest['category'].title()}."
            ),
        })

    return insights


# AI-EXPLAINED PROACTIVE INSIGHTS

def get_ai_insights():
    """
    Converts deterministic financial findings into concise,
    natural-language recommendations using Groq.
    """

    insights = get_proactive_insights()

    if not insights:
        return "There is not enough expense data to generate insights yet."

    client = _get_client()

    findings = "\n".join(
        f"- {item['message']}"
        for item in insights
    )

    prompt = f"""
You are SmartSpend's proactive financial assistant.

The application has already detected the following
financial facts:

{findings}

Create a concise financial insight summary for the user.

Rules:

1. Do not invent any information.
2. Use only the findings provided above.
3. Prioritize over-budget and near-budget situations.
4. Explain what deserves attention.
5. Give practical suggestions.
6. Do not recommend spending unused budget.
7. Mention actual amounts or percentages when provided.
8. Keep the answer under 180 words.
9. Use simple language.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are SmartSpend's proactive financial "
                    "assistant. Explain only the supplied "
                    "financial facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content

# FINANCIAL HEALTH SCORE

def calculate_financial_health():
    """
    Calculates a deterministic Financial Health Score from 0-100.

    The score considers:
    1. Budget adherence
    2. Budget coverage
    3. Spending concentration

    This is a project-level educational indicator, not
    professional financial advice.
    """

    categories = get_category_summary()
    budget_report = get_budget_status()
    total_spending = get_total_spending()

    # No expense data

    if total_spending <= 0 or not categories:

        return {
            "score": None,
            "rating": "Insufficient Data",
            "budget_adherence": None,
            "budget_coverage": None,
            "spending_concentration": None,
            "confidence": "Low",
            "reason": (
                "There is not enough expense data to "
                "calculate a meaningful score."
            ),
        }

    # 1. BUDGET ADHERENCE - 50 POINTS

    if budget_report:

        adherence_scores = []

        for item in budget_report:

            used = float(item["used"])

            if used <= 80:

                category_score = 100.0

            elif used <= 100:

                # Gradually reduce score between 80-100%
                category_score = (
                    100 - ((used - 80) * 2.5)
                )

            else:

                # Penalize overspending
                category_score = max(
                    0,
                    50 - ((used - 100) * 1.5)
                )

            adherence_scores.append(
                category_score
            )

        adherence_percentage = (
            sum(adherence_scores)
            / len(adherence_scores)
        )

    else:

        # No budgets configured.
        # We don't pretend the user has perfect control.
        adherence_percentage = 50.0

    adherence_points = (
        adherence_percentage * 0.50
    )

    # 2. BUDGET COVERAGE - 25 POINTS

    budgeted_categories = {
        str(item["category"]).strip().lower()
        for item in budget_report
    }

    all_categories = {
        str(category).strip().lower()
        for category in categories.keys()
    }

    if all_categories:

        coverage_percentage = (
            len(
                budgeted_categories
                & all_categories
            )
            / len(all_categories)
        ) * 100

    else:

        coverage_percentage = 0.0

    coverage_points = (
        coverage_percentage * 0.25
    )

    # 3. SPENDING CONCENTRATION - 25 POINTS

    highest_category_amount = max(
        categories.values()
    )

    concentration_percentage = (
        highest_category_amount
        / total_spending
    ) * 100

    if concentration_percentage <= 30:

        concentration_score = 100.0

    elif concentration_percentage <= 50:

        concentration_score = (
            100
            - (
                (concentration_percentage - 30)
                * 2
            )
        )

    elif concentration_percentage <= 70:

        concentration_score = max(
            0,
            60
            - (
                (concentration_percentage - 50)
                * 2
            ),
        )

    else:

        concentration_score = 20.0

    concentration_points = (
        concentration_score * 0.25
    )

    # FINAL SCORE

    score = round(
        adherence_points
        + coverage_points
        + concentration_points
    )

    score = max(
        0,
        min(score, 100)
    )

    # RATING

    if score >= 85:

        rating = "Excellent"

    elif score >= 70:

        rating = "Good"

    elif score >= 50:

        rating = "Fair"

    else:

        rating = "Needs Attention"

    # DATA CONFIDENCE

    transaction_count = sum(
        categories.values()
    )

    # Confidence is based mainly on whether budgets
    # and expense categories provide enough information.

    if (
        len(categories) >= 4
        and len(budget_report) >= 2
    ):

        confidence = "High"

    elif (
        len(categories) >= 2
        or len(budget_report) >= 1
    ):

        confidence = "Medium"

    else:

        confidence = "Low"

    # REASON

    if not budget_report:

        reason = (
            "No category budgets are configured, "
            "so budget-related parts of the score "
            "have limited evidence."
        )

    elif coverage_percentage < 50:

        reason = (
            "Only some spending categories have "
            "budgets configured."
        )

    elif concentration_percentage > 50:

        reason = (
            "A large share of spending is concentrated "
            "in one category."
        )

    else:

        reason = (
            "Spending is reasonably controlled across "
            "the available budget data."
        )

    return {
        "score": score,
        "rating": rating,
        "budget_adherence": round(
            adherence_percentage,
            1,
        ),
        "budget_coverage": round(
            coverage_percentage,
            1,
        ),
        "spending_concentration": round(
            concentration_percentage,
            1,
        ),
        "confidence": confidence,
        "reason": reason,
    }

# AI FINANCIAL HEALTH EXPLANATION

def get_financial_health_explanation():
    """
    Uses Groq to explain the deterministic Financial Health
    Score and provide practical improvement suggestions.
    """

    health = calculate_financial_health()

    if health["score"] is None:

        return (
            "There is not enough financial data yet to "
            "calculate a meaningful Financial Health Score."
        )

    client = _get_client()

    prompt = f"""
You are SmartSpend's Financial Health Assistant.

The application has already calculated the following
Financial Health information:

Score:
{health['score']}/100

Rating:
{health['rating']}

Budget Adherence:
{health['budget_adherence']}%

Budget Coverage:
{health['budget_coverage']}%

Spending Concentration:
{health['spending_concentration']}%

Confidence:
{health['confidence']}

Primary Reason:
{health['reason']}

Explain the result to the user.

Rules:

1. Do not change or recalculate the score.
2. Use only the information provided above.
3. Explain what the score means in simple language.
4. Mention the strongest area.
5. Mention the biggest area for improvement.
6. Give practical suggestions.
7. Do not invent financial information.
8. Do not describe this as a professional financial diagnosis.
9. Keep the response under 180 words.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain SmartSpend's "
                    "deterministic financial health score. "
                    "Never modify the supplied score."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# AI FORECAST EXPLANATION

def get_forecast_explanation(forecast_data):
    """
    Uses Groq to explain the deterministic month-end
    spending forecast.

    forecast_data must come from forecast_month_end().
    """

    if not forecast_data:
        return "Forecast information is not available."

    if forecast_data.get("status") != "success":

        message = forecast_data.get(
            "message",
            "Forecast unavailable.",
        )

        return message

    client = _get_client()

    current_spending = forecast_data[
        "current_spending"
    ]

    days_remaining = forecast_data[
        "days_remaining"
    ]

    forecast_remaining = forecast_data[
        "forecast_remaining"
    ]

    projected_total = forecast_data[
        "projected_total"
    ]

    monthly_budget = forecast_data[
        "monthly_budget"
    ]

    difference = forecast_data[
        "difference"
    ]

    budget_status = forecast_data[
        "budget_status"
    ]

    prompt = f"""
You are SmartSpend's AI Financial Assistant.

The application has already calculated the following
month-end forecast:

Current spending:
₹{current_spending:,.2f}

Days remaining:
{days_remaining}

Predicted additional spending:
₹{forecast_remaining:,.2f}

Projected month-end spending:
₹{projected_total:,.2f}

Monthly budget:
₹{monthly_budget:,.2f}

Projected difference from budget:
₹{difference:,.2f}

Budget status:
{budget_status}

Explain this forecast to the user.

Rules:

1. Do not change any of the numbers.
2. Do not invent information.
3. Explain whether the projected spending is within,
   near, or above the budget.
4. Give one or two practical suggestions.
5. Clearly mention that this is a forecast, not a guarantee.
6. Keep the explanation under 150 words.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain SmartSpend's deterministic "
                    "spending forecasts. Never modify the "
                    "supplied financial values."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
