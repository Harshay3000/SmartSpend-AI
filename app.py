import os

import pandas as pd
import streamlit as st

from utils import (
    hybrid_parse_expense,
    save_expense,
    summarize_expenses,
    clear_expenses,
    update_category,
)

from classifier import (
    can_train,
    train_classifier,
)

from budget import (
    load_budgets,
    set_budget,
    delete_budget,
    get_budget_status,
)

from dashboard import show_dashboard

from advisor import (
    get_financial_advice,
    get_ai_insights,
    calculate_financial_health,
    get_financial_health_explanation,
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="SmartSpend",
    page_icon="💸",
    layout="wide",
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================================
# HEADER
# ==========================================================

st.title("🧾 SmartSpend - AI Personal Finance Assistant")

st.caption(
    "LLM-powered expense extraction + ML categorization + "
    "analytics + budgeting + AI financial advisor"
)

st.write("Example: **I spent ₹500 on groceries**")


# ==========================================================
# EXPENSE INPUT
# ==========================================================

st.header("💳 Add Expense")

with st.form(
    key="expense_form",
    clear_on_submit=True,
):

    user_input = st.text_input(
        "💬 Describe your expense",
        placeholder="I spent ₹500 on groceries",
    )

    submitted = st.form_submit_button(
        "➕ Add Expense"
    )

    if submitted and user_input:

        # --------------------------------------------------
        # Summary request
        # --------------------------------------------------

        if (
            "summary" in user_input.lower()
            and "spent" not in user_input.lower()
        ):

            total, by_cat = summarize_expenses()

            st.info(
                f"📊 Total Spent: ₹{total:,.2f}"
            )

            if by_cat:

                st.write("### Category Breakdown")

                for cat, amount in by_cat.items():

                    st.write(
                        f"- **{cat.title()}**: "
                        f"₹{amount:,.2f}"
                    )

            else:

                st.write(
                    "No expenses found."
                )

        # --------------------------------------------------
        # Normal expense
        # --------------------------------------------------

        else:

            parsed = hybrid_parse_expense(
                user_input
            )

            if parsed:

                save_expense(parsed)

                source_label = {
                    "llm": "🤖 LLM",
                    "regex": "🔤 Regex fallback",
                    "classifier": "🧠 ML classifier",
                }

                category_source = source_label.get(
                    parsed.get("category_source"),
                    "Unknown",
                )

                confidence = parsed.get(
                    "category_confidence"
                )

                confidence_text = ""

                if confidence is not None:

                    confidence_text = (
                        f" ({confidence:.0%} confidence)"
                    )

                st.success(
                    f"✅ ₹{parsed['amount']:,.2f} added to "
                    f"**{parsed['category'].title()}** "
                    f"— category via {category_source}"
                    f"{confidence_text}"
                )

            else:

                st.warning(
                    "⚠️ I couldn't understand the expense. "
                    "Try: *I spent ₹500 on travel*"
                )


# ==========================================================
# QUICK ACTIONS
# ==========================================================

st.markdown("---")

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "📊 Show Total Spend",
        use_container_width=True,
    ):

        total, by_cat = summarize_expenses()

        st.info(
            f"📊 Total Spent: ₹{total:,.2f}"
        )

        if by_cat:

            st.write(
                "### Category-wise Breakdown"
            )

            for category, amount in by_cat.items():

                st.write(
                    f"- **{category.title()}**: "
                    f"₹{amount:,.2f}"
                )

        else:

            st.write(
                "No expenses found."
            )


with col2:

    if st.button(
        "🗑️ Clear All Expenses",
        use_container_width=True,
    ):

        clear_expenses()

        st.success(
            "🧹 All expense records have been cleared!"
        )


# ==========================================================
# BUDGET MANAGEMENT
# ==========================================================

st.markdown("---")

st.header("💰 Budget Management")


# ----------------------------------------------------------
# Budget Form
# ----------------------------------------------------------

with st.form(
    "budget_form"
):

    budget_category = st.selectbox(
        "Category",

        [
            "books",
            "dining",
            "entertainment",
            "fuel",
            "groceries",
            "health",
            "recharge",
            "shopping",
            "travel",
            "utilities",
            "other",
        ],
    )

    budget_amount = st.number_input(
        "Monthly Budget (₹)",

        min_value=0.0,

        step=100.0,
    )

    budget_submit = st.form_submit_button(
        "💾 Save Budget"
    )

    if budget_submit:

        if budget_amount <= 0:

            st.warning(
                "Budget must be greater than ₹0."
            )

        else:

            set_budget(
                budget_category,
                budget_amount,
            )

            st.success(
                f"✅ Budget for "
                f"**{budget_category.title()}** "
                f"set to ₹{budget_amount:,.2f}"
            )


# ----------------------------------------------------------
# Current Budgets
# ----------------------------------------------------------

budgets = load_budgets()


if budgets:

    st.subheader("📋 Current Budgets")

    for category, amount in budgets.items():

        col1, col2 = st.columns(
            [4, 1]
        )

        with col1:

            st.write(
                f"**{category.title()}** — "
                f"₹{amount:,.2f}"
            )

        with col2:

            if st.button(
                "Delete",
                key=f"delete_budget_{category}",
            ):

                delete_budget(
                    category
                )

                st.rerun()


# ----------------------------------------------------------
# Budget Status
# ----------------------------------------------------------

report = get_budget_status()


if report:

    st.subheader("📊 Budget Status")

    for item in report:

        category = item["category"]

        budget = item["budget"]

        spent = item["spent"]

        remaining = item["remaining"]

        used = item["used"]

        st.markdown(
            f"### {category.title()}"
        )

        progress = min(
            max(used / 100, 0),
            1.0,
        )

        st.progress(
            progress
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Budget",
                f"₹{budget:,.2f}",
            )

        with col2:

            st.metric(
                "Spent",
                f"₹{spent:,.2f}",
            )

        with col3:

            st.metric(
                "Remaining",
                f"₹{remaining:,.2f}",
            )

        if used < 70:

            st.success(
                f"🟢 {used:.1f}% of budget used"
            )

        elif used < 100:

            st.warning(
                f"🟡 {used:.1f}% of budget used"
            )

        else:

            st.error(
                f"🔴 {used:.1f}% used — "
                "Budget Exceeded!"
            )


else:

    st.info(
        "No budgets have been created yet."
    )


# ==========================================================
# ML CATEGORY CLASSIFIER
# ==========================================================

st.markdown("---")

st.subheader(
    "🧠 ML Category Classifier"
)

ready, message = can_train()

st.write(message)


if st.button(
    "🔁 Train / Retrain Classifier",
    disabled=not ready,
):

    try:

        stats = train_classifier()

        st.success(
            f"✅ Trained on "
            f"{stats['n_samples']} examples "
            f"across "
            f"{stats['n_categories']} categories: "
            f"{', '.join(stats['categories'])}"
        )

        if stats["cv_accuracy"] is not None:

            st.write(
                "Cross-validated accuracy estimate: "
                f"**{stats['cv_accuracy']:.1%}**"
            )

    except Exception as e:

        st.error(
            f"Training failed: {e}"
        )


# ==========================================================
# REVIEW / CORRECT RECENT EXPENSES
# ==========================================================

if (
    os.path.exists("expenses.csv")
    and os.path.getsize("expenses.csv") > 0
):

    df_recent = pd.read_csv(
        "expenses.csv"
    )

    if not df_recent.empty:

        df_recent = (
            df_recent
            .reset_index(drop=True)
        )

        with st.expander(
            "✏️ Review & Correct Recent Categories"
        ):

            recent = (
                df_recent
                .tail(10)
                .iloc[::-1]
            )

            for position, (
                index,
                row,
            ) in enumerate(
                recent.iterrows()
            ):

                c1, c2, c3 = st.columns(
                    [3, 2, 1]
                )

                c1.write(
                    f"₹{row['amount']:,.2f} — "
                    f"{row.get('note', '')}"
                )

                new_category = c2.text_input(
                    "Category",

                    value=str(
                        row["category"]
                    ),

                    key=f"category_{position}_{index}",

                    label_visibility="collapsed",
                )

                if c3.button(
                    "Save",

                    key=f"save_category_{position}_{index}",
                ):

                    update_category(
                        "expenses.csv",
                        index,
                        new_category,
                    )

                    st.success(
                        "✅ Category updated. "
                        "Retrain the classifier to "
                        "use this correction."
                    )

                    st.rerun()


# ==========================================================
# PROACTIVE AI INSIGHTS
# ==========================================================

st.markdown("---")

st.header("🔎 Smart Financial Insights")

try:

    with st.spinner(
        "🤖 Analyzing your spending patterns..."
    ):

        ai_insights = get_ai_insights()

    with st.container(border=True):

        st.markdown(ai_insights)

except Exception as e:

    st.warning(
        f"Unable to generate smart insights right now: {e}"
    )

# ==========================================================
# FINANCIAL HEALTH SCORE
# ==========================================================

st.markdown("---")

st.header("❤️ Financial Health Score")

try:

    health = calculate_financial_health()

    if health["score"] is None:

        st.info(
            health["reason"]
        )

    else:

        # --------------------------------------------------
        # Score display
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Financial Health",
                f"{health['score']}/100",
            )

        with col2:

            st.metric(
                "Rating",
                health["rating"],
            )

        with col3:

            st.metric(
                "Confidence",
                health["confidence"],
            )

        # --------------------------------------------------
        # Progress bar
        # --------------------------------------------------

        st.progress(
            health["score"] / 100
        )

        # --------------------------------------------------
        # Component breakdown
        # --------------------------------------------------

        st.subheader(
            "📊 Score Breakdown"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Budget Adherence",
                f"{health['budget_adherence']:.1f}%",
            )

        with col2:

            st.metric(
                "Budget Coverage",
                f"{health['budget_coverage']:.1f}%",
            )

        with col3:

            st.metric(
                "Top Category Share",
                f"{health['spending_concentration']:.1f}%",
            )

        # --------------------------------------------------
        # Explanation
        # --------------------------------------------------

        with st.spinner(
            "🤖 Explaining your Financial Health Score..."
        ):

            explanation = (
                get_financial_health_explanation()
            )

        with st.container(
            border=True
        ):

            st.subheader(
                "🧠 What Your Score Means"
            )

            st.markdown(
                explanation
            )

except Exception as e:

    st.warning(
        f"Unable to calculate Financial Health Score: {e}"
    )

# ==========================================================
# ANALYTICS DASHBOARD
# ==========================================================

st.markdown("---")

show_dashboard()


# ==========================================================
# AI FINANCIAL ADVISOR
# ==========================================================

st.markdown("---")

st.header(
    "🤖 AI Financial Advisor"
)

st.caption(
    "Ask questions about your spending, budgets, "
    "and financial habits."
)


# ----------------------------------------------------------
# Suggested Questions
# ----------------------------------------------------------

st.subheader(
    "💡 Suggested Questions"
)

suggestions = [

    "How much did I spend this month?",

    "Which category should I reduce?",

    "Am I overspending?",

    "Summarize my expenses.",

    "How can I save more money?",

]


selected_question = st.selectbox(
    "Choose a question",

    [
        "",
        *suggestions,
    ],
)


# ----------------------------------------------------------
# Conversation History
# ----------------------------------------------------------

if st.session_state.chat_history:

    st.subheader(
        "💬 Conversation"
    )

    for message in (
        st.session_state.chat_history
    ):

        if message["role"] == "user":

            with st.chat_message(
                "user"
            ):

                st.write(
                    message["content"]
                )

        elif message["role"] == "assistant":

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    message["content"]
                )


# ----------------------------------------------------------
# Question Input
# ----------------------------------------------------------

default_question = selected_question


question = st.text_area(

    "Ask anything about your finances",

    value=default_question,

    height=120,

    placeholder=(
        "Example: Which category is "
        "causing me to overspend?"
    ),

)


# ----------------------------------------------------------
# AI Actions
# ----------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    ask_ai = st.button(
        "🚀 Ask AI",
        use_container_width=True,
    )


with col2:

    clear_chat = st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    )


# ----------------------------------------------------------
# Clear Conversation
# ----------------------------------------------------------

if clear_chat:

    st.session_state.chat_history = []

    st.rerun()


# ----------------------------------------------------------
# Ask AI
# ----------------------------------------------------------

if ask_ai:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "🤖 Analyzing your finances..."
        ):

            try:

                answer = get_financial_advice(
                    question,
                    st.session_state.chat_history,
                )

                # ------------------------------------------
                # Store user message
                # ------------------------------------------

                st.session_state.chat_history.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                # ------------------------------------------
                # Store AI response
                # ------------------------------------------

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                # ------------------------------------------
                # Limit memory
                # ------------------------------------------

                MAX_MESSAGES = 20

                if (
                    len(
                        st.session_state.chat_history
                    )
                    > MAX_MESSAGES
                ):

                    st.session_state.chat_history = (
                        st.session_state.chat_history[
                            -MAX_MESSAGES:
                        ]
                    )

                st.success(
                    "✅ Analysis Complete"
                )

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "🤖 AI Response"
                    )

                    st.markdown(
                        answer
                    )

            except Exception as e:

                st.error(
                    f"AI Error: {e}"
                )


# ==========================================================
# DOWNLOAD EXPENSE CSV
# ==========================================================

st.markdown("---")

if (
    os.path.exists("expenses.csv")
    and os.path.getsize("expenses.csv") > 0
):

    with open(
        "expenses.csv",
        "rb",
    ) as file:

        st.download_button(

            label="⬇️ Download Expenses CSV",

            data=file,

            file_name="my_expenses.csv",

            mime="text/csv",

            use_container_width=True,
        )

else:

    st.info(
        "No expense data available for download."
    )

