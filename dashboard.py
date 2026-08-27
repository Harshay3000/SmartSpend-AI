import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from analytics import get_cumulative_spending
from analytics import (
    get_total_spending,
    get_transaction_count,
    get_average_daily_spending,
    get_highest_category,
    get_largest_transaction,
    get_this_week_spending,
    get_category_summary,
    get_monthly_spending,
    get_weekday_spending,
    get_spending_trend,
    get_recent_transactions
)
from budget import (
    get_budget_status,
    get_over_budget_categories
)
from forecast import (
    forecast_month_end,
    forecast_next_days_improved,
    get_forecast_confidence,
)

from advisor import get_forecast_explanation

def style_chart(fig, ax):
    """
    Styles Matplotlib charts to match SmartSpend's dark UI.
    """

    # Transparent figure and axes
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # Title
    ax.title.set_color("#F5F7FA")
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")

    # Axis labels
    ax.xaxis.label.set_color("#D6D9E0")
    ax.yaxis.label.set_color("#D6D9E0")

    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)

    # Tick labels
    ax.tick_params(
        axis="both",
        colors="#C9CDD4",
        labelsize=9,
    )

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")

    ax.spines["left"].set_alpha(0.6)
    ax.spines["bottom"].set_alpha(0.6)

    # Grid
    ax.grid(
        axis="y",
        color="#64748B",
        alpha=0.18,
        linestyle="--",
    )

    # Legend
    legend = ax.get_legend()

    if legend is not None:

        legend.get_frame().set_facecolor(
            "#161B22"
        )

        legend.get_frame().set_edgecolor(
            "#374151"
        )

        for text in legend.get_texts():
            text.set_color("#F5F7FA")

    # Compact layout
    fig.tight_layout(
        pad=0.8
    )

    return fig

# KPI SECTION
def show_kpis():

    total = get_total_spending()
    transactions = get_transaction_count()
    average = get_average_daily_spending()
    top_category = get_highest_category()
    largest = get_largest_transaction()
    weekly = get_this_week_spending()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 Total Spending", f"₹{total:,.2f}")

    with col2:
        st.metric("📄 Transactions", transactions)

    with col3:
        st.metric("📈 Avg / Day", f"₹{average:,.2f}")

    col4, col5, col6 = st.columns(3)

    with col4:
        if top_category:
            st.metric("🏆 Top Category", top_category[0].title())

    with col5:
        if largest:
            st.metric("💸 Largest Expense", f"₹{largest['amount']:,.2f}")

    with col6:
        st.metric("📅 This Week", f"₹{weekly:,.2f}")


# CATEGORY CHARTS

def show_category_charts():

    st.subheader("📂 Category Analysis")

    categories = get_category_summary()

    if not categories:
        st.info("No category data available.")
        return

    labels = list(categories.keys())
    values = list(categories.values())

    col1, col2 = st.columns(2)

    # ------------------------------------------------------
    # Pie Chart
    # ------------------------------------------------------

    with col1:

        fig, ax = plt.subplots(
        figsize=(5, 3.8)
)

        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            textprops={
                "color": "#F5F7FA",
                "fontsize": 9,
            },
        )

        for text in texts:
            text.set_color("#F5F7FA")

        for autotext in autotexts:
            autotext.set_color("#FFFFFF")
            autotext.set_fontweight("bold")

        ax.set_title(
            "Category Distribution",
            color="#F5F7FA",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )

        fig.tight_layout(
            pad=0.8
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # ------------------------------------------------------
    # Horizontal Bar Chart
    # ------------------------------------------------------

    with col2:
        fig, ax = plt.subplots(
    figsize=(6, 3.8)
)

    ax.barh(
        labels,
        values
    )

    ax.set_title(
        "Category Spending"
    )

    ax.set_xlabel(
        "Amount (₹)"
    )

    # Horizontal grid
    ax.grid(
        axis="x",
        color="#64748B",
        alpha=0.18,
        linestyle="--",
    )

    style_chart(
        fig,
        ax
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    
# TIME ANALYSIS

def show_time_charts():

    st.subheader("📅 Time Analysis")

    col1, col2 = st.columns(2)

    # MONTHLY SPENDING

    with col1:

        monthly = get_monthly_spending()

        if monthly:

            months = [
                str(month)
                for month in monthly.keys()
            ]

            values = list(
                monthly.values()
            )

            fig, ax = plt.subplots(
                figsize=(6, 3.5)
            )

            ax.bar(
                months,
                values
            )

            ax.set_title(
                "Monthly Spending"
            )

            ax.set_ylabel(
                "Amount (₹)"
            )

            style_chart(
                fig,
                ax
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        else:

            st.info(
                "No monthly data available."
            )

    # WEEKDAY SPENDING

    with col2:

        weekday = get_weekday_spending()

        if weekday:

            days = list(
                weekday.keys()
            )

            values = list(
                weekday.values()
            )

            fig, ax = plt.subplots(
            figsize=(6, 3.5)
        )

            ax.bar(
                days,
                values
            )

            ax.set_title(
                "Weekday Spending"
            )

            ax.set_ylabel(
                "Amount (₹)"
            )

            ax.tick_params(
                axis="x",
                rotation=30,
            )

            style_chart(
                fig,
                ax
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:

            st.info(
                "No weekday data available."
            )

    # DAILY TREND

    trend = get_spending_trend()

    if not trend.empty:

        st.markdown("#### 📈 Daily Spending Trend")

        fig, ax = plt.subplots(
        figsize=(10, 3.8)
)

        ax.plot(
            trend["date"],
            trend["amount"],
            marker="o",
            linewidth=2,
            markersize=5,
        )

        ax.set_title(
            "Daily Spending Trend"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Amount (₹)"
        )

        ax.grid(
            color="#64748B",
            alpha=0.18,
            linestyle="--",
        )

        style_chart(
            fig,
            ax
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    else:

        st.info(
            "No trend data available."
        )


# INSIGHTS

def show_insights():

    st.subheader("🧠 Quick Insights")

    category = get_highest_category()
    largest = get_largest_transaction()
    average = get_average_daily_spending()

    if category is None or largest is None:
        st.info("No insights available.")
        return

    summary = get_category_summary()

    total = get_total_spending()

    largest_category = max(summary, key=summary.get)

    largest_percent = (
        summary[largest_category] / total
    ) * 100

    st.success(
    f"""
    ### Financial Summary

    ✅ You have spent **₹{total:,.2f}**

    🏆 {largest_category.title()} contributes **{largest_percent:.1f}%** of all expenses.

    💰 Largest transaction: ₹{largest['amount']:,.2f}

    📈 Average daily spending: ₹{average:,.2f}
    """
    )

# MAIN DASHBOARD

def show_dashboard():

    st.header("📊 Analytics Dashboard")

    # KPI SECTION

    show_kpis()

    st.divider()

    # VISUALIZATION MENU

    st.subheader("📈 Explore Your Analytics")

    col1, col2, col3 = st.columns(3)

    # Budget Analysis

    with col1:

        with st.popover(
            "💰 Budget Analysis",
            use_container_width=True,
        ):

            st.subheader(
                "💰 Budget Analysis"
            )

            show_budget_dashboard()

    # Category Analysis

    with col2:

        with st.popover(
            "📂 Category Analysis",
            use_container_width=True,
        ):

            show_category_charts()

    # Time Analysis
    

    with col3:

        with st.popover(
            "📅 Time Analysis",
            use_container_width=True,
        ):

            show_time_charts()

    # SECOND ROW

    col1, col2, col3 = st.columns(3)

    # Spending Trend

    with col1:

        with st.popover(
            "📈 Spending Trend",
            use_container_width=True,
        ):

            st.subheader(
                "📈 Spending Trend"
            )

            trend = get_spending_trend()

            if trend.empty:

                st.info(
                    "No trend data available."
                )

            else:

                fig, ax = plt.subplots(
                    figsize=(8, 3.5)
                )

                ax.plot(
                    trend["date"],
                    trend["amount"],
                    marker="o",
                    linewidth=2,
                    markersize=5,
                )

                ax.set_title(
                    "Daily Spending Trend"
                )

                ax.set_xlabel(
                    "Date"
                )

                ax.set_ylabel(
                    "Amount (₹)"
                )

                style_chart(
                    fig,
                    ax
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

    # ------------------------------------------------------
    # Forecast
    # ------------------------------------------------------

    with col2:

        with st.popover(
            "🔮 Spending Forecast",
            use_container_width=True,
        ):

            show_forecast()

    # ------------------------------------------------------
    # Cumulative Spending
    # ------------------------------------------------------

    with col3:

        with st.popover(
            "📊 Cumulative Spending",
            use_container_width=True,
        ):

            show_cumulative_chart()

    # TABLES

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        with st.expander(
            "🏆 Top Categories"
        ):

            show_top_categories()

    with col2:

        with st.expander(
            "🧾 Recent Transactions"
        ):

            show_recent_transactions()

    # INSIGHTS

    with st.expander(
        "🧠 Smart Insights"
    ):

        show_insights()


def show_cumulative_chart():

    st.subheader("📈 Running Spending")
    data = get_cumulative_spending()

    if data.empty:

        st.info("No cumulative data available.")

        return

    fig, ax = plt.subplots(figsize=(10,3.5))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        alpha=0.12,
        linestyle="--",
    )

    ax.tick_params(
        labelsize=9
    )

    fig.tight_layout(
        pad=0.8
    )

    ax.plot(
        data["date"],
        data["cumulative"],
        marker="o",
        linewidth=3
    )

    ax.set_title("Cumulative Spending")

    ax.set_ylabel("Amount (₹)")

    plt.xticks(rotation=30)

    st.pyplot(fig)

def show_top_categories():
    summary = get_category_summary()

    if not summary:

        return
    total = sum(summary.values())
    rows = []

    for category, amount in summary.items():

        percentage = (amount / total) * 100

        rows.append({

            "Category": category.title(),

            "Spent (₹)": amount,

            "%": round(percentage,2)

        })
    st.subheader("🏅 Top Categories")

    st.dataframe(
        rows,
        use_container_width=True
    )

def show_recent_transactions():

    st.subheader("🧾 Recent Transactions")

    recent = get_recent_transactions()

    st.dataframe(
        recent,
        use_container_width=True
    )    

def show_budget_dashboard():

    st.subheader("💰 Budget Overview")
    report = get_budget_status()

    if not report:

        st.info("No budgets available.")

        return
    col1, col2, col3 = st.columns(3)

    total_budget = sum(
    item["budget"]
    for item in report
)

    total_spent = sum(
        item["spent"]
        for item in report
    )

    remaining = total_budget - total_spent

    with col1:

        st.metric(
            "Total Budget",
            f"₹{total_budget:,.0f}"
        )

    with col2:

        st.metric(
            "Spent",
            f"₹{total_spent:,.0f}"
        )

    with col3:

        st.metric(
            "Remaining",
            f"₹{remaining:,.0f}"
        )

    categories = [
    item["category"].title()
    for item in report
    ]

    budgets = [
        item["budget"]
        for item in report
    ]

    spent = [
        item["spent"]
        for item in report
    ]
    fig, ax = plt.subplots(
    figsize=(8, 3.8)
)

    x = range(len(categories))

    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        budgets,
        width,
        label="Budget",
    )

    ax.bar(
        [i + width / 2 for i in x],
        spent,
        width,
        label="Spent",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        categories
    )

    ax.set_title(
        "Budget vs Actual Spending"
    )

    ax.set_ylabel(
        "Amount (₹)"
    )

    style_chart(
        fig,
        ax
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    alerts = get_over_budget_categories()
    if alerts:

        st.error(
            "🚨 Overspending Detected"
        )

        for item in alerts:

            st.write(
                f"• {item['category'].title()} "
                f"({item['used']:.1f}%)"
            )
    else:

        st.success(
            "🎉 All budgets are within limits."
        )
    ranking = sorted(
    report,
    key=lambda x: x["used"],
    reverse=True
)
    st.subheader("🏆 Budget Usage Ranking")

    st.dataframe(
        ranking,
        use_container_width=True
    )

    st.subheader("💡 Recommendations")
    for item in report:

        used = item["used"]

        category = item["category"].title()
    if used >= 100:

        st.error(
            f"Reduce spending in {category}. "
            "Budget exceeded."
        )

    elif used >= 80:

        st.warning(
            f"Careful! {category} "
            "is approaching its limit."
        )

    else:

        st.success(
            f"{category} spending "
            "is well under budget."
        )

# SPENDING FORECAST

def show_forecast():

    st.subheader("📈 Spending Forecast")

    # Get month-end forecast

    result = forecast_month_end()

    # Insufficient data

    if result.get("status") == "insufficient_data":

        st.info(
            "📊 Not enough historical data yet to "
            "generate a reliable spending forecast."
        )

        confidence = get_forecast_confidence()

        st.caption(
            f"Current forecast confidence: {confidence}"
        )

        return

    # No budget

    if result.get("status") == "no_budget":

        st.warning(
            "A forecast is available, but no total "
            "monthly budget is configured."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Current Spending",
                f"₹{result['current_spending']:,.2f}",
            )

        with col2:

            st.metric(
                "Forecast Remaining",
                f"₹{result['forecast_remaining']:,.2f}",
            )

        with col3:

            st.metric(
                "Projected Month-End",
                f"₹{result['projected_total']:,.2f}",
            )

        return

    # Successful forecast

    current = result[
        "current_spending"
    ]

    remaining = result[
        "forecast_remaining"
    ]

    projected = result[
        "projected_total"
    ]

    budget = result[
        "monthly_budget"
    ]

    difference = result[
        "difference"
    ]

    status = result[
        "budget_status"
    ]

    # KPI cards

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Spending",
            f"₹{current:,.0f}",
        )

    with col2:

        st.metric(
            "Forecast Remaining",
            f"₹{remaining:,.0f}",
        )

    with col3:

        st.metric(
            "Projected Total",
            f"₹{projected:,.0f}",
        )

    with col4:

        st.metric(
            "Monthly Budget",
            f"₹{budget:,.0f}",
        )

    # Budget utilization forecast

    projected_usage = (
        projected / budget
        if budget > 0
        else 0
    )

    projected_usage = max(
        0,
        projected_usage,
    )

    st.write(
        f"**Projected budget usage:** "
        f"{projected_usage:.1%}"
    )

    st.progress(
        min(
            projected_usage,
            1.0,
        )
    )

    # Status message

    if status == "AT_RISK":

        st.error(
            f"🚨 Forecast suggests you may exceed "
            f"your budget by approximately "
            f"₹{difference:,.2f}."
        )

    elif status == "NEAR_LIMIT":

        st.warning(
            f"⚠️ Your projected spending is close "
            f"to the monthly budget."
        )

    else:

        st.success(
            f"✅ Your projected spending is "
            f"within the monthly budget."
        )

    # Forecast chart

    forecast = forecast_next_days_improved(
        days=result["days_remaining"]
    )

    if not forecast.empty:

        chart_data = forecast.copy()

        chart_data["date"] = pd.to_datetime(
            chart_data["date"]
        )

        chart_data = chart_data.set_index(
            "date"
        )

        st.write(
            "### 🔮 Predicted Daily Spending"
        )

        st.line_chart(
            chart_data[
                ["predicted_amount"]
            ],
            x_label="Date",
            y_label="Predicted Spending (₹)",
        )

    # AI explanation

    try:

        with st.spinner(
            "🤖 Explaining the forecast..."
        ):

            explanation = (
                get_forecast_explanation(
                    result
                )
            )

        with st.container(
            border=True
        ):

            st.subheader(
                "🧠 What This Forecast Means"
            )

            st.markdown(
                explanation
            )

    except Exception as e:

        st.warning(
            f"Unable to generate AI explanation: {e}"
        )

    # Forecast confidence

    confidence = (
        get_forecast_confidence()
    )

    st.caption(
        f"Forecast confidence: **{confidence}**. "
        "This is an estimate based on historical spending "
        "patterns, not a guarantee."
    )