# 💸 SmartSpend AI

> An AI-powered personal finance assistant combining LLMs, machine learning, data analytics, budgeting, conversational AI, and spending forecasting.

SmartSpend AI allows users to record expenses naturally, automatically categorize them, analyze spending patterns, create budgets, receive AI-powered financial insights, and interact with a conversational financial advisor.

---

## 🚀 What makes SmartSpend different?

SmartSpend is not just a CRUD-based expense tracker.

It combines several AI and Data Science components into one application:

- 🤖 Groq-powered natural-language expense extraction
- 🧠 ML-based expense categorization
- 📊 Financial analytics and interactive dashboard
- 💰 Category-based budget management
- 🔎 Proactive financial insights
- ❤️ Financial Health Score
- 💬 Conversational AI Financial Advisor
- 📈 Machine-learning spending forecasting
- ⚠️ Forecast vs. budget analysis

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │       User           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         Natural Language Input
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │       Groq LLM            │
                    │   Expense Extraction      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Pydantic Validation     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         Expense Data Storage
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            Analytics Engine              ML Classifier
                    │                           │
                    ▼                           ▼
             Budget Engine              Category Prediction
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         Financial Analytics
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
        Dashboard          AI Financial Advisor   Forecasting
             │                    │                    │
             │                    ▼                    ▼
             │                  Groq                ML Model
             │                    │                    │
             └────────────────────┴────────────────────┘
                                  │
                                  ▼
                         User-facing Insights

## ✨ Features

### 1. Natural-Language Expense Entry
Users can enter expenses naturally:

> *"I spent ₹500 on coffee"*

The application extracts:
* **Amount:** ₹500
* **Category:** Dining
* **Note:** Coffee purchase
* **Date:** Current date

*Note: If the Groq API is unavailable, the application gracefully falls back to a regex-based parser.*

---

### 2. ML Expense Categorization
SmartSpend uses a **TF-IDF + Logistic Regression** classifier trained on the user's historical expense notes. 

The classifier only overrides the LLM/regex category when its prediction is sufficiently confident and similar to previous training examples.

---

### 3. Financial Analytics
The analytics engine provides detailed insights including:

* **Key Metrics:** Total spending, transaction count, average daily spending, highest spending category, and largest transaction.
* **Breakdowns:** Daily, monthly, weekday, and cumulative spending patterns.
* **Summaries:** Category-level spending overviews.

---

### 4. Budget Management
Users can set custom category budgets:

| Category | Budget |
| :--- | :--- |
| **Groceries** | ₹5,000 |
| **Fuel** | ₹2,500 |
| **Books** | ₹1,500 |
| **Entertainment** | ₹3,000 |

SmartSpend automatically tracks and calculates:
* Budget usage percentage
* Total amount spent
* Remaining budget
* Real-time budget status and overspending alerts

---

### 5. AI Financial Advisor
Ask natural, conversational questions about your money, such as:
* *"How much did I spend this month?"*
* *"Which category should I reduce?"*
* *"Am I overspending?"*
* *"How can I save more money?"*

The advisor synthesizes **Analytics + Budget Data + Conversation History + Groq** to deliver personalized, contextual advice.

---

### 6. Conversational Memory
The financial advisor remembers context across turns so you can ask natural follow-up questions:

> **User:** Which category had the highest spending?  
> **AI:** Dining.  
> **User:** How can I reduce that?  
> **AI:** Dining is your highest spending category...

*Conversation history is persisted seamlessly via Streamlit session state.*

---

### 7. Proactive Financial Insights
SmartSpend automatically identifies key financial patterns and triggers:
* Categories exceeding or approaching budget limits
* Highest spending categories and unusually large transactions
* Active categories missing a configured budget

> **Architecture Note:** Core financial calculations are processed deterministically in Python, while the Groq API generates clear, natural-language insights from the results.

---

### 8. Financial Health Score
SmartSpend computes an application-specific **0–100 Financial Health Score** based on key behaviors:
* **Budget Adherence:** Staying within set limits
* **Budget Coverage:** Percentage of spending covered by active budgets
* **Spending Concentration:** Tracking over-reliance on single spending categories

Each score comes with a natural-language breakdown and a confidence indicator.

> *Disclaimer: The Financial Health Score is an educational metric designed for this application and does not constitute a professional financial assessment.*

---

### 9. Spending Forecasting
SmartSpend features an end-to-end machine learning forecasting pipeline:

```text
Historical Expenses ➔ Daily Aggregation ➔ Feature Engineering ➔ Baseline Model
        ➔ Improved Model ➔ Time-Aware Validation ➔ Model Comparison 
        ➔ Recursive Forecasting ➔ Month-End Projection
