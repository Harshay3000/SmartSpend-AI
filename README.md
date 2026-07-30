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
```
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

Historical Expenses ➔ Daily Aggregation ➔ Feature Engineering ➔ Baseline Model ➔ Improved Model ➔ Time-Aware Validation ➔ Model Comparison ➔ Recursive Forecasting ➔ Month-End Projection

* **Engineered Features:** Day of week, weekend indicators, day of month, month, previous-day/week lag spending, and 7-day rolling averages.
* **Data Safeguards:** Automatically detects insufficient historical data to prevent misleading or erratic projections.

---

## 🧠 Machine Learning Approach

### Expense Categorization
Expense Note ➔ TF-IDF ➔ Logistic Regression ➔ Category Prediction

* **Similarity Guard:** Prevents the classifier from assigning high-confidence predictions to unfamiliar or out-of-distribution notes.

### Spending Forecasting
* **Baseline vs. Improved:** Evaluated against a basic Linear Regression baseline by adding temporal and lag-based features.
* **Time-Aware Validation:** Uses temporal split validation (`TimeSeriesSplit`) rather than random shuffling to respect time sequence integrity.
* **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

---

## 🛡️ Reliability & Safety

SmartSpend uses a **deterministic-first design principle**—core financial calculations are strictly handled by Python, utilizing the LLM exclusively for parsing, explanation, and conversational context.

Key safeguards include:
* **Pydantic Validation:** Strict schema enforcement on LLM outputs.
* **Regex Fallback:** Automatic failover parsing if the LLM fails or is offline.
* **ML Safeguards:** Confidence thresholds and TF-IDF similarity checks.
* **Data Guardrails:** Sufficient-data checks for forecasts, empty-data handling, and budget validation.
* **Robust Error Handling:** Resilient execution against API timeouts and outages.

---

## 🛠️ Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Language** | Python |
| **Frontend** | Streamlit |
| **Data Science** | Pandas, NumPy, Matplotlib, Scikit-learn |
| **Machine Learning** | TF-IDF, Logistic Regression, Linear Regression, TimeSeriesSplit (MAE, RMSE, R²) |
| **Generative AI** | Groq API, Llama 3.3, Pydantic, Prompt Engineering |
| **Storage** | CSV, JSON |

## 📁 Project Structure

```text
SmartSpend-AI/
│
├── app.py
├── advisor.py
├── analytics.py
├── budget.py
├── classifier.py
├── dashboard.py
├── forecast.py
├── llm_parser.py
├── utils.py
│
├── test_forecast.py
│
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
│
└── screenshots/
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/SmartSpend-AI.git
cd SmartSpend-AI
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it depending on your operating system:

* **Windows:**
  ```cmd
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key
Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Security Note:** Never commit your `.env` file to public source control.

## ▶️ Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Testing

Forecasting components can be tested independently:

```bash
python test_forecast.py
```

The project also includes safeguards for insufficient historical data.

---

## 📊 Example Workflow

```text
"I spent ₹800 on groceries"
             ↓
       Groq extraction
             ↓
      Pydantic validation
             ↓
      ML categorization
             ↓
         Save expense
             ↓
       Analytics engine
             ↓
     Budget + Dashboard
             ↓
       AI Financial Advisor
```

---

## 🔐 Environment Variables

Create a local `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Note:** The actual `.env` file is intentionally excluded from the repository.

---

## ⚠️ Current Limitations

* **Data Requirements:** Forecasting requires sufficient historical data before producing predictions.
* **Metric Design:** The Financial Health Score is a project-designed heuristic.
* **Storage Layer:** The current application uses CSV/JSON storage rather than a production database.
* **Cold Start:** Forecast accuracy will naturally improve as more real historical spending data becomes available.
* **Disclaimer:** The project is intended for educational and portfolio purposes, not professional financial advice.

---

## 🔮 Future Improvements

Planned improvements include:

* **Detection & Analytics:** Anomaly detection for unusual expenses and automated monthly or PDF financial reports.
* **Machine Learning:** Better time-series models and category-level spending forecasts.
* **Database & Auth:** SQLite/PostgreSQL storage, authentication, and bank statement expense import.
* **DevOps & Infrastructure:** Cloud deployment, Docker support, and automated testing pipelines.

---

## 👨‍💻 Author

**Harshay Chouhan**  
Data Science & AI Student

---

## 📄 License

This project is intended as a portfolio and educational project.
