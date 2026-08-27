import sqlite3
import pandas as pd
from datetime import datetime

DATABASE_NAME = "smartspend.db"

def normalize_category(category):
    """
    Normalize category names so that variations such as
    'health', 'HEALTH', and ' health ' are stored consistently.
    """

    if category is None:
        return ""

    return str(category).strip().title()
# CONNECTION

def get_connection():
    """
    Returns a SQLite connection.
    """

    conn = sqlite3.connect(DATABASE_NAME)

    conn.row_factory = sqlite3.Row

    return conn


# DATABASE INITIALIZATION

def initialize_database():

    with get_connection() as conn:

        cursor = conn.cursor()

        # Expenses Table

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date TEXT NOT NULL,

            amount REAL NOT NULL,

            category TEXT NOT NULL,

            note TEXT,

            parser_used TEXT,

            category_source TEXT,

            category_confidence REAL,

            created_at TEXT NOT NULL,

            updated_at TEXT NOT NULL

        )
        """)

        # Budgets Table

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS budgets(

            category TEXT PRIMARY KEY,

            amount REAL NOT NULL,

            created_at TEXT NOT NULL,

            updated_at TEXT NOT NULL

        )
        """)

        conn.commit()


# EXPENSE CRUD

def add_expense(
    date,
    amount,
    category,
    note,
    parser_used,
    category_source,
    category_confidence,
):

    category = normalize_category(category)
    now = datetime.now().isoformat()

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO expenses(

            date,

            amount,

            category,

            note,

            parser_used,

            category_source,

            category_confidence,

            created_at,

            updated_at

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            date,

            amount,

            category,

            note,

            parser_used,

            category_source,

            category_confidence,

            now,

            now

        ))

        conn.commit()


def get_all_expenses():

    with get_connection() as conn:

        df = pd.read_sql_query("""

            SELECT *

            FROM expenses

            ORDER BY date DESC, id DESC

        """, conn)

    return df


def get_expense(expense_id):

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""

            SELECT *

            FROM expenses

            WHERE id = ?

        """, (expense_id,))

        row = cursor.fetchone()

    return row

def update_expense_category(expense_id, category):
    """
    Update the category of an existing expense.
    """

    category = normalize_category(category)
    now = datetime.now().isoformat()

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            UPDATE expenses
            SET
                category = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            category,
            now,
            expense_id
        ))

        conn.commit()

def clear_expenses():

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""

            DELETE FROM expenses

        """)

        conn.commit()

# BUDGET CRUD

def save_budget(category, amount):
    """
    Insert or update a budget.
    """

    category = normalize_category(category)
    now = datetime.now().isoformat()

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""

        INSERT INTO budgets(
            category,
            amount,
            created_at,
            updated_at
        )

        VALUES (?, ?, ?, ?)

        ON CONFLICT(category)

        DO UPDATE SET

            amount = excluded.amount,

            updated_at = excluded.updated_at

        """, (

            category,

            amount,

            now,

            now

        ))

        conn.commit()

def get_budget(category):

    category = normalize_category(category)

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""

            SELECT *

            FROM budgets

            WHERE LOWER(category) = LOWER(?)

        """, (category,))

        return cursor.fetchone()

def get_all_budgets():

    with get_connection() as conn:

        df = pd.read_sql_query("""

            SELECT *

            FROM budgets

            ORDER BY category

        """, conn)

    return df

def delete_budget(category):

    category = normalize_category(category)

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM budgets
            WHERE LOWER(category) = LOWER(?)
        """, (category,))

        conn.commit()

def clear_budgets():

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""

            DELETE FROM budgets

        """)

        conn.commit()

def get_total_expenses():

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""

            SELECT COUNT(*)

            FROM expenses

        """)

        return cursor.fetchone()[0]

def get_total_spending():

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""

            SELECT SUM(amount)

            FROM expenses

        """)

        result = cursor.fetchone()[0]

    return result if result else 0

def get_categories():

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""

            SELECT DISTINCT category

            FROM expenses

            ORDER BY category

        """)

        rows = cursor.fetchall()

    return [row[0] for row in rows]

def budgets_as_dict():
    """
    Returns budgets as:

    {
        "fuel": 2500,
        "books": 1500
    }
    """

    df = get_all_budgets()

    if df.empty:
        return {}

    return dict(zip(df["category"], df["amount"]))