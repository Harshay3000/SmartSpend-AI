import re
import os
import logging
from datetime import datetime
import pandas as pd
import dateparser

from database import (
    add_expense,
    get_all_expenses,
    clear_expenses as db_clear_expenses,
    normalize_category
)

from llm_parser import ask_llm
from classifier import predict_category, CONFIDENCE_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "date", "amount", "category", "note",
    "parser_used", "category_source", "category_confidence",
]


def extract_date(text):
    parsed_date = dateparser.parse(text)
    if parsed_date:
        return parsed_date.strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")


def parse_expense(text):
    """
    Original regex-based parser. Kept as the offline fallback for when
    the LLM is unavailable (no API key, network error, rate limit, etc).
    """
    text = text.lower()

    patterns = [
        r"(?:spent|paid|cost|bought)\s+₹?(\d+)\s+(?:on|for)?\s*([\w\s]+)?",
        r"₹?(\d+)\s+(?:spent|for|on)\s+([\w\s]+)",
        r"(?:spent|paid)\s+₹?(\d+)",
    ]

    amount = None
    category = "uncategorized"
    stopwords = {"at", "for", "the", "a", "an", "in", "on", "my", "some", "with"}

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            amount = int(match.group(1))
            if len(match.groups()) > 1 and match.group(2):
                raw = re.sub(r"\s+", " ", match.group(2).lower().strip())
                # keep it short and consistent: drop filler words, cap at 2
                # meaningful tokens, so the fallback still yields clean
                # training labels for the ML classifier (e.g. "groceries at
                # big bazaar" -> "groceries big", "cab travel" -> "cab travel")
                tokens = [t for t in raw.split() if t not in stopwords][:2]
                category = " ".join(tokens) if tokens else "uncategorized"
            break

    if amount is None:
        return None

    return {
        "date": extract_date(text),
        "amount": amount,
        "category": category,
        "note": text,
    }


def hybrid_parse_expense(text):
    """
    The main entry point the app should call.

    1. Try the LLM parser first (best quality extraction + category guess).
    2. If the LLM call fails for any reason, fall back to the regex parser
       so the app never breaks just because the API key/network is unavailable.
    3. If a trained ML classifier exists and is confident about the category
       for this note, its prediction overrides the LLM/regex guess -- it's
       trained on this user's own historical categorization habits.

    Returns None if no amount could be extracted at all.
    """
    parsed = None
    source = None

    try:
        parsed = ask_llm(text)
        source = "llm"
    except Exception as e:
        logger.warning(f"LLM parsing failed ({e}); falling back to regex parser.")
        parsed = parse_expense(text)
        source = "regex"

    if parsed is None:
        return None

    parsed["parser_used"] = source
    parsed["category_source"] = source
    parsed["category_confidence"] = None

    predicted_cat, confidence = predict_category(parsed.get("note", ""))
    if predicted_cat and confidence >= CONFIDENCE_THRESHOLD:
        parsed["category"] = predicted_cat
        parsed["category_source"] = "classifier"
        parsed["category_confidence"] = round(confidence, 3)

    return parsed

def save_expense(entry):
    """
    Save one expense into SQLite.
    """

    entry["category"] = normalize_category(
        entry.get("category")
    )

    add_expense(
        date=entry.get("date"),
        amount=entry.get("amount"),
        category=entry.get("category"),
        note=entry.get("note"),
        parser_used=entry.get("parser_used"),
        category_source=entry.get("category_source"),
        category_confidence=entry.get("category_confidence")
    )

def summarize_expenses():

    df = get_all_expenses()

    if df.empty:
        return 0, {}

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    ).fillna(0)

    total = df["amount"].sum()

    by_category = (
        df.groupby("category")["amount"]
        .sum()
        .to_dict()
    )

    return total, by_category

def clear_expenses():

    db_clear_expenses()
