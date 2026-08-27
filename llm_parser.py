"""
LLM-based expense extraction using Groq's hosted Llama model.

Fixes vs. the original version:
- messages= (was message=)
- response.choices[0] (was response.choice[0])
- single well-formed user prompt (was two competing "system" messages)
- Pydantic schema validation so malformed/hallucinated LLM output raises
  a clear error instead of silently corrupting saved data
- lazy client init so importing this module doesn't crash when
  GROQ_API_KEY isn't set (lets the caller fall back gracefully)
"""

from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from datetime import datetime
import os
import json

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment (.env)")
        _client = Groq(api_key=api_key)
    return _client


class ExpenseSchema(BaseModel):
    amount: float
    category: str
    note: str
    date: str

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("amount must be a positive number")
        return v

    @field_validator("category")
    @classmethod
    def category_not_empty(cls, v):
        v = v.strip().lower()
        if not v:
            raise ValueError("category cannot be empty")
        return v


def ask_llm(
    text: str,
    model: str = "openai/gpt-oss-120b"
) -> dict:
    """
    Extract structured expense data from free text using an LLM.

    Raises RuntimeError / json.JSONDecodeError / pydantic.ValidationError
    on any failure. The caller (see utils.hybrid_parse_expense) is
    responsible for catching these and falling back to the regex parser.
    """
    client = _get_client()

    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Extract expense details from the user's message below and return ONLY valid JSON.
Do not include any preamble, explanation, or markdown code fences.

Required JSON format:
{{
  "amount": number,
  "category": "short string, e.g. groceries, travel, rent, dining, entertainment",
  "note": "short string describing the expense",
  "date": "YYYY-MM-DD"
}}

Rules:
- Infer the category intelligently from context, keep it short (one or two words).
- note should briefly describe what the expense was for, in your own words.
- If no date is mentioned in the input, use today's date: {today}.

User input: "{text}"
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expense extraction assistant. You respond "
                    "only with a single valid JSON object matching the "
                    "requested schema, nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)  # raises json.JSONDecodeError if malformed

    validated = ExpenseSchema(**data)  # raises ValidationError if schema mismatch
    return validated.model_dump()
