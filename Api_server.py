"""
api_server.py
-------------
Simple FastAPI server to expose the saved bug prediction model.

Endpoint:
  POST /predict_bug
    - Accepts JSON: {"code": "<raw C or Java code here>"}
    - Returns JSON: {"bug_detected": bool, "confidence_score": float, "likely_issue": str}

Notes:
 - This loads the model saved by `model_trainer.py` in `saved_bug_predictor_model`.
 - The `likely_issue` field is a best-effort heuristic mapping based on
   common patterns (null, malloc, array index), not a learned classifier.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CodeInput(BaseModel):
    code: str


app = FastAPI(title="Semantic Bug & Exception Predictor")

# Try to load the saved model at startup. If the model is not present,
# the server will raise an error on prediction telling the user to run
# the trainer first.
MODEL_DIR = "saved_bug_predictor_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
except Exception:
    tokenizer = None
    model = None


def heuristic_issue_guess(code: str) -> str:
    """A tiny heuristic to guess likely issue types from code text.

    This is not perfect — it's just for friendly output in the demo.
    """
    low = code.lower()
    if "null" in low or "nullptr" in low or ".length()" in low:
        return "Null Pointer / Dereference"
    if "malloc" in low or "free" in low or "segfault" in low:
        return "Memory management (segfault)"
    if "[" in low and "]" in low and "printf" in low:
        return "Out of bounds array access"
    if "division" in low or "/0" in low:
        return "Division by zero"
    return "General runtime/error-prone pattern"


@app.post("/predict_bug")
def predict_bug(payload: CodeInput) -> Dict:
    """Predict whether the provided code snippet has a semantic bug.

    Returns a JSON response with bug detection flag, confidence, and
    a friendly label describing the likely type of issue.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not found. Run `model_trainer.py` to train and save the model to 'saved_bug_predictor_model'.")

    code = payload.code
    if not code or not code.strip():
        raise HTTPException(status_code=400, detail="Empty code provided.")

    # Tokenize and prepare tensors
    encoded = tokenizer(code, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Label 1 -> buggy, Label 0 -> clean (consistent with training setup)
    bug_confidence = float(probs[1])
    bug_detected = bool(bug_confidence >= 0.5)

    likely_issue = heuristic_issue_guess(code)

    return {
        "bug_detected": bug_detected,
        "confidence_score": round(bug_confidence, 4),
        "likely_issue": likely_issue,
    }


if __name__ == "__main__":
    # Run with: `uvicorn api_server:app --reload` from the workspace root
    print("This module implements a FastAPI app. Run with uvicorn:")
    print("uvicorn api_server:app --reload --port 8000")
