"""
ai_models.py
------------
AI Brain for the Semantic Bug Predictor & Auto-Fixer.

This module initializes two pre-trained Hugging Face pipelines:

  1. BugDetector (Model A): Uses microsoft/codebert-base for sequence
     classification to detect if code is buggy or clean.

  2. BugFixer (Model B): Uses Salesforce/codet5-base for text2text
     generation to automatically fix buggy code.

Both models are loaded via the `transformers` library's pipeline API,
which handles tokenization, inference, and decoding automatically.

Architecture Notes:
  - Pipelines abstract away tokenization and tensor handling.
  - Pre-trained weights are cached locally after first download.
  - For production, consider model quantization and batching.
"""

from transformers import pipeline
import torch
from typing import Dict, Tuple


class BugDetector:
    """Detects whether a code snippet contains a semantic bug.

    Uses a fine-tuned CodeBERT model for binary classification:
      - Label 0: Clean code
      - Label 1: Buggy code (contains semantic error)

    The pipeline returns logits which we convert to probabilities.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """Initialize the bug detection pipeline.

        Args:
            model_name: HuggingFace model identifier. Default is CodeBERT.
        """
        print(f"[BugDetector] Loading model: {model_name}")
        # The 'zero-shot-classification' pipeline can work with any
        # sequence classifier. We use 'text-classification' which is
        # more standard for binary classification tasks.
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
            top_k=None,  # Return all class scores
        )
        print("[BugDetector] Model loaded successfully")

    def predict(self, code: str) -> Tuple[bool, float]:
        """Predict if code is buggy.

        Args:
            code: Raw code snippet (C or Java).

        Returns:
            Tuple of (is_buggy: bool, confidence: float [0-1]).
            is_buggy=True if model predicts bug with confidence >= 0.5.
        """
        # Remove extra whitespace
        code = code.strip()
        if not code:
            return False, 0.0

        # Run inference through the pipeline
        # Returns a list of dicts: [{"label": "LABEL_0", "score": 0.9}, ...]
        results = self.pipeline(code)

        # Parse results: typically LABEL_1 = buggy, LABEL_0 = clean
        # We'll find the score for label 1 (buggy)
        buggy_score = 0.0
        for result in results:
            # Labels are usually "LABEL_0", "LABEL_1", etc.
            if "LABEL_1" in result["label"]:
                buggy_score = result["score"]
                break

        is_buggy = buggy_score >= 0.5
        return is_buggy, buggy_score


class BugFixer:
    """Automatically fixes buggy code using a generative AI model.

    Uses Salesforce/codet5-base (a pre-trained code-to-code T5 model)
    for sequence-to-sequence generation. Given a buggy code snippet,
    it predicts the corrected version.

    Note: This is a lightweight demonstration. For best results, the
    model should be fine-tuned on pairs of buggy/fixed code.
    """

    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        """Initialize the bug fixing pipeline.

        Args:
            model_name: HuggingFace model identifier. Default is CodeT5.
        """
        print(f"[BugFixer] Loading model: {model_name}")
        # Use the 'text2text-generation' pipeline for seq2seq models
        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
        )
        print("[BugFixer] Model loaded successfully")

    def fix(self, buggy_code: str, max_length: int = 256) -> str:
        """Generate a corrected version of buggy code.

        Args:
            buggy_code: The buggy code snippet.
            max_length: Maximum length of generated output tokens.

        Returns:
            Corrected code string. If generation fails, returns original.
        """
        buggy_code = buggy_code.strip()
        if not buggy_code:
            return buggy_code

        try:
            # Add a simple prefix hint to the model
            # In production, you'd fine-tune the model for this task
            input_text = f"fix code: {buggy_code}"

            # Generate the correction using the pipeline
            # Returns list of dicts: [{"generated_text": "..."}, ...]
            results = self.pipeline(input_text, max_length=max_length, num_beams=4)

            if results and len(results) > 0:
                generated = results[0]["generated_text"]
                # Remove the prefix if the model echoed it
                if generated.startswith("fix code:"):
                    generated = generated[len("fix code:"):].strip()
                return generated
            else:
                return buggy_code
        except Exception as e:
            print(f"[BugFixer] Error during fix generation: {e}")
            return buggy_code


class AIModelManager:
    """Manager class to handle both detector and fixer models.

    This is a convenience wrapper that initializes both pipelines
    and provides a unified interface for the API layer.
    """

    def __init__(
        self,
        detector_model: str = "microsoft/codebert-base",
        fixer_model: str = "Salesforce/codet5-base",
    ):
        """Initialize both AI models.

        Args:
            detector_model: Model for bug detection.
            fixer_model: Model for bug fixing.
        """
        print("[AIModelManager] Initializing AI models...")
        self.detector = BugDetector(model_name=detector_model)
        self.fixer = BugFixer(model_name=fixer_model)
        print("[AIModelManager] All models initialized successfully")

    def analyze_and_fix(self, code: str) -> Dict:
        """Full pipeline: detect bug, then fix if found.

        Args:
            code: Raw code snippet.

        Returns:
            Dict with keys:
              - has_bug: bool
              - confidence: float [0-1]
              - original_code: str
              - fixed_code: str (only if has_bug=True)
        """
        # Step 1: Detect if code has a bug
        is_buggy, confidence = self.detector.predict(code)

        result = {
            "has_bug": is_buggy,
            "confidence": round(confidence, 4),
            "original_code": code,
        }

        # Step 2: If buggy, generate a fix
        if is_buggy:
            fixed = self.fixer.fix(code)
            result["fixed_code"] = fixed
        else:
            result["fixed_code"] = None

        return result


if __name__ == "__main__":
    # Quick test when running this file directly
    print("=" * 60)
    print("Testing AI Models Locally")
    print("=" * 60)

    manager = AIModelManager()

    # Test with a buggy Java snippet
    test_code = "int[] arr = new int[5];\nSystem.out.println(arr[10]);"
    print(f"\nAnalyzing code:\n{test_code}\n")

    result = manager.analyze_and_fix(test_code)
    print("Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
