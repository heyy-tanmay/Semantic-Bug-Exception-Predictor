"""
dataset_loader.py
-----------------
Simple data handling utilities for the Semantic Bug & Exception Predictor.

This module creates a small dummy dataset of C/Java code snippets and
provides a PyTorch-friendly Dataset wrapper that tokenizes code using
the CodeBERT tokenizer (`microsoft/codebert-base`).

The goal is to keep things clear and beginner-friendly so you can
present and extend this for your college project.
"""

from typing import List, Tuple
import random
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_dummy_dataset() -> pd.DataFrame:
    """Create a very small dummy dataset of code snippets.

    Returns a DataFrame with two columns:
      - `code_snippet`: the raw code as a string (C or Java)
      - `is_buggy`: 0 (clean) or 1 (contains a semantic bug)

    In a real project you would load a properly labeled dataset from
    files or a database. This dummy data is only for experimentation
    and to demonstrate the end-to-end pipeline.
    """

    samples = [
        # Java examples (some buggy, some clean)
        ("public class Main { public static void main(String[] a){ int[] x = new int[5]; System.out.println(x[10]); } }", 1),
        ("public class Hello { public static void main(String[] args){ System.out.println(\"Hello world\"); } }", 0),
        ("String s = null; System.out.println(s.length());", 1),

        # C examples
        ("#include <stdio.h>\nint main(){ int *p = NULL; printf(\"%d\", *p); return 0; }", 1),
        ("#include <stdio.h>\nint main(){ int a = 10; printf(\"%d\", a); return 0; }", 0),
        ("#include <stdlib.h>\nint main(){ char *p = malloc(10); free(p); p[0] = 'a'; return 0; }", 0),
        ("#include <stdlib.h>\nint main(){ char *p = malloc(5); p[10] = 'x'; return 0; }", 1),
    ]

    # Duplicate and shuffle to make a slightly larger toy set
    expanded = samples + random.sample(samples, k=4)
    random.shuffle(expanded)

    df = pd.DataFrame(expanded, columns=["code_snippet", "is_buggy"])
    return df


class CodeDataset(Dataset):
    """A simple PyTorch Dataset that tokenizes code snippets with CodeBERT.

    Each item returned is a dict expected by Hugging Face models and
    Trainer/DataLoader utilities.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        code = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize using the provided tokenizer. We use truncation and
        # return PyTorch tensors later via a data collator in training.
        encoded = self.tokenizer(code, truncation=True, padding=False, max_length=self.max_length)

        # Convert lists to plain Python ints/lists; the Trainer will
        # collate these into tensors.
        item = {k: v for k, v in encoded.items()}
        item["labels"] = label
        return item


def prepare_datasets(tokenizer_name: str = "microsoft/codebert-base") -> Tuple[CodeDataset, CodeDataset]:
    """Load the dummy dataset and prepare train / eval CodeDataset instances.

    - tokenizer_name: model name for `AutoTokenizer`.
    Returns (train_dataset, eval_dataset)
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    df = load_dummy_dataset()
    texts = df["code_snippet"].tolist()
    labels = df["is_buggy"].tolist()

    # A simple 80/20 split
    split_idx = int(0.8 * len(texts))
    train_texts, eval_texts = texts[:split_idx], texts[split_idx:]
    train_labels, eval_labels = labels[:split_idx], labels[split_idx:]

    train_dataset = CodeDataset(train_texts, train_labels, tokenizer)
    eval_dataset = CodeDataset(eval_texts, eval_labels, tokenizer)

    return train_dataset, eval_dataset


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    train_ds, eval_ds = prepare_datasets()
    print(f"Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")
