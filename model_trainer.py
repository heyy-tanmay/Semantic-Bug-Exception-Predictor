"""
model_trainer.py
----------------
Training script for the Semantic Bug & Exception Predictor.

This file sets up a sequence classification model using CodeBERT
(`microsoft/codebert-base`) and the Hugging Face `Trainer` API.

It performs a short training run on the tiny dummy dataset, evaluates
using Accuracy/Precision/Recall, and saves the trained model and
tokenizer to `saved_bug_predictor_model`.

Notes for students:
 - `transformers` provides tokenizers, model classes, and Trainer.
 - `sklearn.metrics` is used for simple evaluation metrics.
 - This is a minimal example to demonstrate the flow; for real
   experiments, use a larger, properly labeled dataset and more
   careful hyperparameter tuning.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Import dataset utilities we created
from dataset_loader import prepare_datasets


def compute_metrics(eval_pred):
    """Compute metrics given predictions from the Trainer.

    - eval_pred: EvalPrediction object from Hugging Face Trainer
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    # For binary tasks, set average='binary' and ensure labels are 0/1
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec}


def train_and_save(model_name: str = "microsoft/codebert-base", output_dir: str = "saved_bug_predictor_model"):
    """Train a classifier and save artifacts.

    - model_name: pretrained model for tokenizer and base weights
    - output_dir: folder where model and tokenizer will be saved
    """

    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, eval_dataset = prepare_datasets(tokenizer_name=model_name)

    # Load a model for sequence classification (binary -> num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments: short run for demo purposes.
    training_args = TrainingArguments(
        output_dir="./hf_tmp",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=False,
        fp16=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training (this may download model weights)...")
    trainer.train()

    print("Training complete. Evaluating on evaluation set...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Save the model and tokenizer to the requested output directory
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to '{output_dir}'.")


if __name__ == "__main__":
    # When running directly, perform a quick train+save.
    train_and_save()
