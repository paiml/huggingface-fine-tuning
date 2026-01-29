import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# 1. Load your CSV
df = pd.read_csv("status.csv")

# 2. Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# 3. Split into train/test
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Train: {len(split_dataset['train'])} examples")
print(f"Test: {len(split_dataset['test'])} examples")

# 4. Load model for classification (2 labels: open/closed)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,  # Binary classification
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# 5. Tokenize function
def tokenize_and_label(examples):
    # Tokenize
    tokenized = tokenizer(examples["status"], truncation=True, padding=True)
    # Add labels (convert to list if single value)
    tokenized["labels"] = examples["Blankets_Creek"]
    return tokenized


tokenized_datasets = split_dataset.map(tokenize_and_label, batched=True)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./trail_classifier",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    eval_strategy="epoch",
    metric_for_best_model="accuracy",
)


def compute_metrics(eval_pred):
    """Calculate accuracy from predictions"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


# 7. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# 8. Train!
print("Starting training...")
trainer.train()
print("\nTraining complete!")

print("\n=== Debugging: Overfitting Check ===")
train_results = trainer.predict(tokenized_datasets["train"])
eval_results = trainer.predict(tokenized_datasets["test"])

train_acc = accuracy_score(tokenized_datasets["train"]["labels"], np.argmax(train_results.predictions, axis=1))
eval_acc = accuracy_score(tokenized_datasets["test"]["labels"], np.argmax(eval_results.predictions, axis=1))

print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {eval_acc:.4f}")
print(f"Gap: {train_acc - eval_acc:.4f}")

if train_acc - eval_acc > 0.15:  # If training accuracy is much higher
    print("   WARNING: Large accuracy gap suggests overfitting")
    print("   Possible causes: Dataset too small, model too complex")
    print("   Try: More data, simpler model, or stronger regularization")
elif train_acc - eval_acc > 0.05:
    print("   Moderate gap - monitor for overfitting")
else:
    print("   Good generalization - similar train/test performance")


# 9. Save everything
trainer.save_model("./trail_classifier")
tokenizer.save_pretrained("./trail_classifier")
