import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class FileLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Write to file
            with open(self.log_file, "a") as f:
                f.write(f"[Step {state.global_step:4d}] ")
                for key, value in logs.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}  ")
                    else:
                        f.write(f"{key}: {value}  ")
                f.write("\n")

            # Also print to console (optional)
            print(f"[Step {state.global_step:4d}] ", end="")
            for key, value in logs.items():
                if key in ["loss", "eval_loss", "eval_accuracy", "learning_rate"]:
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}  ", end="")
            print()


# Simple console logging callback
class ConsoleLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Format the output cleanly
        print(f"[Step {state.global_step:4d}] ", end="")
        for key, value in logs.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}  ", end="")
            else:
                print(f"{key}: {value}  ", end="")
        print()


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
    disable_tqdm=True,
    report_to="none",
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), FileLoggerCallback()],
)

# 8. Train!
print("Starting training...")
trainer.train()
print("\nTraining complete!")

# 9. Save everything
trainer.save_model("./trail_classifier")
tokenizer.save_pretrained("./trail_classifier")
