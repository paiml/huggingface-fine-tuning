# Capstone Project: Sentiment Analysis Pipeline

Build an end-to-end sentiment analysis system that fine-tunes a transformer model on custom data and deploys it to Hugging Face Hub. This capstone demonstrates mastery of the complete fine-tuning workflow.

## Overview

The Sentiment Analysis Pipeline provides:

- **Data Pipeline** — Load, clean, and augment training data from multiple formats
- **Training System** — Fine-tune with Trainer API, callbacks, and optimization
- **Evaluation Suite** — Comprehensive metrics and model comparison
- **Deployment** — Publish to Hugging Face Hub with proper documentation

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run the complete pipeline
uv run python capstone/train.py

# Evaluate the model
uv run python capstone/evaluate.py

# Publish to Hub
uv run python capstone/publish.py
```

## Architecture

```
capstone/
├── data/
│   ├── __init__.py
│   ├── loader.py           # Multi-format data loading
│   ├── preprocessor.py     # Text cleaning and normalization
│   └── augmenter.py        # Data augmentation strategies
├── training/
│   ├── __init__.py
│   ├── tokenizer.py        # Tokenization with padding/truncation
│   ├── trainer.py          # Trainer API configuration
│   ├── callbacks.py        # Custom callbacks (logging, early stopping)
│   └── metrics.py          # Evaluation metrics (accuracy, F1, precision, recall)
├── models/
│   ├── __init__.py
│   ├── config.py           # Model and training configurations
│   └── inference.py        # Prediction utilities
├── publishing/
│   ├── __init__.py
│   ├── hub.py              # Hugging Face Hub integration
│   └── model_card.py       # Model card generation
├── train.py                # Main training script
├── evaluate.py             # Evaluation and comparison
├── publish.py              # Hub publishing
└── demo.py                 # Interactive demo
```

## Components

### Data Pipeline (`capstone/data/`)

#### Loader (`loader.py`)

Multi-format data loading with validation:

```python
from capstone.data.loader import DataLoader

loader = DataLoader()

# Load from multiple formats
csv_data = loader.load_csv("reviews.csv", text_col="review", label_col="sentiment")
json_data = loader.load_json("reviews.json")
hub_data = loader.load_from_hub("imdb", split="train[:1000]")

# Combine datasets
combined = loader.combine([csv_data, json_data, hub_data])
print(f"Total samples: {len(combined)}")
```

Features:
- CSV, JSON, Parquet, and Hugging Face Hub support
- Column mapping and validation
- Automatic label encoding
- Dataset statistics and validation

#### Preprocessor (`preprocessor.py`)

Text cleaning and normalization:

```python
from capstone.data.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor(
    lowercase=True,
    remove_urls=True,
    remove_special_chars=False,
    min_length=10,
    max_length=512,
)

cleaned = preprocessor.transform(dataset)
stats = preprocessor.get_stats()
print(f"Removed {stats['filtered_count']} samples below min length")
```

#### Augmenter (`augmenter.py`)

Data augmentation for imbalanced datasets:

```python
from capstone.data.augmenter import DataAugmenter

augmenter = DataAugmenter(
    strategies=["synonym_replacement", "random_deletion", "back_translation"],
    augment_ratio=0.3,  # Augment 30% of minority class
)

# Check class distribution
print(augmenter.get_class_distribution(dataset))

# Balance the dataset
balanced = augmenter.balance(dataset, target_ratio=1.0)
```

### Training System (`capstone/training/`)

#### Tokenizer (`tokenizer.py`)

Configurable tokenization:

```python
from capstone.training.tokenizer import SentimentTokenizer

tokenizer = SentimentTokenizer(
    model_name="bert-base-uncased",
    max_length=256,
    padding="max_length",
    truncation=True,
)

# Tokenize dataset
tokenized = tokenizer.tokenize_dataset(
    dataset,
    text_column="text",
    label_column="label",
)

# Get tokenization stats
stats = tokenizer.get_stats(dataset)
print(f"Average tokens: {stats['avg_tokens']:.1f}")
print(f"Truncated: {stats['truncated_pct']:.1%}")
```

#### Trainer (`trainer.py`)

Trainer API wrapper with best practices:

```python
from capstone.training.trainer import SentimentTrainer
from capstone.training.callbacks import EarlyStoppingCallback, LoggingCallback

trainer = SentimentTrainer(
    model_name="bert-base-uncased",
    num_labels=3,  # negative, neutral, positive
    output_dir="./sentiment_model",
)

# Configure training
trainer.configure(
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,  # Mixed precision training
)

# Add callbacks
trainer.add_callback(EarlyStoppingCallback(patience=3))
trainer.add_callback(LoggingCallback(log_dir="./logs"))

# Train
results = trainer.train(train_dataset, eval_dataset)
print(f"Best accuracy: {results['best_accuracy']:.2%}")
```

#### Callbacks (`callbacks.py`)

Custom training callbacks:

```python
from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_loss = metrics.get("eval_loss", float("inf"))
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
                print(f"Early stopping at epoch {state.epoch}")


class LoggingCallback(TrainerCallback):
    """Log training progress to file and console."""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def on_log(self, args, state, control, logs, **kwargs):
        if "loss" in logs:
            print(f"Step {state.global_step}: loss={logs['loss']:.4f}")
```

#### Metrics (`metrics.py`)

Comprehensive evaluation metrics:

```python
from capstone.training.metrics import compute_metrics, ClassificationReport

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro"),
    }

# Generate detailed report
report = ClassificationReport(predictions, labels, label_names=["negative", "neutral", "positive"])
print(report.to_markdown())
```

### Inference (`capstone/models/inference.py`)

Production-ready inference:

```python
from capstone.models.inference import SentimentPredictor

predictor = SentimentPredictor("./sentiment_model")

# Single prediction
result = predictor.predict("This product is amazing!")
print(f"Sentiment: {result['label']} ({result['confidence']:.1%})")

# Batch prediction
texts = [
    "Great quality, highly recommend!",
    "Average product, nothing special.",
    "Terrible experience, waste of money.",
]
results = predictor.predict_batch(texts)
for text, result in zip(texts, results):
    print(f"{text[:30]}... -> {result['label']}")

# With confidence threshold
filtered = predictor.predict_batch(texts, min_confidence=0.8)
```

### Publishing (`capstone/publishing/`)

#### Hub Integration (`hub.py`)

```python
from capstone.publishing.hub import HubPublisher

publisher = HubPublisher(
    model_path="./sentiment_model",
    repo_name="your-username/sentiment-classifier",
    private=False,
)

# Validate before publishing
publisher.validate()

# Push to Hub
publisher.push(
    commit_message="Initial release v1.0",
    tags=["sentiment-analysis", "bert", "fine-tuned"],
)

# Create model card
publisher.generate_model_card(
    description="Fine-tuned BERT for sentiment analysis",
    training_data="Custom product reviews dataset",
    metrics={"accuracy": 0.92, "f1": 0.91},
)
```

#### Model Card (`model_card.py`)

```python
from capstone.publishing.model_card import ModelCardGenerator

generator = ModelCardGenerator(
    model_name="Sentiment Classifier",
    base_model="bert-base-uncased",
    task="text-classification",
)

# Add sections
generator.add_description(
    "Fine-tuned BERT model for 3-class sentiment analysis (positive, neutral, negative)."
)

generator.add_training_details(
    dataset="Custom product reviews (10,000 samples)",
    epochs=5,
    batch_size=16,
    learning_rate="2e-5",
)

generator.add_evaluation(
    metrics={
        "accuracy": 0.92,
        "f1_macro": 0.91,
        "precision": 0.90,
        "recall": 0.91,
    },
    confusion_matrix=cm_image_path,
)

generator.add_usage_example()
generator.add_limitations(
    "Trained on English product reviews. May not generalize to other domains."
)

# Generate and save
card = generator.generate()
card.save("./sentiment_model/README.md")
```

## Main Scripts

### Training Script (`train.py`)

```python
#!/usr/bin/env python
"""Main training script for sentiment analysis pipeline."""

import argparse
from capstone.data.loader import DataLoader
from capstone.data.preprocessor import TextPreprocessor
from capstone.data.augmenter import DataAugmenter
from capstone.training.tokenizer import SentimentTokenizer
from capstone.training.trainer import SentimentTrainer
from capstone.training.callbacks import EarlyStoppingCallback

def main(args):
    # 1. Load data
    print("Loading data...")
    loader = DataLoader()
    dataset = loader.load_csv(args.data_path, text_col="text", label_col="label")

    # 2. Preprocess
    print("Preprocessing...")
    preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
    dataset = preprocessor.transform(dataset)

    # 3. Handle imbalance
    if args.balance:
        print("Balancing dataset...")
        augmenter = DataAugmenter(strategies=["synonym_replacement"])
        dataset = augmenter.balance(dataset)

    # 4. Split
    splits = dataset.train_test_split(test_size=0.2, seed=42)

    # 5. Tokenize
    print("Tokenizing...")
    tokenizer = SentimentTokenizer(model_name=args.model, max_length=args.max_length)
    train_data = tokenizer.tokenize_dataset(splits["train"])
    eval_data = tokenizer.tokenize_dataset(splits["test"])

    # 6. Train
    print("Training...")
    trainer = SentimentTrainer(
        model_name=args.model,
        num_labels=args.num_labels,
        output_dir=args.output_dir,
    )
    trainer.configure(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
    )
    trainer.add_callback(EarlyStoppingCallback(patience=args.patience))

    results = trainer.train(train_data, eval_data)

    # 7. Save
    trainer.save()
    print(f"Model saved to {args.output_dir}")
    print(f"Final accuracy: {results['eval_accuracy']:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--output-dir", default="./sentiment_model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-labels", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    main(parser.parse_args())
```

### Evaluation Script (`evaluate.py`)

```python
#!/usr/bin/env python
"""Evaluate and compare trained models."""

import argparse
from capstone.models.inference import SentimentPredictor
from capstone.training.metrics import ClassificationReport
from capstone.data.loader import DataLoader

def main(args):
    # Load test data
    loader = DataLoader()
    test_data = loader.load_csv(args.test_path)

    # Load model
    predictor = SentimentPredictor(args.model_path)

    # Get predictions
    predictions = predictor.predict_batch(test_data["text"])
    pred_labels = [p["label_id"] for p in predictions]
    true_labels = test_data["label"]

    # Generate report
    report = ClassificationReport(
        pred_labels,
        true_labels,
        label_names=["negative", "neutral", "positive"],
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(report.to_markdown())

    # Save report
    if args.output:
        report.save(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output", default="evaluation_report.md")
    main(parser.parse_args())
```

### Publishing Script (`publish.py`)

```python
#!/usr/bin/env python
"""Publish model to Hugging Face Hub."""

import argparse
from capstone.publishing.hub import HubPublisher
from capstone.publishing.model_card import ModelCardGenerator

def main(args):
    # Generate model card
    print("Generating model card...")
    generator = ModelCardGenerator(
        model_name="Sentiment Classifier",
        base_model=args.base_model,
        task="text-classification",
    )
    generator.add_description(args.description)
    generator.add_training_details(
        dataset=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    card = generator.generate()
    card.save(f"{args.model_path}/README.md")

    # Publish
    print(f"Publishing to {args.repo_name}...")
    publisher = HubPublisher(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private,
    )
    publisher.validate()
    publisher.push(
        commit_message=args.commit_message,
        tags=["sentiment-analysis", "fine-tuned"],
    )

    print(f"Published: https://huggingface.co/{args.repo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--description", default="Fine-tuned sentiment classifier")
    parser.add_argument("--dataset-name", default="Custom dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--commit-message", default="Initial release")
    parser.add_argument("--private", action="store_true")
    main(parser.parse_args())
```

## Evaluation Criteria

### Functionality (40 points)

- [ ] Data loading from multiple formats (10 pts)
- [ ] Text preprocessing and cleaning (5 pts)
- [ ] Data augmentation for imbalanced classes (5 pts)
- [ ] Training with Trainer API (10 pts)
- [ ] Inference with confidence scores (5 pts)
- [ ] Publishing to Hugging Face Hub (5 pts)

### Code Quality (30 points)

- [ ] Clean, modular code structure (10 pts)
- [ ] Type hints and docstrings (10 pts)
- [ ] Error handling and validation (10 pts)

### Documentation (20 points)

- [ ] README with setup instructions (5 pts)
- [ ] Model card with proper metadata (10 pts)
- [ ] Usage examples (5 pts)

### Bonus Features (10 points)

- [ ] Hyperparameter tuning with Optuna (5 pts)
- [ ] Model comparison report (5 pts)

## Challenge Extensions

### 1. Hyperparameter Tuning

```python
import optuna

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    trainer = SentimentTrainer(...)
    trainer.configure(
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
    )
    results = trainer.train(train_data, eval_data)
    return results["eval_accuracy"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(f"Best params: {study.best_params}")
```

### 2. Multi-Model Comparison

```python
models = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
]

results = []
for model_name in models:
    trainer = SentimentTrainer(model_name=model_name)
    metrics = trainer.train(train_data, eval_data)
    results.append({
        "model": model_name,
        "accuracy": metrics["eval_accuracy"],
        "f1": metrics["eval_f1"],
        "train_time": metrics["train_runtime"],
    })

# Generate comparison table
df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

### 3. Gradio Demo

```python
import gradio as gr
from capstone.models.inference import SentimentPredictor

predictor = SentimentPredictor("./sentiment_model")

def classify(text):
    result = predictor.predict(text)
    return {
        "Positive": result["scores"][2],
        "Neutral": result["scores"][1],
        "Negative": result["scores"][0],
    }

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.Label(num_top_classes=3),
    title="Sentiment Classifier",
    examples=[
        "This product exceeded my expectations!",
        "It's okay, nothing special.",
        "Worst purchase ever, complete waste of money.",
    ],
)

demo.launch()
```

### 4. Distributed Training

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

## Testing

```bash
# Run all tests
uv run pytest capstone/tests/ -v

# Run with coverage
uv run pytest capstone/tests/ --cov=capstone --cov-report=term-missing

# Test specific modules
uv run pytest capstone/tests/test_loader.py -v
uv run pytest capstone/tests/test_trainer.py -v
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
trainer.configure(batch_size=8)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
trainer.configure(fp16=True)
```

### Slow Training

```python
# Use a smaller model
model_name = "distilbert-base-uncased"  # 66M vs 110M params

# Reduce max sequence length
tokenizer = SentimentTokenizer(max_length=128)

# Use fewer evaluation steps
trainer.configure(eval_steps=500)
```

### Hub Authentication

```bash
# Check login status
huggingface-cli whoami

# Re-login
huggingface-cli login

# Or use environment variable
export HF_TOKEN="your_token_here"
```

## Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [Trainer API Guide](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)

## Skills Demonstrated

| Course Lab | Skills Applied |
|------------|----------------|
| Lab 1: Loading Data | DataLoader with multi-format support |
| Lab 2: Tokenization | SentimentTokenizer with padding/truncation |
| Lab 3: Augmentation | DataAugmenter for imbalanced data |
| Lab 4: Training | SentimentTrainer with Trainer API |
| Lab 5: Callbacks | EarlyStoppingCallback, LoggingCallback |
| Lab 6: Publishing | HubPublisher, ModelCardGenerator |

This capstone integrates all course concepts into a production-ready fine-tuning pipeline with comprehensive testing, documentation, and deployment to Hugging Face Hub.
