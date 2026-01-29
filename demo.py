#!/usr/bin/env python3
"""Interactive demo showcasing fine-tuning concepts with rich terminal output.

This demo provides a visual walkthrough of Hugging Face fine-tuning workflows.
Run with: uv run python demo.py
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

console = Console()


def print_banner() -> None:
    """Print the demo banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ███████╗██╗███╗   ██╗███████╗    ████████╗██╗   ██╗███╗   ██╗███████╗║
║   ██╔════╝██║████╗  ██║██╔════╝    ╚══██╔══╝██║   ██║████╗  ██║██╔════╝║
║   █████╗  ██║██╔██╗ ██║█████╗         ██║   ██║   ██║██╔██╗ ██║█████╗  ║
║   ██╔══╝  ██║██║╚██╗██║██╔══╝         ██║   ██║   ██║██║╚██╗██║██╔══╝  ║
║   ██║     ██║██║ ╚████║███████╗       ██║   ╚██████╔╝██║ ╚████║███████╗║
║   ╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝       ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝║
║                                                                       ║
║   Fine-tuning with Hugging Face - Interactive Demo                    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


def show_course_outline() -> None:
    """Display the course outline as a tree."""
    tree = Tree("[bold magenta]Fine-tuning Course[/bold magenta]")

    lesson1 = tree.add("[cyan]Lesson 1: Working with Data[/cyan]")
    lesson1.add("Loading datasets (CSV, JSON, Parquet, Hub)")
    lesson1.add("Transforming datasets (map, filter)")
    lesson1.add("Handling imbalanced data")
    lesson1.add("Data augmentation techniques")

    lesson2 = tree.add("[cyan]Lesson 2: Tokenization[/cyan]")
    lesson2.add("Tokenization with padding")
    lesson2.add("Tokenization with truncation")
    lesson2.add("Special tokens and attention masks")

    lesson3 = tree.add("[cyan]Lesson 3: Models and Training[/cyan]")
    lesson3.add("Pre-trained model selection")
    lesson3.add("Training with Trainer API")
    lesson3.add("Custom training configurations")

    lesson4 = tree.add("[cyan]Lesson 4: Advanced Training[/cyan]")
    lesson4.add("Training callbacks and logging")
    lesson4.add("Early stopping")
    lesson4.add("Learning rate scheduling")

    lesson5 = tree.add("[cyan]Lesson 5: Inference and Publishing[/cyan]")
    lesson5.add("Running inference")
    lesson5.add("Publishing to Hugging Face Hub")

    console.print(tree)


def show_examples() -> None:
    """Display available examples in a table."""
    table = Table(title="Available Examples", show_header=True, header_style="bold magenta")
    table.add_column("Directory", style="cyan")
    table.add_column("Topic", style="green")
    table.add_column("Key Concepts")

    table.add_row("loading/", "Dataset Loading", "CSV, JSON, Parquet, Hub datasets")
    table.add_row("transform/", "Data Transformation", "map(), filter(), batched processing")
    table.add_row("tokenize/", "Tokenization", "Padding, truncation, special tokens")
    table.add_row("augment/", "Data Augmentation", "Text augmentation techniques")
    table.add_row("imbalance/", "Imbalanced Data", "Oversampling, class weights")
    table.add_row("models/", "Pre-trained Models", "Model selection, AutoModel")
    table.add_row("training/", "Trainer API", "TrainingArguments, Trainer")
    table.add_row("custom/", "Custom Training", "Custom metrics, loss functions")
    table.add_row("callback/", "Callbacks", "EarlyStoppingCallback, logging")
    table.add_row("inferencing/", "Inference", "Pipeline, batch inference")
    table.add_row("publishing/", "Publishing", "push_to_hub, model cards")

    console.print(table)


def simulate_training() -> None:
    """Simulate a training run with progress indicators."""
    console.print("\n[bold cyan]Simulating Fine-tuning Workflow...[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading dataset...", total=100)
        import time

        time.sleep(0.5)
        progress.update(task, advance=100, description="[green]Dataset loaded: 10,000 samples")

        task2 = progress.add_task("[cyan]Tokenizing text...", total=100)
        time.sleep(0.5)
        progress.update(task2, advance=100, description="[green]Tokenization complete")

        task3 = progress.add_task("[cyan]Loading pre-trained model...", total=100)
        time.sleep(0.5)
        progress.update(task3, advance=100, description="[green]Model: bert-base-uncased")

        task4 = progress.add_task("[cyan]Training epoch 1/3...", total=100)
        time.sleep(0.3)
        progress.update(task4, advance=100, description="[green]Epoch 1: loss=0.45")

        task5 = progress.add_task("[cyan]Training epoch 2/3...", total=100)
        time.sleep(0.3)
        progress.update(task5, advance=100, description="[green]Epoch 2: loss=0.28")

        task6 = progress.add_task("[cyan]Training epoch 3/3...", total=100)
        time.sleep(0.3)
        progress.update(task6, advance=100, description="[green]Epoch 3: loss=0.15")

    # Show results
    results = Table(title="Training Results", show_header=True)
    results.add_column("Metric", style="cyan")
    results.add_column("Value", style="green", justify="right")

    results.add_row("Final Loss", "0.15")
    results.add_row("Accuracy", "94.2%")
    results.add_row("F1 Score", "0.941")
    results.add_row("Training Time", "2m 34s")

    console.print(results)


def show_labs() -> None:
    """Display available labs."""
    table = Table(title="Hands-on Labs", show_header=True, header_style="bold magenta")
    table.add_column("Lab", style="cyan", justify="center")
    table.add_column("Topic", style="green")
    table.add_column("Examples Used")

    table.add_row("1", "Loading and Exploring Datasets", "loading/")
    table.add_row("2", "Transformations and Tokenization", "transform/, tokenize/")
    table.add_row("3", "Custom Datasets and Augmentation", "augment/, imbalance/")
    table.add_row("4", "Training with Trainer API", "training/, models/")
    table.add_row("5", "Advanced Training and Callbacks", "custom/, callback/")
    table.add_row("6", "Publishing Models", "publishing/")

    console.print(table)


def main() -> int:
    """Run the interactive demo."""
    print_banner()
    console.print("\n[bold]Welcome to Fine-tuning with Hugging Face![/bold]\n")

    show_course_outline()
    console.print()

    show_examples()
    console.print()

    show_labs()
    console.print()

    simulate_training()

    console.print(
        Panel(
            "[green]Demo complete![/green]\n\n"
            "Next steps:\n"
            "1. Explore examples: [cyan]cd examples/loading && uv run python load_csv.py[/cyan]\n"
            "2. Complete Lab 1: [cyan]labs/lab-1.md[/cyan]\n"
            "3. Read the docs: [cyan]https://huggingface.co/docs/transformers[/cyan]",
            title="What's Next?",
            border_style="green",
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
