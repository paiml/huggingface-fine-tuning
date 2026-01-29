from transformers import AutoTokenizer

# Load tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example sentences from your trail data
sentences = [
    "All trails are closed due to wet conditions",
    "Blankets Creek is open, Rope Mill is closed",
    "Check website for updates",
]

# Basic tokenization
print("Basic tokenization:")
tokens = tokenizer(sentences[0])
print(f"Input: {sentences[0]}")
print(f"Token IDs: {tokens['input_ids'][:5]}...")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'])[:5]}...")

# With padding (for batch processing)
print("\nWith padding (for equal length):")
padded = tokenizer(sentences, padding=True)
print(f"Padded lengths: {[len(ids) for ids in padded['input_ids']]}")
print(f"All same length: {len(set(len(ids) for ids in padded['input_ids'])) == 1}")
