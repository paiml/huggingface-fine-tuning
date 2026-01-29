from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

long_status = "All trails at Blankets Creek and Rope Mill are closed due to dangerous conditions including heavy rain, flooding, and potential landslides. Please check back tomorrow for updates."

print("Truncation strategies:")

# 1. Truncate from end (default)
truncated_end = tokenizer(long_status, truncation=True, max_length=20)
print("\n1. Truncate from end:")
print(f"Length: {len(truncated_end['input_ids'])}")
print(f"First 10 tokens: {tokenizer.convert_ids_to_tokens(truncated_end['input_ids'][:10])}")

# 2. Truncate from start (using only=True for older tokenizers)
truncated_start = tokenizer(long_status, truncation="only_first", max_length=20)
print("\n2. Truncate long sequences:")
print(f"Length: {len(truncated_start['input_ids'])}")

# 3. Return overflow with sliding window
print("\n3. Return overflowing tokens:")
long_result = tokenizer(long_status, truncation="only_first", max_length=30, return_overflowing_tokens=True, stride=10)
print(f"Created {len(long_result['input_ids'])} chunks")
print(f"Chunk sizes: {[len(chunk) for chunk in long_result['input_ids']]}")

# 4. Truncation on both sequences
text1 = "Blankets Creek status"
text2 = long_status
print("\n4. Truncation with text pairs:")
truncated_pair = tokenizer(text1, text2, truncation=True, max_length=20)
print(f"Total length: {len(truncated_pair['input_ids'])}")
