from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def paraphrase_with_ollama(text, num_variations=3, model="qwen3-coder"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            # {"role": "system", "content": "Paraphrase the following text. Return only the paraphrased version."},
            {
                "role": "system",
                "content": f"You are an API, you respond with plain text. You paraphrase and generate synthetic data. Do not enumerate or use any markup. Each line in a new line, you always generate {num_variations} phrases",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.7,  # Adjust for randomness
        max_tokens=150,
    )
    return response.choices[0].message.content


# Example usage
text = "The quick brown fox jumps over the lazy dog."
results = paraphrase_with_ollama(text, num_variations=5)
for line in results.split("\n"):
    if line:
        print(line)
