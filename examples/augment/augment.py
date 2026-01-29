from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "tuner007/pegasus_paraphrase"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer([text], truncation=True, padding="longest", return_tensors="pt")

# paraphrased = model.generate(**inputs, num_return_sequences=1, num_beams=5)
# result = tokenizer.decode(paraphrased[0], skip_special_tokens=True)
#
# print(f"Original: {text}")
# print(f"Paraphrased: {result}")

for i in range(5):
    paraphrased = model.generate(
        **inputs, num_return_sequences=1, do_sample=True, num_beams=10, temperature=0.1, top_p=0.95, top_k=50
    )
    result = tokenizer.decode(paraphrased[0], skip_special_tokens=True)

    print(f"Paraphrased: {result}")
