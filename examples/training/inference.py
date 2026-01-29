# run_inference.py
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = sys.argv[-1]

# 1. Load the trained model
model = AutoModelForSequenceClassification.from_pretrained("./" + model_name)
tokenizer = AutoTokenizer.from_pretrained("./" + model_name)

# 2. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")


# 3. Prediction function
def predict_trail_status(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():  # Don't compute gradients for inference
        outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
    return prediction, confidence


# 4. Test it!
test_cases = [
    "trails are open today",
    "park closed due to storm",
    "blankets creek is open",
    "everything is closed",
    "all trails are open yippie!!",
    "blankets creek is closed rope mill is open",
]

print("\nTrail Status Predictions:")
print("=" * 50)
for text in test_cases:
    prediction, confidence = predict_trail_status(text)
    status = "OPEN ✅" if prediction == 1 else "CLOSED ❌"
    print(f"'{text}'")
    print(f"  → {status} ({confidence:.1%} confidence)")
    print()
