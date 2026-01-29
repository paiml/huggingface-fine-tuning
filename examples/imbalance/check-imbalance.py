import pandas as pd

# Load your CSV with pandas
# df = pd.read_csv('status.csv')
df = pd.read_csv("imbalanced.csv")

# 1. Check class distribution in one line
print("Class distribution:")
print(df["Blankets_Creek"].value_counts())
print("\nRatios:")
print(df["Blankets_Creek"].value_counts(normalize=True))

# 2. Detect imbalance
open_count = df["Blankets_Creek"].sum()
closed_count = len(df) - open_count

print(f"\nOpen trails: {open_count}")
print(f"Closed trails: {closed_count}")

# 3. Simple decision
if open_count < closed_count / 2:
    print("✓ Need OVERSAMPLING for open trails")
elif closed_count < open_count / 2:
    print("✓ Need OVERSAMPLING for closed trails")
else:
    print("✓ Dataset is balanced")
