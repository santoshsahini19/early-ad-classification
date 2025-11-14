import pandas as pd
from sklearn.model_selection import train_test_split

FULL_CSV = r"D:\projects\research\preprocessed_metadata_128.csv"

df = pd.read_csv(FULL_CSV)

# Split 70% train, 15% val, 15% test
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

train_df.to_csv(r"D:\projects\research\train_metadata_128.csv", index=False)
val_df.to_csv(r"D:\projects\research\val_metadata_128.csv", index=False)
test_df.to_csv(r"D:\projects\research\test_metadata_128.csv", index=False)

print("Splits created:")
print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
