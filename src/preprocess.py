import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/mnt/data/iris.csv')
df = df.dropna()
print(f"Loaded {len(df)} rows, {df.isnull().sum().sum()} nulls dropped")

train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_csv('/mnt/data/train.csv', index=False)
test.to_csv('/mnt/data/test.csv', index=False)
print(f"Train: {len(train)} rows | Test: {len(test)} rows")
