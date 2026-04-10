import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('/mnt/data/train.csv')
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('/mnt/data/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to /mnt/data/model.pkl")
