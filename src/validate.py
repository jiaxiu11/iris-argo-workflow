import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

test_df = pd.read_csv('/mnt/data/test.csv')
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

with open('/mnt/data/model.pkl', 'rb') as f:
    model = pickle.load(f)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Validation accuracy: {accuracy:.4f}")

with open('/tmp/accuracy.txt', 'w') as f:
    f.write(str(round(accuracy, 4)))
