from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('/mnt/data/iris.csv', index=False)
print(f"Saved {len(df)} rows to /mnt/data/iris.csv")
