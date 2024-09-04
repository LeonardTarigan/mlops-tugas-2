import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

data.iloc[:, :-1] = (data.iloc[:, :-1] - data.iloc[:, :-1].min()) / (data.iloc[:, :-1].max() - data.iloc[:, :-1].min())

data.to_csv('data/processed_iris.csv', index=False)
print("Data has been processed and stored in data/processed_iris.csv")
