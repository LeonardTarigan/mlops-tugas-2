import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv('data/processed_iris.csv')
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

joblib.dump(model, 'models/trained_iris_model.pkl')
print("Model has been trained and stored in models/trained_iris_model.pkl")
