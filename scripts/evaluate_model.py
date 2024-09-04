import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

model = joblib.load('models/trained_iris_model.pkl')

data = pd.read_csv('data/processed_iris.csv')
X = data.drop('target', axis=1)
y = data['target']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

with open('models/evaluation_report.txt', 'w') as f:
    f.write(report)
print("Evaluation result has been stored in models/evaluation_report.txt")
