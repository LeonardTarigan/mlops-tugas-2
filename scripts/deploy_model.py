import joblib

model = joblib.load('models/trained_iris_model.pkl')

joblib.dump(model, 'models/deployed_iris_model.pkl')
print("Model has been stored  for deployment in models/deployed_iris_model.pkl")