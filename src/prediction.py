import joblib
import pandas as pd
 
model = joblib.load("../model/loan_default.pkl")
 
# create a test sample data
test_df = pd.read_csv("../data/x_test_sample.csv")
 
x_test_sample = test_df.sample(1)
 
yhat = model.predict(x_test_sample)
 
print(f"Prediction: {yhat}")
 
