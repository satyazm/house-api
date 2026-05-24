from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

data = {
    "area":[1000,1200,1500,1800,2000],
    "bedrooms":[2,2,3,3,4],
    "price":[50,60,75,90,100]
}

df = pd.DataFrame(data)

X = df[["area","bedrooms"]]
y = df["price"]

model = LinearRegression()

model.fit(X,y)

joblib.dump(model,"house_model.pkl")

print("Model saved successfully")