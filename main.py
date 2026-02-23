import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Setting the Data
data = pd.read_csv('train.csv')

print(f"Data loaded: {data.shape}")

# Setting the values and colums
num_cols = data.select_dtypes(include='number').columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())
cat_cols = data.select_dtypes(include='str').columns
data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

data = pd.get_dummies(data , drop_first=True)

# Setting the x and y-axis for our data
x_axis = data.drop("SalePrice", axis=1)
y_axis = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=0.2, random_state=42)

# truing our AI
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Agglutinating the house prices
y_pred = model.predict(X_test)
print(f"MAE: ${mean_absolute_error(y_test, y_pred)}:,0f")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# Saving the model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved!")
