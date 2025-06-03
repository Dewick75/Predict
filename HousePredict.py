import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load dataset here
df = pd.read_csv('random_house_prices 2025 may assignment.csv')  


# 2. Shape of dataset
print("Shape:", df.shape)

# 3. Preprocessing 
df = df.dropna()

# 4. Split dataset
X = df.drop('Price', axis=1) 
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R² Score:", r2)

# 7. Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted')
plt.show()
