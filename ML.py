import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("electricity_dataset_150.csv")

X = df[['Number_of_Appliances', 'AC_Usage_Hours', 'Family_Members',
        'Average_Temperature', 'Daily_Usage_Duration']]
y = df['Monthly_Consumption_kWh']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# LINEAR REGRESSION MODEL
# -------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# -------------------------------
# RANDOM FOREST MODEL
# -------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# -------------------------------
# PLOTS
# -------------------------------

# ----- Linear Regression Plot -----
plt.figure(figsize=(8, 5))
plt.scatter(y_test, lr_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--')
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title("Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.show()

# ----- Random Forest Plot -----
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--')
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title("Random Forest: Actual vs Predicted")
plt.grid(True)
plt.show()

# -------------------------------
# PRINT PERFORMANCE METRICS
# -------------------------------
print("===== Linear Regression Metrics =====")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R² Score:", r2_score(y_test, lr_pred))

print("\n===== Random Forest Metrics =====")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R² Score:", r2_score(y_test, rf_pred))
