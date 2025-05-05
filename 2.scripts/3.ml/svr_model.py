# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# ===============================
# 1. DATA LOADING AND PREPARATION
# ===============================
# File Loading
df_weather = pd.read_csv("../resources/df_weather_cleaned.csv")
df_sales = pd.read_csv("../resources/fact_sales.csv")
df_dates = pd.read_csv("../resources/dim_date.csv")

# Merge sales with dates to add temporal variables
df_sales = df_sales.merge(df_dates, left_on="created_date_key", right_on="surrogate_key", how="left")

# Average weather per day (national average)
df_weather_avg = df_weather.groupby('date')[['TMAX_C', 'TMIN_C', 'PRCP_MM']].mean().reset_index()

# Ensure date formats
df_sales['date'] = pd.to_datetime(df_sales['date'])
df_weather_avg['date'] = pd.to_datetime(df_weather_avg['date'])

# Merge sales with weather
df = df_sales.merge(df_weather_avg, on='date', how='left')


# ===============================
# 2. FEATURES & TARGET
# ===============================

# Define the predictor variables (X) and the target variable (y)
X = df[['TMAX_C', 'TMIN_C', 'PRCP_MM', 'year', 'month', 'day', 'day_of_week', 'is_weekend']]
y = df['total_price']

# ===============================
# 3. TRAIN TEST SPLIT
# ===============================

# Split the dataset into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# ===============================
# 4. BASIC TRAINING AND EVALUATION
# ===============================

# Baseline SVR model
model = SVR()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"SVR - RMSE (base): {rmse:.2f}")
print(f"SVR - R2 Score (base): {r2:.4f}")
print(f"SVR - MAE (base): {mae:.2f}")

# ===============================
# 5. GRID SEARCH
# ===============================

# Define the hyperparameter grid to optimize the model
param_grid = {
    'C': [0.1, 1, 10],# Model regularization parameter
    'epsilon': [0.1, 1, 5],
    'kernel': ['linear', 'rbf', 'poly'] # Lineal, radial and polinomial kernel
}

# Configure GridSearchCV with cross-validation
grid_search = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Retrieve the best model found
best_model = grid_search.best_estimator_
print("✅ Best SVR parameters:", grid_search.best_params_)

# Final evaluation
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Optimized SVR RMSE: {rmse:.2f}")
print(f"Optimized SVR R2 Score: {r2:.4f}")
print(f"Optimized SVR MAE: {mae:.2f}")

# ===============================
# 6. CROSS VALIDATION
# ===============================

# Cross-validation RMSE
cv_rmse_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
cv_rmse = -cv_rmse_scores
print(f"📊 SVR Cross-Validation RMSE scores: {cv_rmse}")
print(f"📉 Mean CV RMSE: {cv_rmse.mean():.2f}")
print(f"📈 Std CV RMSE: {cv_rmse.std():.2f}")

# Cross-validation MAE
cv_mae_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_mae = -cv_mae_scores
print(f"📊 SVR Cross-Validation MAE scores: {cv_mae}")
print(f"📉 Mean CV MAE: {cv_mae.mean():.2f}")
print(f"📈 Std CV MAE: {cv_mae.std():.2f}")

# ===============================
# 7. VISUALIZATIONS
# ===============================

# Real vs Predicted
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Orders")
plt.ylabel("Predicted Orders")
plt.title("Actual vs Predicted Orders")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30, color="salmon")
plt.axvline(0, color='black', linestyle='--')
plt.title("Distribución de residuos")
plt.xlabel("Error residual = real - predicho")
plt.grid(True)
plt.tight_layout()
plt.show()

# Create results summary DataFrame
results = pd.DataFrame({
    'Model': ['Base', 'Optimized'],
    'RMSE': [np.sqrt(mean_squared_error(y_test, model.predict(X_test))), rmse],
    'R2': [r2_score(y_test, model.predict(X_test)), r2],
    'MAE': [mean_absolute_error(y_test, model.predict(X_test)), mae]
})

print(results)
