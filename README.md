import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Mock historical EV charging demand data (Week 1)
# Let's say we have 7 days of data
data = {
    'day': np.arange(1, 8),  # Days 1 to 7
    'charging_sessions': [120, 135, 160, 145, 170, 180, 200]  # Example sessions per day
}
df = pd.DataFrame(data)

# Prepare data for prediction (Week 2: Days 8 to 14)
X_train = df[['day']]
y_train = df['charging_sessions']
X_pred = pd.DataFrame({'day': np.arange(8, 15)})

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict charging demand for Week 2
predictions = model.predict(X_pred)

# Output results
week2_df = X_pred.copy()
week2_df['predicted_charging_sessions'] = predictions.astype(int)
print("Week 2 EV Charging Demand Prediction:")
print(week2_df)

# Optional: Plot the results
plt.figure(figsize=(10,6))
plt.plot(df['day'], df['charging_sessions'], label='Week 1 Actual', marker='o')
plt.plot(week2_df['day'], week2_df['predicted_charging_sessions'], label='Week 2 Predicted', marker='x')
plt.xlabel('Day')
plt.ylabel('Charging Sessions')
plt.title('EV Charging Demand Prediction')
plt.legend()
plt.grid(True)
plt.show()
