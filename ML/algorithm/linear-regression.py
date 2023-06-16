import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data from the CSV file
data = pd.read_csv('./ML/algorithm/test.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train.values.reshape(-1, 1), y_train)


print('Coefficients: ', model.coef_)


y_pred = model.predict(X_test.values.reshape(-1, 1))

print('Predicted values: ', y_pred)

# Plot the regression line and test data
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
