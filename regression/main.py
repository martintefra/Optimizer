import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA_PATH = 'regression/adult.csv'


# Load the dataset
data = pd.read_csv(DATA_PATH)

# Preprocessing the data
# Encode categorical variables
label_encoders = {}
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Map income column to binary values
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

# Splitting the data into features and target variable
X = data.drop('income', axis=1)
y = data['income']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Benchmarking training time
start_time = time.time()

# Creating and training the regression model
model = LinearRegression()
model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

# Benchmarking prediction time
start_time = time.time()

# Making predictions
y_pred = model.predict(X_test)

end_time = time.time()
prediction_time = end_time - start_time

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)

print("Training Time:", training_time)
print("Prediction Time:", prediction_time)
print("Mean Squared Error:", mse)
