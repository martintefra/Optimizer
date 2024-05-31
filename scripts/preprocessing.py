import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

### Adult dataset ###

# Load dataset
df_adults = pd.read_csv('datasets/adult.csv')

# Preprocessing the data
# Encode categorical variables
label_encoders = {}
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df_adults[col] = label_encoders[col].fit_transform(df_adults[col])

# Map income column to binary values
df_adults['income'] = df_adults['income'].map({'<=50K': 0, '>50K': 1})

# Splitting the data into features and target variable
X = df_adults.drop('income', axis=1)
y = df_adults['income']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Save the split data for reproducibility
train_data = pd.concat([pd.DataFrame(X_train_scaled, columns=X.columns), pd.Series(y_train.values, name='income')], axis=1)
test_data = pd.concat([pd.DataFrame(X_test_scaled, columns=X.columns), pd.Series(y_test.values, name='income')], axis=1)

train_data.to_csv('data/adult_train.csv', index=False)
test_data.to_csv('data/adult_test.csv', index=False)


### IMDB dataset ###

# Load dataset
df_imdb = pd.read_csv('datasets/imdb.csv')

# Vectorize the reviews
# vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# X = vectorizer.fit_transform(df_imdb['review']).toarray()
# y = (df_imdb['sentiment'] == 'positive').astype(int)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Convert data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.bool)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.bool)

# # Save the split data for reproducibility
# train_data = pd.concat([pd.DataFrame(X_train, columns=vectorizer.get_feature_names_out()), pd.Series(y_train, name='sentiment')], axis=1)
# test_data = pd.concat([pd.DataFrame(X_test, columns=vectorizer.get_feature_names_out()), pd.Series(y_test, name='sentiment')], axis=1)

train_data.to_csv('data/imdb_train.csv', index=False)
test_data.to_csv('data/imdb_test.csv', index=False)
