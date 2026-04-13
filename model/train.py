import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Define absolute path to dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', '_All_Cities_Cleaned.csv')

print("Loading dataset from:", data_path)
df = pd.read_csv(data_path)

# Drop rows with missing values for simplicity
df = df.dropna()

# We want to predict 'price'
target = 'price'
y = df[target]

# Select features
features = ['bedroom', 'area', 'bathroom', 'furnish_type', 'city', 'property_type', 'seller_type']
X = df[features]

print("Preparing pipeline...")
# Preprocessing for numerical data
numeric_features = ['bedroom', 'area', 'bathroom']
numeric_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_features = ['furnish_type', 'city', 'property_type', 'seller_type']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model extending the pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Linear Regression model...")
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model trained successfully. Test R^2 Score: {score:.4f}")

# Save the model
model_path = os.path.join(base_dir, 'model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")
