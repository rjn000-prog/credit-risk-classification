import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset/credit.csv")

# Target
X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

# Numerical & Categorical columns
num_cols = ['age','amount','duration','installment_rate',
            'present_residence','number_credits','people_liable']

cat_cols = ['status','credit_history','purpose','savings',
            'employment_duration','personal_status_sex','other_debtors',
            'property','other_installment_plans','housing','job',
            'telephone','foreign_worker']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train model
model.fit(X, y)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

print(" Model trained and saved successfully!")