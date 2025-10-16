# train_model.py
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("loan_data.csv")
df.columns = df.columns.str.strip().str.lower()

# -----------------------------
# Define columns
# -----------------------------
categorical_cols = ['person_gender', 'person_education', 'person_home_ownership',
                    'loan_intent', 'previous_loan_defaults_on_file']
numeric_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                'credit_score']
target_col = 'loan_status'


# -----------------------------
# Handle missing values
# -----------------------------
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------
# Encode categorical columns
# -----------------------------
encoders = {}
for col in categorical_cols + [target_col]:
    le = LabelEncoder()
    df[col] = df[col].astype(str).str.lower()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Transform + Scale numeric columns
# -----------------------------
pt = PowerTransformer()
df[numeric_cols] = pt.fit_transform(df[numeric_cols])

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# Train-Test Split
# -----------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# Save the exact column order
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))

# -----------------------------
# Save all components
# -----------------------------
pickle.dump(rf, open("loan_model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(pt, open("power_transformer.pkl", "wb"))
pickle.dump(list(X.columns), open("feature_columns.pkl", "wb"))

print("✅ Model training complete. Files saved:")
print("loan_model.pkl, encoders.pkl, scaler.pkl, power_transformer.pkl, feature_columns.pkl")

