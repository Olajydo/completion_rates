import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder

# Load preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")

# Encode categorical features
categorical_cols = ["CourseCategory"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Define features and target
target_col = "CourseCompletion"
X, y = df.drop(columns=[target_col]), df[target_col]

# Feature Selection with RFE
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Save selected features correctly
selected_features = X.columns[rfe.support_].tolist()
pd.DataFrame({"Features": selected_features}).to_csv("selected_features.csv", index=False)

# Save processed datasets
df[selected_features].to_csv("selected_data.csv", index=False)
df[target_col].to_csv("target_data.csv", index=False)