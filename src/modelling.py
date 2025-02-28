import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")

# Encode categorical features (using LabelEncoder)
categorical_cols = ["CourseCategory"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders for later use
joblib.dump(label_encoders, "label_encoders.pkl")

# Define features and target
X = df.drop(columns=["CourseCompletion"])
y = df["CourseCompletion"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical columns
numerical_cols = list(set(X.columns) - set(categorical_cols))

# Preprocessing pipelines
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical pipeline (using SimpleImputer only)
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Combine numerical and categorical pipelines
preprocessor = ColumnTransformer([
    ("num", num_transformer, numerical_cols),
    ("cat", cat_transformer, categorical_cols)
])

# Define the model pipeline
rfc_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced_subsample", random_state=42))
])

# Define hyperparameter grid for tuning
param_grid = {
    "classifier__n_estimators": [100, 150, 200],
    "classifier__max_depth": [5, 10],
    "classifier__min_samples_split": [5],
    "classifier__min_samples_leaf": [1],
    "classifier__max_features": ["sqrt"],
    "classifier__bootstrap": [True]
}

# Perform hyperparameter tuning
model = GridSearchCV(rfc_pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

# Save the best model
joblib.dump(model.best_estimator_, "trained_model.pkl")

# Save the fitted preprocessor
preprocessor_fitted = model.best_estimator_.named_steps["preprocessor"]
joblib.dump(preprocessor_fitted, "preprocessor.pkl")

# Get the support of the RFE.
support = model.best_estimator_.named_steps['classifier'].feature_importances_

# get all of the column names.
all_cols = numerical_cols + categorical_cols

# filter the columns based on the support.
selected_features = [col for col, supported in zip(all_cols, support) if supported > 0]

# save the selected features.
pd.DataFrame({"Features": selected_features}).to_csv("selected_features.csv", index=False)

# Predictions and evaluation on the test set
best_model = model.best_estimator_
y_probs = best_model.predict_proba(X_test)[:, 1]
threshold = 0.35
y_preds_adjusted = (y_probs >= threshold).astype(int)
print("Classification Report:\n", classification_report(y_test, y_preds_adjusted))