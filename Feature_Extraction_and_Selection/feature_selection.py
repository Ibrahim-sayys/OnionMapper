import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Titanic dataset
df = pd.read_csv('titanic.csv')

# Display basic info about data
print("Head of dataset:")
print(df.head())

print("\nInfo about dataset:")
print(df.info())

print("\nStatistical description:")
print(df.describe())

# Target and features
TARGET = 'Survived'

# Basic preprocessing
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Split features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Apply Random Forest for Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances (Random Forest):")
print(feature_importance_df)

# Select top 1 feature based on importance
top_features = feature_importance_df.head(1)['Feature'].tolist()
print(f"\nSelected Features for Training: {top_features}")

# Train with selected features only
X_train_selected = X_train[:, feature_importance_df.index[:3]]
X_test_selected = X_test[:, feature_importance_df.index[:3]]

# Train Logistic Regression with selected features
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42, max_iter=500)
clf.fit(X_train_selected, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression Accuracy on Selected Features: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
