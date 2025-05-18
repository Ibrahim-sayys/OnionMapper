import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

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
    X_scaled, y, test_size=0.3, random_state=42
)

# Feature Extraction: PCA and ICA (2 components each)
pca = PCA(n_components=3, random_state=42)
ica = FastICA(n_components=3, random_state=42)

pca_train = pca.fit_transform(X_train)
ica_train = ica.fit_transform(X_train)

pca_test = pca.transform(X_test)
ica_test = ica.transform(X_test)

# Combine extracted features only
X_train_ext = np.hstack((pca_train, ica_train))
X_test_ext = np.hstack((pca_test, ica_test))

# Train Logistic Regression on extracted features
clf = LogisticRegression(random_state=42, max_iter=500)
clf.fit(X_train_ext, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_ext)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression Accuracy on Extracted Features: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
