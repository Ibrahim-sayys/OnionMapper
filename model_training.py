import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib

# Function to preprocess the dataset
def preprocess_data(input_csv):
    print("Loading dataset...")
    data = pd.read_csv(input_csv)

    # Drop the 'Urls' column
    if 'Urls' in data.columns:
        print("Dropping the 'Urls' column")
        data.drop(columns=['Urls'], inplace=True)

    # Shuffle the dataset to randomize the rows
    print("Shuffling the dataset...")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop rows where Keywords or Category is missing
    print("Dropping rows with missing Keywords or Category...")
    data.dropna(subset=['Keywords', 'Category'], inplace=True)

    # Clean the Keywords column
    print("Cleaning the Keywords column...")
    data['Keywords'] = data['Keywords'].apply(lambda x: ' '.join(x.split(', ')))

    # Encode the Category column as labels
    print("Encoding the Category column as numerical labels...")
    label_encoder = LabelEncoder()
    data['Category_Label'] = label_encoder.fit_transform(data['Category'])

    # Vectorize the Keywords column using TF-IDF
    print("Vectorizing the Keywords column using TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(data['Keywords'])
    y = data['Category_Label']

    # Visualize class distribution before balancing
    print("\nClass distribution before balancing:")
    print(pd.Series(y).value_counts())
    visualize_class_distribution(pd.Series(y), title="Class Distribution Before Balancing")

    # Ask the user which imbalancing technique to use
    print("\nWhich imbalancing technique would you like to use?")
    print("[1] SMOTE")
    print("[2] Random Oversampling")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\nBalancing the dataset using SMOTE...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif choice == "2":
        print("\nBalancing the dataset using Random Oversampling...")
        ros = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = ros.fit_resample(X, y)
    else:
        print("Invalid choice. Exiting...")
        exit()

    # Visualize class distribution after balancing
    print("\nClass distribution after balancing:")
    print(pd.Series(y_balanced).value_counts())
    visualize_class_distribution(pd.Series(y_balanced), title="Class Distribution After Balancing")

    return X_balanced, y_balanced

# Function to visualize class distribution
def visualize_class_distribution(y, title):
    plt.figure(figsize=(8, 6))
    y.value_counts().plot(kind="bar", color="skyblue")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    #plt.show()

# Function to train Random Forest
def train_random_forest(X, y):
    print("\nTraining the Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    evaluate_model(classifier, X_test, y_test, "random_forest.pkl")

# Function to train Naive Bayes
def train_naive_bayes(X, y):
    print("\nTraining the Naive Bayes Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = MultinomialNB()  # Multinomial Naive Bayes for text data
    classifier.fit(X_train, y_train)

    evaluate_model(classifier, X_test, y_test, "naive_bayes.pkl")

# Function to train XGBoost
def train_xgboost(X, y):
    print("\nTraining the XGBoost Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=6)
    classifier.fit(X_train, y_train)

    evaluate_model(classifier, X_test, y_test, "xgboost.pkl")

# Function to train CatBoost
def train_catboost(X, y):
    print("\nTraining the CatBoost Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = CatBoostClassifier(random_state=42, iterations=100, learning_rate=0.1, depth=6, verbose=False)  # Use 'False' instead of 0
    classifier.fit(X_train, y_train)

    evaluate_model(classifier, X_test, y_test, "catboost.pkl")

# Function to train Stochastic Gradient Boosting
def train_stochastic_gradient_boosting(X, y):
    print("\nTraining the Stochastic Gradient Boosting Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,  # Enables stochastic behavior
        random_state=42
    )
    classifier.fit(X_train, y_train)

    evaluate_model(classifier, X_test, y_test, "stochastic_gradient_boosting.pkl")

# Function to evaluate the model
def evaluate_model(classifier, X_test, y_test, model_filename):
    print("\nEvaluating the model...")
    y_pred = classifier.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Display the confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    #plt.show()

    # Save the trained model
    joblib.dump(classifier, model_filename)
    print(f"\nModel saved as {model_filename}.")

# Main function
if __name__ == "__main__":
    # Input CSV file path
    input_csv = "output_keywords.csv"  # Ensure this file exists

    # Preprocess the data
    X_balanced, y_balanced = preprocess_data(input_csv)

    # Ask the user which model to train
    print("\nWhich model would you like to train?")
    print("[1] Random Forest")
    print("[2] Naive Bayes")
    print("[3] XGBoost")
    print("[4] CatBoost")
    print("[5] Stochastic Gradient Boosting")  # Added this option
    choice = input("Enter 1, 2, 3, 4 or 5: ").strip()

    if choice == "1":
        train_random_forest(X_balanced, y_balanced)
    elif choice == "2":
        train_naive_bayes(X_balanced, y_balanced)
    elif choice == "3":
        train_xgboost(X_balanced, y_balanced)
    elif choice == "4":
        train_catboost(X_balanced, y_balanced)
    elif choice == "5":
        train_stochastic_gradient_boosting(X_balanced, y_balanced)
    else:
        print("Invalid choice. Exiting...")
        exit()