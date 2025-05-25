import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
import pandas as pd

def preprocess_data(input_csv):
    print("Loading dataset...")
    data = pd.read_csv(input_csv)

    # Standardize column names
    data.columns = data.columns.str.strip().str.lower()

    # Drop the 'urls' column if it exists
    if 'urls' in data.columns:
        print("Dropping the 'urls' column...")
        data.drop(columns=['urls'], inplace=True)

    # Shuffle the dataset
    print("Shuffling the dataset...")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Select only keyword columns (exclude freq columns)
    keyword_cols = [col for col in data.columns if col.startswith('keyword') and not col.startswith('freq')]

    if not keyword_cols:
        raise ValueError("No keyword columns found!")

    # Treat empty strings in keyword columns as NaN
    for col in keyword_cols:
        data[col] = data[col].astype(str).str.strip().str.lower().replace('', pd.NA)

    # Clean category
    data['category'] = data['category'].astype(str).str.strip().str.lower()

    # Count missing values before filling
    total_missing_before = data[keyword_cols].isna().sum().sum()
    print(f"\n Total missing (empty) keyword values before filling: {total_missing_before}")

    # Fill missing keyword columns using mode per category
    for col in keyword_cols:
        data[col] = data.groupby('category')[col].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )

    total_missing_after = data[keyword_cols].isna().sum().sum()
    print(f" Total missing keyword values after filling: {total_missing_after}")
    print(f"Total filled: {total_missing_before - total_missing_after}\n")

    # Combine all keyword columns into one
    print(" Combining keyword columns into a single text column...")
    data['Keywords'] = data[keyword_cols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

    # Encode category labels
    print("Encoding category labels...")
    label_encoder = LabelEncoder()
    data['category_label'] = label_encoder.fit_transform(data['category'])

    joblib.dump(label_encoder, "label_encoder.pkl")
    print("LabelEncoder saved as label_encoder.pkl.")

    # TF-IDF Vectorization
    print(" Vectorizing keywords using TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(data['Keywords'])
    y = data['category_label']

    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Vectorizer saved as vectorizer.pkl.")

    # Show original class distribution
    print("\n Class distribution before balancing:")
    print(pd.Series(y).value_counts())

    # Ask user for balancing technique
    print("\nWhich balancing technique would you like to use?")
    print("[1] SMOTE")
    print("[2] Random Oversampling")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print(" Applying SMOTE balancing...")
        X_balanced, y_balanced = SMOTE(random_state=42).fit_resample(X, y)
    elif choice == "2":
        print("Applying Random Oversampling...")
        X_balanced, y_balanced = RandomOverSampler(random_state=42).fit_resample(X, y)
    else:
        print(" Invalid choice. Exiting...")
        exit()

    # Show new class distribution
    print("\n Class distribution after balancing:")
    print(pd.Series(y_balanced).value_counts())

    return X_balanced, y_balanced, label_encoder, vectorizer


# Function to visualize class distribution
def visualize_class_distribution(y, title):
    plt.figure(figsize=(8, 6))
    y.value_counts().plot(kind="bar", color="skyblue")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    #plt.show()
# ... [keep all previous imports and code unchanged above this point]

# Modified evaluate_model function to accept save_model flag
def evaluate_model(classifier, X_test, y_test, model_filename, save_model):  #  Added save_model parameter
    print("\nEvaluating the model...")
    y_pred = classifier.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    if save_model:  # Save model only if user wants
        joblib.dump(classifier, model_filename)
        print(f"\nModel saved as {model_filename}.")

# Updated all train functions to include save_model parameter
def train_random_forest(X, y, save_model):
    print("\nTraining the Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    evaluate_model(classifier, X_test, y_test, "random_forest.pkl", save_model)

def train_naive_bayes(X, y, save_model):
    print("\nTraining the Naive Bayes Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    evaluate_model(classifier, X_test, y_test, "naive_bayes.pkl", save_model)

def train_xgboost(X, y, save_model):
    print("\nTraining the XGBoost Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=6)
    classifier.fit(X_train, y_train)
    evaluate_model(classifier, X_test, y_test, "xgboost.pkl", save_model)

def train_catboost(X, y, save_model):
    print("\nTraining the CatBoost Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = CatBoostClassifier(random_state=42, iterations=100, learning_rate=0.1, depth=6, verbose=False)
    classifier.fit(X_train, y_train)
    evaluate_model(classifier, X_test, y_test, "catboost.pkl", save_model)

def train_stochastic_gradient_boosting(X, y, save_model):
    print("\nTraining the Stochastic Gradient Boosting Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42
    )
    classifier.fit(X_train, y_train)
    evaluate_model(classifier, X_test, y_test, "stochastic_gradient_boosting.pkl", save_model)


# Main loop with rerun support
if __name__ == "__main__":
    while True:  #  Added loop for rerun
        input_csv = "output_keywords.csv"
        X_balanced, y_balanced, label_encoder, vectorizer = preprocess_data(input_csv)

        print("\nWhich model would you like to train?")
        print("[1] Random Forest")
        print("[2] Naive Bayes")
        print("[3] XGBoost")
        print("[4] CatBoost")
        print("[5] Stochastic Gradient Boosting")
        choice = input("Enter 1, 2, 3, 4 or 5: ").strip()

        # Ask user if they want to save the model
        save_prompt = input("\n Do you want to save the model as a pickle file? (yes/no): ").strip().lower()
        save_model = save_prompt == "yes"

        if choice == "1":
            train_random_forest(X_balanced, y_balanced, save_model)
        elif choice == "2":
            train_naive_bayes(X_balanced, y_balanced, save_model)
        elif choice == "3":
            train_xgboost(X_balanced, y_balanced, save_model)
        elif choice == "4":
            train_catboost(X_balanced, y_balanced, save_model)
        elif choice == "5":
            train_stochastic_gradient_boosting(X_balanced, y_balanced, save_model)
        else:
            print("Invalid choice. Exiting...")
            exit()

        #  Ask if user wants to run again
        rerun = input("\n Do you want to run the script again? (yes/no): ").strip().lower()
        if rerun != "yes":
            print("Exiting. Have a great day!")
            break
