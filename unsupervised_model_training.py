import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
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

    # Drop rows where Keywords are missing
    print("Dropping rows with missing Keywords...")
    data.dropna(subset=['Keywords'], inplace=True)

    # Clean the Keywords column
    print("Cleaning the Keywords column...")
    data['Keywords'] = data['Keywords'].apply(lambda x: ' '.join(x.split(', ')))

    # Vectorize the Keywords column using TF-IDF
    print("Vectorizing the Keywords column using TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))  # Simple unigrams since there's only one feature
    X = vectorizer.fit_transform(data['Keywords'])

    # Save the TF-IDF vectorizer for future use
    joblib.dump(vectorizer, "tfidf_vectorizer_unsupervised.pkl")
    print("TF-IDF vectorizer saved as 'tfidf_vectorizer_unsupervised.pkl'.")

    return X, data


# Function to train K-Means clustering
def train_kmeans(X, n_clusters=6):
    print(f"\nTraining K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X)

    # Save the trained K-Means model
    joblib.dump(kmeans, "kmeans_model.pkl")
    print("K-Means model saved as 'kmeans_model.pkl'.")

    return kmeans


# Function to train Gaussian Mixture Model (GMM)
def train_gmm(X, n_clusters=6):
    print(f"\nTraining GMM with {n_clusters} components...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X.toarray())  # Convert sparse matrix to dense for GMM

    # Save the trained GMM model
    joblib.dump(gmm, "gmm_model.pkl")
    print("GMM model saved as 'gmm_model.pkl'.")

    return gmm


# Function to validate clusters
def validate_clusters(X, model, is_gmm=False):
    print("\nValidating clusters...")

    # Get cluster labels
    labels = model.predict(X.toarray()) if is_gmm else model.labels_

    # Calculate Silhouette Score
    if X.shape[0] > 1:  # Ensure enough data points for silhouette evaluation
        silhouette_avg = silhouette_score(X, labels)
        print(f"Silhouette Score: {silhouette_avg:.4f} (Higher is better, range -1 to 1)")
    else:
        print("Not enough data points to compute Silhouette Score.")

    # Visualize cluster sizes
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    cluster_sizes.plot(kind="bar", color="skyblue")
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Points")
    plt.xticks(rotation=0)
    plt.show()


# Main function
if __name__ == "__main__":
    # Input CSV file path
    input_csv = "output_keywords.csv"  # Ensure this file exists

    # Preprocess the data
    X, data = preprocess_data(input_csv)

    # Menu to select clustering algorithm
    print("\nSelect the clustering algorithm:")
    print("1. K-Means")
    print("2. Gaussian Mixture Model (GMM)")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        # Train and validate K-Means
        kmeans = train_kmeans(X, n_clusters=6)
        validate_clusters(X, kmeans)
    elif choice == "2":
        # Train and validate GMM
        gmm = train_gmm(X, n_clusters=6)
        validate_clusters(X, gmm, is_gmm=True)
    else:
        print("Invalid choice! Please run the program again and select 1 or 2.")