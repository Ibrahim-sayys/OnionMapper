import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import joblib

def preprocess_data(input_csv):
    print("Loading dataset...")
    data = pd.read_csv(input_csv)

    # Drop 'urls' and 'category' columns if they exist
    for col in ['urls', 'category']:
        if col in data.columns:
            print(f"Dropping column: {col}")
            data.drop(columns=[col], inplace=True)

    # Identify keyword and freq columns
    keyword_cols = [col for col in data.columns if col.startswith('keyword')]
    freq_cols = [col for col in data.columns if col.startswith('freq')]

    # Build document per row by repeating keywords by frequency
    def row_to_text(row):
        words = []
        for k_col, f_col in zip(keyword_cols, freq_cols):
            keyword = str(row[k_col]).strip().lower()
            try:
                freq = int(row[f_col])
            except (ValueError, TypeError):
                freq = 1
            if keyword and keyword != 'nan' and freq > 0:
                words.extend([keyword] * freq)
        return ' '.join(words)

    print("Building text corpus from keywords and frequencies...")
    data['document'] = data.apply(row_to_text, axis=1)

    # Drop empty documents
    data = data[data['document'].str.strip().astype(bool)]

    # Vectorize using TF-IDF
    print("Vectorizing text corpus using TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    X = vectorizer.fit_transform(data['document'])

    # Save vectorizer
    joblib.dump(vectorizer, "tfidf_vectorizer_unsupervised.pkl")
    print("TF-IDF vectorizer saved as 'tfidf_vectorizer_unsupervised.pkl'.")

    return X, data

def train_kmeans(X, n_clusters=6):
    print(f"\nTraining K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X)
    joblib.dump(kmeans, "kmeans_model.pkl")
    print("K-Means model saved as 'kmeans_model.pkl'.")
    return kmeans

def train_gmm(X, n_clusters=6):
    print(f"\nTraining GMM with {n_clusters} components...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X.toarray())
    joblib.dump(gmm, "gmm_model.pkl")
    print("GMM model saved as 'gmm_model.pkl'.")
    return gmm

def validate_clusters(X, model, is_gmm=False):
    print("\nValidating clusters...")
    labels = model.predict(X.toarray()) if is_gmm else model.labels_
    if X.shape[0] > 1:
        silhouette_avg = silhouette_score(X, labels)
        print(f"Silhouette Score: {silhouette_avg:.4f} (Higher is better, range -1 to 1)")
    else:
        print("Not enough data points to compute Silhouette Score.")
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    cluster_sizes.plot(kind="bar", color="skyblue")
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Points")
    plt.xticks(rotation=0)
    plt.show()

if __name__ == "__main__":
    input_csv = "output_keywords.csv"
    X, data = preprocess_data(input_csv)
    print("\nSelect the clustering algorithm:")
    print("1. K-Means")
    print("2. Gaussian Mixture Model (GMM)")
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice == "1":
        kmeans = train_kmeans(X, n_clusters=6)
        validate_clusters(X, kmeans)
    elif choice == "2":
        gmm = train_gmm(X, n_clusters=6)
        validate_clusters(X, gmm, is_gmm=True)
    else:
        print("Invalid choice! Please run the program again and select 1 or 2.")
