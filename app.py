import os
import requests
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import re
from collections import Counter
import joblib
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/developers")
def developers():
    return render_template("developers.html")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model file mapping (add your actual .pkl files here)
MODEL_FILES = {
    "naive_bayes": {
        "name": "Naive Bayes",
        "model": joblib.load(os.path.join(BASE_DIR, "naive_bayes.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "naive_bayes_vectorizer.pkl")),
        "label_encoder": joblib.load(os.path.join(BASE_DIR, "naive_bayes_label_encoder.pkl"))
    },
    "xgboost": {
        "name": "XGBoost",
        "model": joblib.load(os.path.join(BASE_DIR, "xgboost.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "xgboost_vectorizer.pkl")),
        "label_encoder": joblib.load(os.path.join(BASE_DIR, "xgboost_label_encoder.pkl"))
    },
    "catboost": {
        "name": "CatBoost",
        "model": joblib.load(os.path.join(BASE_DIR, "catboost.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "catboost_vectorizer.pkl")),
        "label_encoder": joblib.load(os.path.join(BASE_DIR, "catboost_label_encoder.pkl"))
    },
    "random_forest": {
        "name": "Random Forest",
        "model": joblib.load(os.path.join(BASE_DIR, "random_forest.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "random_forest_vectorizer.pkl")),
        "label_encoder": joblib.load(os.path.join(BASE_DIR, "random_forest_label_encoder.pkl"))
    },
    "sgb": {
        "name": "Stochastic Gradient Boosting",
        "model": joblib.load(os.path.join(BASE_DIR, "stochastic_gradient_boosting.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "stochastic_gradient_boosting_vectorizer.pkl")),
        "label_encoder": joblib.load(os.path.join(BASE_DIR, "stochastic_gradient_boosting_label_encoder.pkl"))
    }
}

# Tor proxy settings
PROXIES = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050',
}

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    full_text = soup.get_text(separator=' ')
    clean_text = ' '.join(full_text.split())
    return clean_text

def extract_keywords(text, num_keywords=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(num_keywords)]

@app.route("/", methods=["GET", "POST"])
def index():
    category = None
    error = None
    extracted_keywords = None
    selected_model = "naive_bayes"

    if request.method == "POST":
        onion_url = request.form["onion_url"].strip()
        selected_model = request.form.get("model", "naive_bayes")

        if not onion_url.startswith("http"):
            onion_url = "http://" + onion_url

        try:
            response = requests.get(onion_url, proxies=PROXIES, timeout=30)
            if response.status_code == 200:
                html = response.text
                text = extract_text_from_html(html)
                if not text or len(text.strip()) < 10:
                    error = "The website was accessed but contains little or no text to analyze."
                else:
                    keywords = extract_keywords(text, num_keywords=10)
                    extracted_keywords = ', '.join(keywords)
                    keywords_joined = ' '.join(keywords)
                    # Use the selected model
                    model_info = MODEL_FILES[selected_model]
                    vectorizer = model_info["vectorizer"]
                    model = model_info["model"]
                    label_encoder = model_info["label_encoder"]
                    X = vectorizer.transform([keywords_joined])
                    y_pred = model.predict(X)
                    category = label_encoder.inverse_transform(y_pred)[0]
            else:
                error = f"Failed to access the website (HTTP {response.status_code}). The site may be offline, restricted, or not a valid .onion service."
        except requests.exceptions.ConnectionError:
            error = "Unable to connect to the website. Please verify your Tor connection and check if the .onion site is online."
        except requests.exceptions.Timeout:
            error = "The connection to the website timed out. The site may be too slow, offline, or your Tor proxy is not working."
        except requests.exceptions.RequestException as e:
            error = f"Network error: {str(e)}"
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"

    # For GET and POST, send model names to the frontend
    model_choices = [(key, info["name"]) for key, info in MODEL_FILES.items()]
    return render_template(
        "index.html",
        category=category,
        error=error,
        extracted_keywords=extracted_keywords,
        model_choices=model_choices,
        selected_model=selected_model
    )

if __name__ == "__main__":
    app.run(debug=True)