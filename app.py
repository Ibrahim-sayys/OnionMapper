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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, vectorizer, and label encoder
model = joblib.load(os.path.join(BASE_DIR, "naive_bayes.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

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
    # Only count words of length >=4, not stopwords, as in training
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(num_keywords)]

@app.route("/", methods=["GET", "POST"])
def index():
    category = None
    error = None
    extracted_keywords = None

    if request.method == "POST":
        onion_url = request.form["onion_url"].strip()
        if not onion_url.startswith("http"):
            onion_url = "http://" + onion_url  # add scheme if missing

        try:
            response = requests.get(onion_url, proxies=PROXIES, timeout=30)
            if response.status_code == 200:
                html = response.text
                text = extract_text_from_html(html)
                keywords = extract_keywords(text, num_keywords=10)
                extracted_keywords = ', '.join(keywords)
                # Join keywords for the vectorizer (space-separated)
                keywords_joined = ' '.join(keywords)
                X = vectorizer.transform([keywords_joined])
                y_pred = model.predict(X)
                category = label_encoder.inverse_transform(y_pred)[0]
            else:
                error = f"Failed to crawl site (HTTP {response.status_code})"
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template("index.html", category=category, error=error, extracted_keywords=extracted_keywords)

if __name__ == "__main__":
    app.run(debug=True)