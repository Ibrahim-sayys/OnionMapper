# OnionMapper 🌌

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)

## 🌟 Overview
**OnionMapper** is a state-of-the-art web application for classifying and analyzing `.onion` websites on the Dark Web using advanced machine learning.  
With a beautiful UI and seamless Tor integration, OnionMapper empowers researchers, analysts, and enthusiasts to explore hidden services **responsibly and anonymously**.

> This project is designed for researchers, cybersecurity professionals, and enthusiasts exploring the hidden corners of the web responsibly.

## 🚀 Features
- 🔍 **Onion Site Scraping**: Crawl `.onion` websites securely via the Tor network.
- ✨ **Keyword Extraction**: Instantly extract the most relevant keywords from dark web sites.
- 🤖 **Multi-Model Classification**: Choose from several trained ML models—Naive Bayes, XGBoost, CatBoost, Random Forest, and Stochastic Gradient Boosting—for high-accuracy categorization.
- 🖥️ **Modern Web UI**: Clean, responsive, and animated interface built with Flask and Bootstrap 5.
- 🛡️ **Privacy-First**: All requests routed through Tor; no site data is stored.
- 🧑‍💻 **Developer & About Pages**: Animated and interactive "About Project" and "Developers" sections.
- 🌐 **Tor Integration**: All scraping and browsing is routed through your local Tor proxy for anonymity.

## ⚠️ Disclaimer
This project is strictly for **educational and research purposes**.  
**Do not use OnionMapper to access illegal content or for any unlawful purposes.** Always operate responsibly and respect local laws.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **Frontend**: Bootstrap 5, custom CSS & animation
- **ML Libraries**: 
  - `scikit-learn`, `xgboost`, `catboost`, `joblib`, `nltk`
- **Scraping & Parsing**: 
  - `requests`, `beautifulsoup4`
- **Proxy**: Tor (`socks5h://127.0.0.1:9050`)
- **Other**: 
  - `nltk` for stopword filtering and keyword extraction

## 🎯 Getting Started

### Prerequisites
1. Python 3.8+
2. [Tor service](https://2019.www.torproject.org/docs/tor-doc-unix.html.en) running locally (default port 9050)
3. Required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ibrahim-sayys/OnionMapper.git
   cd OnionMapper
   ```
2. **Run Tor:**
   ```bash
   sudo apt-get install tor
   tor
   ```
3. **Run the Flask web app:**
   ```bash
   python app.py
   ```
   The web interface will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. **(Optional) Download NLTK stopwords:**
   The app will attempt to download stopwords automatically, but you can manually do:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

### Usage
- Open the OnionMapper web app in your browser.
- Enter any `.onion` URL, select a machine learning model, and click **Predict Category**.
- See the predicted category and extracted keywords, and explore the About/Developers pages for more info.

## 📈 Roadmap
- [x] Tor-based secure web scraping
- [x] Fast keyword extraction from scraped content
- [x] Multiple ML models for classification
- [x] Interactive, animated web UI with model selection
- [x] Animated About & Developer sections
- [x] Error handling and user-friendly alerts
- [ ] Visualize keyword/category trends
- [ ] Automated reporting for classified sites
- [ ] Docker support & deployment guides

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the project
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request

## 🛡️ Security
If you find any security vulnerabilities, please open an issue or contact the repository owners.

## 🌟 Acknowledgments
- Thanks to the open-source and cybersecurity communities for inspiration and tools.
- Special thanks to all contributors!

---

> Made with ❤️ by [Syed Muhammad Ibrahim](https://github.com/Ibrahim-sayys) & [Hassan Arshad](https://github.com/hassan-arshad1)