import csv
import requests
import re
from collections import Counter
from bs4 import BeautifulSoup
import os
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded (only needs to be run once)
# nltk.download('stopwords')

# Define proxy settings for Tor
PROXIES = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050',
}


# Function to extract text from HTML content using BeautifulSoup
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements to avoid irrelevant content
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Extract text from the entire page
    full_text = soup.get_text(separator=' ')

    # Clean up the text by removing extra whitespace
    clean_text = ' '.join(full_text.split())
    return clean_text


# Function to extract keywords from the text
def extract_keywords(text, num_keywords=10):
    stop_words = set(stopwords.words('english'))  # Get predefined English stopwords
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # Match words with 4 or more letters
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(num_keywords)]  # Return only the keywords


# Function to process URLs from input CSV and write results to output CSV
def process_urls(input_csv, output_csv):
    # Ensure the input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found! Please provide a valid input file.")
        return

    with open(input_csv, 'r', encoding='utf-8-sig') as infile, open(output_csv, 'w', encoding='utf-8',
                                                                    newline='') as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)

        # Write header row to the output CSV
        csv_writer.writerow(['URL', 'Keywords'])

        # Process each URL in the input CSV
        for index, row in enumerate(csv_reader, start=1):  # Add a counter with enumerate
            url = row[0].strip()
            if not url:  # Skip empty rows
                continue

            print(f"Processing link #{index}: {url}")  # Output the link number

            try:
                # Perform GET request with Tor proxy
                response = requests.get(url, proxies=PROXIES, timeout=30)
                if response.status_code == 200:
                    # Extract and process content
                    clean_text = extract_text_from_html(response.text)
                    keywords = extract_keywords(clean_text, num_keywords=10)

                    # Join keywords into a comma-separated string
                    keywords_str = ', '.join(keywords)
                    csv_writer.writerow([url, keywords_str])
                else:
                    print(f"Failed to fetch {url}: HTTP {response.status_code}")
                    # Write fallback keywords for HTTP errors
                    fallback_keywords_str = 'down, offline, not reachable'
                    csv_writer.writerow([url, fallback_keywords_str])
            except requests.exceptions.RequestException as e:
                # Write fallback keywords for connection errors
                fallback_keywords_str = 'down, offline, not reachable'
                csv_writer.writerow([url, fallback_keywords_str])


if __name__ == "__main__":
    # Define input and output CSV file paths
    input_csv = "input_url.csv"  # Ensure this file exists
    output_csv = "output_keywords.csv"

    # Process URLs and extract keywords
    process_urls(input_csv, output_csv)
    print(f"Keyword extraction completed. Results saved to {output_csv}.")