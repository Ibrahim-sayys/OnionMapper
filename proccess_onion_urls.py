import csv
import requests
import re
from collections import Counter
from bs4 import BeautifulSoup
import os

# Define proxy settings for Tor
PROXIES = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050',
}

# Function to extract text from HTML content using BeautifulSoup
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator=' ')

# Function to extract keywords from the text
def extract_keywords(text, num_keywords=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # Match words with 4 or more letters
    word_counts = Counter(words)
    return word_counts.most_common(num_keywords)

# Function to process URLs from input CSV and write results to output CSV
def process_urls(input_csv, output_csv):
    # Handle BOM in input CSV
    with open(input_csv, 'r', encoding='utf-8-sig') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)

        # Write header row to the output CSV
        csv_writer.writerow(['URL', 'Keywords'])

        # Process each URL in the input CSV
        for row in csv_reader:
            url = row[0].strip()
            if not url:  # Skip empty rows
                continue

            print(f"Processing URL: {url}")

            try:
                # Perform GET request with Tor proxy
                response = requests.get(url, proxies=PROXIES, timeout=30)
                if response.status_code == 200:
                    # Extract and process content
                    clean_text = extract_text_from_html(response.text)
                    keywords = extract_keywords(clean_text, num_keywords=10)

                    # Format keywords as "word(count)"
                    keywords_str = ', '.join([f"{word}({count})" for word, count in keywords])
                    csv_writer.writerow([url, keywords_str])
                else:
                    print(f"Failed to fetch {url}: HTTP {response.status_code}")
                    csv_writer.writerow([url, f"Failed: HTTP {response.status_code}"])
            except requests.exceptions.RequestException as e:
                print(f"Error processing {url}: {e}")
                csv_writer.writerow([url, f"Error: {e}"])

if __name__ == "__main__":
    # Define input and output CSV file paths
    input_csv = "input_url.csv"
    output_csv = "output_keywords.csv"

    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found!")
        # Create a sample input file as fallback
        with open(input_csv, 'w', encoding='utf-8') as f:
            f.write("http://example1.onion/\nhttp://example2.onion/")  # Add sample URLs
        print(f"Sample input file created at '{input_csv}'. Please add URLs to this file and rerun the script.")
    else:
        # Process URLs and extract keywords
        process_urls(input_csv, output_csv)
        print(f"Keyword extraction completed. Results saved to {output_csv}.")