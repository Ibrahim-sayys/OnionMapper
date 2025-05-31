import requests
import re
import csv

def extract_full_urls(text):
    # This regex matches only links starting with http:// or https:// followed by at least one slash after the domain
    url_pattern = re.compile(
        r'https?://[a-zA-Z0-9\-.]+(?:\.[a-zA-Z]{2,}|\.onion)(?:/[^\s"\'<>()]*)?',
        re.IGNORECASE
    )
    urls = url_pattern.findall(text)
    # Remove duplicates and strip unwanted trailing punctuation
    urls = list(set(url.strip('.,;:!?)]}') for url in urls if url))
    return urls

def main():
    url = "https://raw.githubusercontent.com/joshhighet/ransomwatch/main/groups.json"
    output_csv = "extracted_links.csv"
    resp = requests.get(url)
    resp.raise_for_status()
    text = resp.text

    urls = extract_full_urls(text)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["link"])
        for url in urls:
            writer.writerow([url])
    print(f"Extracted {len(urls)} full URLs and saved to {output_csv}")

if __name__ == "__main__":
    main()