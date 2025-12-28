# Here’s a complete, runnable Python example for data collection in a Machine Learning workflow.
# This script demonstrates how to collect data from three common sources:
# CSV file (local or remote)
# Web API (JSON data)
# Web scraping (HTML parsing)

# # Guardian API (also free)
# https://open-platform.theguardian.com/access/
# New York Times API (free tier available)
# https://developer.nytimes.com/apis

Pythonimport pandas as pd
import requests
from bs4 import BeautifulSoup

# -------------------------------
# 1. Collect data from a CSV file
# -------------------------------
def collect_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"[CSV] Loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"[CSV] File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[CSV] Error reading file: {e}")
        return pd.DataFrame()

# --------------------------------
# 2. Collect data from a Web API
# --------------------------------
def collect_from_api(api_url):
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Raise error for bad status
        data = response.json()
        df = pd.DataFrame(data)
        print(f"[API] Retrieved {len(df)} records from API")
        return df
    except requests.exceptions.RequestException as e:
        print(f"[API] Request failed: {e}")
        return pd.DataFrame()
    except ValueError:
        print("[API] Failed to parse JSON")
        return pd.DataFrame()

# --------------------------------
# 3. Collect data via Web Scraping
# --------------------------------
def collect_from_web(url, table_index=0):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = pd.read_html(str(soup))
        if tables:
            df = tables[table_index]
            print(f"[Web] Extracted table with {len(df)} rows from {url}")
            return df
        else:
            print("[Web] No tables found on page")
            return pd.DataFrame()
    except Exception as e:
        print(f"[Web] Scraping failed: {e}")
        return pd.DataFrame()

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example CSV (replace with your file path or URL)
    csv_data = collect_from_csv("sample_data.csv")

    # Example API (public placeholder API)
    api_data = collect_from_api("https://jsonplaceholder.typicode.com/posts")

    # Example Web Scraping (Wikipedia table)
    web_data = collect_from_web("https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)")

    # Combine all collected data (optional)
    combined_data = pd.concat([csv_data, api_data, web_data], ignore_index=True, sort=False)
    print(f"[Combined] Total collected rows: {len(combined_data)}")

    # Save combined data for ML preprocessing
    combined_data.to_csv("collected_data.csv", index=False)
    print("[Saved] Data saved to collected_data.csv")


# How This Works

# CSV Collection
# Reads local or remote CSV files into a Pandas DataFrame.
# Handles missing files and read errors.

# API Collection
# Fetches JSON data from a REST API.
# Converts JSON to a DataFrame for ML use.

Web Scraping
# Downloads HTML content and extracts tables using pandas.read_html.
# Uses BeautifulSoup for parsing.

# Combining Data
# Merges all collected datasets into one DataFrame.
# Saves to a CSV for later ML preprocessing.


# CODE SECTION STARTS HERE

# Sample fake news templates (exaggerated/sensational)
fake_news_templates = [
    "SHOCKING: {} conspiracy revealed by anonymous source!",
    "You won't believe what {} does to your {}!",
    "BREAKING: {} secretly controls {} worldwide!",
    "Doctors HATE this one weird trick for {}!",
    "URGENT: {} banned in {} countries - find out why!",
    "Miracle cure for {} discovered in {}!",
    "EXPOSED: {} hiding the truth about {}!",
    "Celebrity {} reveals secret about {} that changes everything!"
]

def generate_article(is_fake):
    """Generate a single news article"""
    if is_fake:
        title = random.choice(fake_news_templates).format(
            random.choice(subjects + topics),
            random.choice(objects + topics)
        )
    else:
        title = random.choice(real_news_templates).format(
            random.choice(places),
            random.choice(topics)
        )
    
    # Generate random date within last year
    days_ago = random.randint(1, 365)
    date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    # Generate simple content
    content = f"{title}. Additional details about this story are being gathered. " \
              f"Experts from various fields are analyzing the situation. " \
              f"More updates will follow as information becomes available."
    
    return {
        'title': title,
        'content': content,
        'date': date,
        'label': 1 if is_fake else 0  # 1 = Fake, 0 = Real
    }

def create_dataset(num_real=100, num_fake=100):
    """Create a balanced dataset of real and fake news articles"""
    articles = []
    
    # Generate real news
    for _ in range(num_real):
        articles.append(generate_article(is_fake=False))
    
    # Generate fake news
    for _ in range(num_fake):
        articles.append(generate_article(is_fake=True))
    
    # Create DataFrame
    df = pd.DataFrame(articles)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Generate dataset
print("Generating fake news detection dataset...")
df = create_dataset(num_real=150, num_fake=150)

# Display info
print(f"\nDataset created with {len(df)} articles")
print(f"Real news: {(df['label']==0).sum()}")
print(f"Fake news: {(df['label']==1).sum()}")

# Show sample
print("\nSample articles:")
print(df.head(10))

# Save to CSV
df.to_csv('fake_news_dataset.csv', index=False)
print("\nDataset saved to 'fake_news_dataset.csv'")


# FETCH REAL NEWS FROM APIs
# Code Generated by Sidekick is for learning and experimentation purposes only.
import pandas as pd
import requests
from datetime import datetime, timedelta
import random
import time

# NewsAPI configuration
NEWS_API_KEY = 'YOUR_API_KEY_HERE'  # Get free API key from https://newsapi.org/
NEWS_API_URL = 'https://newsapi.org/v2/everything'

def fetch_real_news(num_articles=100, days_back=30):
    """Fetch real news articles from NewsAPI"""
    articles = []
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Categories to fetch diverse news
    queries = ['technology', 'business', 'science', 'health', 'politics']
    
    print("Fetching real news articles...")
    
    for query in queries:
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(100, num_articles // len(queries) + 20),
            'apiKey': NEWS_API_KEY
        }
        
        try:
            response = requests.get(NEWS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                for article in data['articles']:
                    if article['title'] and article['description']:
                        articles.append({
                            'title': article['title'],
                            'content': article['description'] or article['content'] or '',
                            'date': article['publishedAt'][:10],
                            'source': article['source']['name'],
                            'url': article['url'],
                            'label': 0  # 0 = Real news
                        })
                print(f"Fetched {len(data['articles'])} articles for '{query}'")
            
            time.sleep(1)  # Respect API rate limits
            
        except Exception as e:
            print(f"Error fetching news for '{query}': {str(e)}")
    
    # Remove duplicates and limit to requested number
    df = pd.DataFrame(articles).drop_duplicates(subset=['title'])
    return df.head(num_articles)

