import requests
from bs4 import BeautifulSoup
import time
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd
import streamlit as st
import os
import json
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# ========================== #
#        Web Scraping        #
# ========================== #

# Configuration
BASE_URL = "https://www.vogue.pl"  # Replace with your client's website
CATEGORY_URLS = [
    BASE_URL+"/b/moda",
    BASE_URL+"/b/uroda",
    BASE_URL+"/b/kultura",
    BASE_URL+"/b/ludzie",
]
ARTICLE_LINK_PREFIX = "/a/"  # Adjust based on URL structure
SLEEP_TIME = 1  # Seconds to wait between requests
DATA_DIR = "data"
SCRAPED_DATA_FILE = os.path.join(DATA_DIR, "articles_metadata.csv")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_article_links_selenium(category_urls: List[str], base_url: str = BASE_URL, max_scrolls: int = 10, scroll_pause_time: float = 1.0) -> List[str]:
    """Scrape article links from given category URLs using Selenium, handling infinite scrolling."""
    from selenium.webdriver.chrome.options import Options

    article_links = set()
    
    # Setup Selenium with headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Initialize WebDriver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    
    for category_url in category_urls:
        st.write(f"Processing category: {category_url}")
        try:
            driver.get(category_url)
            time.sleep(scroll_pause_time)
            
            last_height = driver.execute_script("return document.body.scrollHeight")
            scrolls = 0
            while scrolls < max_scrolls:
                # Scroll down to bottom
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
                
                # Calculate new scroll height and compare with last scroll height
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    # Reached the end
                    break
                last_height = new_height
                scrolls += 1
                st.write(f"Scrolled {scrolls}/{max_scrolls} times")
            
            # After scrolling, parse the page source
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.startswith(ARTICLE_LINK_PREFIX):
                    full_url = base_url.rstrip('/') + href
                    article_links.add(full_url)
            st.write(f"Found {len(article_links)} unique links so far.")
        except Exception as e:
            st.error(f"Error processing {category_url}: {e}")
    
    driver.quit()
    return list(article_links)

def scrape_article(url: str) -> Dict[str, str]:
    """Scrape the heading and content from an article URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Adjust selectors based on the website's structure
        heading_tag = soup.find('h1')
        heading = heading_tag.get_text(strip=True) if heading_tag else 'No Heading'
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        content = ' '.join(paragraphs)
        return {
            'url': url,
            'heading': heading,
            'content': content
        }
    except requests.RequestException as e:
        st.warning(f"Failed to fetch {url}: {e}")
        return {
            'url': url,
            'heading': 'Failed to fetch',
            'content': ''
        }

def scrape_all_articles(category_urls: List[str], base_url: str = BASE_URL) -> List[Dict[str, str]]:
    """Scrape all articles from the given categories."""
    st.info("Starting to scrape article links using Selenium...")
    links = get_article_links_selenium(category_urls, base_url)
    st.success(f"Found {len(links)} unique articles.")
    
    articles = []
    progress_bar = st.progress(0)
    total = len(links)
    for idx, link in enumerate(links):
        article = scrape_article(link)
        articles.append(article)
        progress_bar.progress((idx + 1) / total)
    progress_bar.empty()
    return articles

def save_scraped_data(articles: List[Dict[str, str]], file_path: str = SCRAPED_DATA_FILE):
    """Save scraped articles to a CSV file."""
    df = pd.DataFrame(articles)
    df.to_csv(file_path, index=False)
    st.success(f"Scraped data saved to {file_path}")
    return df

# ========================== #
#     Data Processing        #
# ========================== #

# Initialize the multilingual model
@st.cache_resource
def load_model():
    model_name = 'paraphrase-xlm-r-multilingual-v1'  # Supports Polish and multiple languages
    return SentenceTransformer(model_name)

model = load_model()

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    return embeddings.cpu().numpy()

# ========================== #
#      Vector Database       #
# ========================== #

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a FAISS index from embeddings."""
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.Index, file_path: str = FAISS_INDEX_FILE):
    """Save FAISS index to a file."""
    faiss.write_index(index, file_path)
    st.success(f"FAISS index saved to {file_path}")

def load_faiss_index(file_path: str = FAISS_INDEX_FILE) -> faiss.Index:
    """Load FAISS index from a file."""
    if not os.path.exists(file_path):
        st.error(f"FAISS index file {file_path} not found.")
        return None
    return faiss.read_index(file_path)

def load_metadata(file_path: str = SCRAPED_DATA_FILE) -> pd.DataFrame:
    """Load scraped metadata from CSV."""
    if not os.path.exists(file_path):
        st.error(f"Metadata file {file_path} not found.")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    return df

# ========================== #
#         Streamlit UI       #
# ========================== #

def main():
    st.title("SEO Helper")
    menu = ["Build Database", "Search Articles"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Build Database":
        st.header("Build Article Database")
        if st.button("Start Scraping and Building Database"):
            with st.spinner('Scraping articles...'):
                articles = scrape_all_articles(CATEGORY_URLS, BASE_URL)
                if articles:
                    df = save_scraped_data(articles, SCRAPED_DATA_FILE)
                else:
                    st.error("No articles scraped.")
                    return

            with st.spinner('Generating embeddings...'):
                texts = df['content'].fillna("").tolist()
                embeddings = generate_embeddings(texts)
                st.success("Embeddings generated.")

            with st.spinner('Creating FAISS index...'):
                index = create_faiss_index(embeddings)
                save_faiss_index(index, FAISS_INDEX_FILE)
                st.success("FAISS index created.")

            st.balloons()

    elif choice == "Search Articles":
        st.header("Find Similar Articles")
        index = load_faiss_index(FAISS_INDEX_FILE)
        metadata = load_metadata(SCRAPED_DATA_FILE)

        if index is None or metadata.empty:
            st.error("Please build the database first.")
            return

        # Only "By Input Text" search mode is retained
        input_text = st.text_area("Enter text to find similar articles:", height=200)
        if st.button("Find Similar Articles"):
            if not input_text.strip():
                st.error("Please enter some text to search.")
            else:
                with st.spinner('Generating embedding and retrieving similar articles...'):
                    query_embedding = generate_embeddings([input_text]).astype('float32')
                    D, I = index.search(query_embedding, 5)
                    
                    # Convert distances to similarity scores (optional scaling)
                    # Assuming L2 distance, smaller distance means higher similarity
                    # To visualize, we can invert and normalize them
                    distances = D[0]
                    max_distance = distances.max() if distances.max() != 0 else 1
                    similarities = [(1 - (dist / max_distance)) for dist in distances]

                    similar_articles = metadata.iloc[I[0]]
                    similar_articles = similar_articles.copy()
                    similar_articles['similarity'] = similarities

                    st.write("### Similar Articles:")
                    for idx, row in similar_articles.iterrows():
                        similarity_percentage = row['similarity'] * 100
                        st.markdown(f"**[{row['heading']}]({row['url']})**")
                        st.progress(similarity_percentage / 100)
                        st.write(f"Similarity: {similarity_percentage:.2f}%")
    
    st.sidebar.markdown("""
    ---
    ### SEO Helper
    Developed with ❤️ by Aleksander Skubała.
    """)

if __name__ == '__main__':
    main()