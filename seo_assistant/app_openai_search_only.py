import os
import time
import requests
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from bs4 import BeautifulSoup
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler

from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


from openai import OpenAI
openai_api_key = st.secrets["openai"]["api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()

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
DATA_DIR = "seo_assistant/data"
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
    """Scrape the heading, content, category, and tags from an article URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Adjust selectors based on the website's structure
        heading_tag = soup.find('h1', class_='maintitle')  # main title selector
        heading = heading_tag.get_text(strip=True) if heading_tag else 'No Heading'
        
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        paragraph_titles = [f"## {h2.get_text(strip=True)}" for h2 in soup.find_all('h2', class_='title')]  # each paragraph title

        # Connect paragraphs with titles (with exception for the first paragraph)
        if paragraphs:
            lead_paragraph = paragraphs.pop(0)
        else:
            lead_paragraph = ''

        content = lead_paragraph + '\n' + '\n'.join([val for pair in zip(paragraph_titles, paragraphs) for val in pair])
        
        category_tags_container = soup.find('div', class_='maindataright')
        if category_tags_container:
          category_tags = category_tags_container.find_all('a', class_='category') 
          categories = [tag.get_text(strip=True) for tag in category_tags] if category_tags else ['Uncategorized']
          category = ', '.join(categories)
        else:
          category_tags = soup.find_all('a', class_='category') 
          categories = [tag.get_text(strip=True) for tag in category_tags] if category_tags else ['Uncategorized']
          category = ', '.join(categories)
        
        tag_elements = soup.find_all('meta', property='article:tag')  # Example selector
        tags = [tag['content'].strip() for tag in tag_elements] if tag_elements else []
        tag_elements = ', '.join(tags)
        
        return {
            'url': url,
            'heading': heading,
            'content': content,
            'category': category,
            'tags': tag_elements
        }
    except requests.RequestException as e:
        st.warning(f"Failed to fetch {url}: {e}")
        return {
            'url': url,
            'heading': 'Failed to fetch',
            'content': '',
            'category': 'Unknown',
            'tags': ''
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
    # model_name = 'paraphrase-xlm-r-multilingual-v1'  # Supports Polish and multiple languages
    model_name = 'radlab/polish-sts-v2'
    # model_name = 'Voicelab/sbert-base-cased-pl'
    return SentenceTransformer(model_name)

model = load_model()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks.
    :param text: The text to split.
    :param chunk_size: The size of each chunk.
    :param overlap: The number of overlapping characters between chunks.
    :return: List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    # return chunks
    return [text]

# def generate_embeddings(texts: List[str]) -> np.ndarray:
#     """Generate embeddings for a list of texts."""
#     embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
#     return embeddings.cpu().numpy()

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI's embedding model."""
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# ========================== #
#      Vector Database       #
# ========================== #

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a FAISS index from embeddings."""
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Using inner product (cosine similarity)
    # Normalize embeddings to unit length for cosine similarity
    faiss.normalize_L2(embeddings)
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

def find_similar_articles(query_embedding: np.ndarray, metadata: pd.DataFrame, index: faiss.Index, top_k: int = 5) -> pd.DataFrame:
  """Find similar articles based on query embedding and metadata using cosine similarity."""
  # Perform the search using FAISS
  D, I = index.search(query_embedding, top_k)
  indices = I[0]
  
  # Calculate cosine similarity (dot product since vectors are normalized)
  similarities = np.dot(index.reconstruct_n(0, index.ntotal), query_embedding.T).flatten()
  
  similar_articles = metadata.iloc[indices].copy()
  similar_articles['similarity'] = similarities[indices]
  
  # Optional: Filter by category or tags to improve relevance
  # For example, prioritize articles in the same category
  current_categories = metadata.iloc[indices]['category']
  similar_articles['category_match'] = similar_articles['category'] == current_categories.iloc[0]
  
  # Re-rank similar articles: first by similarity, then by category match
  similar_articles = similar_articles.sort_values(by=['category_match', 'similarity'], ascending=[False, False])
  
  return similar_articles

# ========================== #
#         Streamlit UI       #
# ========================== #

def main():
    st.title("SEO Assistant")
    st.header("Find and Suggest Internal Links")
    index = load_faiss_index(FAISS_INDEX_FILE)
    metadata = load_metadata(SCRAPED_DATA_FILE)
    chunk_metadata_file = os.path.join(DATA_DIR, "chunk_metadata.csv")
    if os.path.exists(chunk_metadata_file):
        chunk_metadata = pd.read_csv(chunk_metadata_file)
    else:
        st.error(f"Chunk metadata file {chunk_metadata_file} not found.")
        chunk_metadata = pd.DataFrame()

    if index is None or metadata.empty or chunk_metadata.empty:
        st.error("Please build the database first.")
        return

    input_text = st.text_area("Enter text to find similar articles for internal linking:", height=200)
    if st.button("Find & Suggest Links"):
        if not input_text.strip():
            st.error("Please enter some text to search.")
        else:
            with st.spinner('Generating embedding and retrieving similar articles...'):
                query_embedding = generate_embeddings([input_text]).astype('float32')
                similar_articles = find_similar_articles(query_embedding, chunk_metadata, index)
                    
                # Normalize similarity for progress bar
                similar_articles['similarity_percentage'] = similar_articles['similarity'] * 100

                st.write("### Suggested Internal Links:")
                for idx, row in similar_articles.iterrows():
                  similarity = row['similarity_percentage']
                  st.markdown(f"**[{row['heading']}]({row['url']})**")
                  st.progress(similarity / 100)
                  st.write(f"Similarity: {similarity:.2f}% | Categories Match: {'Yes' if row['category_match'] else 'No'}")
                  st.write(f"**Categories found:** {row['category']}")
                  st.write(f"**Tags found:** {row['tags']}")

    st.sidebar.markdown("""
    ---
    ### SEO Assistant
    Prepared by Aleksander Skubała.
    """)

if __name__ == '__main__':
    main()