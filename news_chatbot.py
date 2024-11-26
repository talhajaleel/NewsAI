import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import os
import openai
from decouple import config

# OpenAI API key setup
openai.api_key = config("OPENAI_API_KEY")

# File to save embeddings
EMBEDDINGS_FILE = "embeddings.json"



def scrape_google_news(query, max_articles=7):
    """Fetch article URLs from Google News for a given query."""
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=nws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract URLs from Google News results
    articles = soup.find_all("a", href=True)
    article_urls = []
    for link in articles:
        href = link["href"]
        if "/url?q=" in href:
            url = href.split("/url?q=")[1].split("&")[0]
            article_urls.append(url)
            if len(article_urls) >= max_articles:
                break
    return article_urls

def scrape_article_content(url):
    """Scrape and return the main text content from a given article URL."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs)
    except Exception as e:
        st.error(f"Failed to scrape {url}: {e}")
        return None

def generate_embedding(text):
    """Generate embeddings using OpenAI's text-embedding-ada-002 model."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def generate_embedding(text):
    """Generate embeddings using OpenAI's text-embedding-ada-002 model."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


def save_embeddings(data):
    """Save embeddings to a JSON file."""
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f)

def load_embeddings():
    """Load embeddings from the JSON file."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_relevant_articles(user_query):
    """Find the most relevant articles based on user query."""
    embeddings = load_embeddings()
    if not embeddings:
        st.error("No articles have been processed yet. Please scrape data first.")
        return []

    query_embedding = generate_embedding(user_query)
    scores = [
        (url, cosine_similarity(query_embedding, np.array(data["embedding"])))
        for url, data in embeddings.items()
    ]
    return sorted(scores, key=lambda x: x[1], reverse=True)

def ask_ai_bot(query, context):
    """Generate a response using OpenAI GPT-4 or GPT-3.5-turbo model."""
    prompt = f"""
You are a helpful AI assistant. Based on the following context, answer the user's question concisely:

Context: {context}

User's Question: {query}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Use "gpt-4" if you have access and need higher accuracy
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

with st.sidebar:
    st.header("Admin Panel")

    # Input topic for Google News scraping
    topic = st.text_input("Enter a topic to scrape articles:", "f1 max verstappen")
    max_articles = st.number_input("Max articles to scrape:", min_value=1, max_value=50, value=5)

    if st.button("Scrape and Process Articles"):
        st.write("Scraping articles...")
        # urls = scrape_google_news(topic, max_articles)
        urls = [
            "https://tribune.com.pk/story/2511704/verstappen-takes-fourth-f1-title",
            "https://www.formula1.com/en/latest/article/the-four-time-world-champions-verstappen-joins-in-the-all-time-list-and.DIIOvqAthyjnqsyOrzMon",
            "https://www.bbc.com/sport/formula1/articles/cj4vknj92jpo",
            "https://www.skysports.com/f1/news/12433/13259676/las-vegas-gp-max-verstappens-verdict-on-where-2024-f1-title-win-ranks-amid-year-of-red-bull-difficutlies",
            "https://edition.cnn.com/2024/11/24/sport/max-verstappen-wins-world-championship-las-vegas-spt-intl/index.html"
        ]
        print(urls)
        embeddings = {}

        for url in urls:
            st.write(f"Processing: {url}")
            content = scrape_article_content(url)
            if content:
                # Generate summary manually or use content as-is
                summary = content[:2000]  # For simplicity, use the first 2000 characters
                embedding = generate_embedding(summary)
                embeddings[url] = {"summary": summary, "embedding": embedding}

        # Save embeddings to a file
        save_embeddings(embeddings)
        st.success("Articles processed and embeddings saved!")

st.title("AI News Chatbot")

# User query input
user_query = st.text_input("Ask a question about the topic:")
if st.button("Submit Query"):
    if user_query.strip():
        relevant_articles = find_relevant_articles(user_query)

        if relevant_articles:
            # Use the most relevant article's summary
            top_article_url, _ = relevant_articles[0]
            embeddings = load_embeddings()
            context = embeddings[top_article_url]["summary"]

            # Generate a response
            response = ask_ai_bot(user_query, context)
            st.write(f"### Most Relevant Article: [Read More]({top_article_url})")
            st.write(f"**Response from AI:** {response}")
        else:
            st.error("No relevant articles found.")
