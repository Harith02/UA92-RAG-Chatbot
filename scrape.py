import requests
import bs4
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urljoin
from collections import deque
import json

# Web crawler function
def crawl_website(start_url, max_pages=110):
    urls_to_visit = deque([start_url])
    visited_urls = set()
    collected_docs = []
    bs4_strainer = bs4.SoupStrainer("body")

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.popleft()

        if current_url in visited_urls:
            continue

        print(f"Crawling: {current_url}")
        visited_urls.add(current_url)

        try:
            # Load the page content using WebBaseLoader
            loader = WebBaseLoader(
                web_paths=(current_url,),
                bs_kwargs={"parse_only": bs4_strainer}
            )
            docs = loader.load()
            collected_docs.extend(docs)

            response = requests.get(current_url, timeout=5)
            response.raise_for_status()
            soup = bs4.BeautifulSoup(response.content, "html.parser")

            for link in soup.find_all("a", href=True):
                absolute_url = urljoin(current_url, link["href"])
                if (absolute_url.startswith(start_url) and 
                    absolute_url not in visited_urls and 
                    absolute_url not in urls_to_visit):
                    urls_to_visit.append(absolute_url)

        except Exception as e:
            print(f"Error crawling {current_url}: {e}")
            continue

    return collected_docs

def save_docs_to_file(docs, filename="crawled_docs.json"):
    # Convert documents to a serializable format
    docs_serializable = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(docs_serializable, f, ensure_ascii=False, indent=2)
    print(f"💾 Crawled documents saved to: {filename}")

def main():
    start_url = "https://ua92.ac.uk/"
    print("🚀 Starting web crawl...")
    all_docs = crawl_website(start_url, max_pages=110)
    print(f"✅ Crawled {len(all_docs)} pages.")
    save_docs_to_file(all_docs)

if __name__ == "__main__":
    main()