import os
import json
import subprocess
import requests
import time
import logging
import sys
import shutil
from bs4 import BeautifulSoup

# --- Configuration ---
REPO_URL = "https://github.com/Guang000/Awesome-Dataset-Distillation.git"
CLONE_DIR = "temp_paper_repo"
PDF_FOLDER = "pdf_files"
FAISS_INDEX_PATH = "faiss_index"
ARTICLES_JSON_PATH = os.path.join(CLONE_DIR, "data", "articles.json")

# --- Headers to mimic a browser ---
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Setup logging to print to stdout, which Streamlit will capture
logging.basicConfig(level=logging.INFO, format="[DownloadScript] %(asctime)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

def sync_repo():
    """Clones or pulls the latest version of the paper repo."""
    if os.path.exists(CLONE_DIR):
        logging.info(f"Pulling updates from {REPO_URL}...")
        subprocess.run(["git", "-C", CLONE_DIR, "pull"], check=True, capture_output=True, text=True)
    else:
        logging.info(f"Cloning {REPO_URL} into {CLONE_DIR}...")
        subprocess.run(["git", "clone", REPO_URL, CLONE_DIR], check=True, capture_output=True, text=True)
    logging.info("Repository sync successful.")

def get_pdf_url(paper_info):
    """Attempts to find a direct PDF download link from the paper's URL."""
    url = paper_info.get("url")
    if not url:
        return None
    try:
        if "arxiv.org/abs" in url:
            pdf_url = url.replace("/abs/", "/pdf/")
            if not pdf_url.endswith(".pdf"):
                pdf_url += ".pdf"
            return pdf_url
        if "openaccess.thecvf.com" in url:
            response = requests.get(url, headers=HTTP_HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_link = soup.find('a', href=lambda href: href and 'paper.pdf' in href)
            if pdf_link:
                return requests.compat.urljoin(url, pdf_link['href'])
        if "aclanthology.org" in url:
            if url.endswith('/'):
                return url[:-1] + ".pdf"
            return url + ".pdf"
        if "ieeexplore.ieee.org" in url or "dl.acm.org" in url:
            logging.warning(f"Skipping paywalled link: {url}")
            return None
        if url.endswith(".pdf"):
            return url
    except requests.RequestException as e:
        logging.warning(f"Could not check URL {url}: {e}")
        return None
    except Exception as e:
        logging.warning(f"Error parsing URL {url}: {e}")
        return None
    logging.warning(f"Could not determine PDF link for: {url}")
    return None

def download_specific_papers(keys_to_download):
    """Downloads only the papers specified in the list of keys."""
    if not os.path.exists(ARTICLES_JSON_PATH):
        logging.error(f"Could not find {ARTICLES_JSON_PATH}. Run the 'check' script first.")
        return False

    os.makedirs(PDF_FOLDER, exist_ok=True)
    
    try:
        with open(ARTICLES_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to parse articles.json: {e}")
        return False

    # Create a quick-lookup dictionary of all papers
    all_papers_dict = {}
    for section_list in data.get('main', {}).values():
        for paper in section_list:
            if paper.get("cite"):
                all_papers_dict[paper["cite"]] = paper
    for section_list in data.get('applications', {}).values():
        for paper in section_list:
            if paper.get("cite"):
                all_papers_dict[paper["cite"]] = paper

    logging.info(f"Starting download for {len(keys_to_download)} papers...")
    new_files_added = 0

    for i, cite_key in enumerate(keys_to_download):
        paper = all_papers_dict.get(cite_key)
        pdf_filename = f"{cite_key}.pdf"
        
        if not paper:
            logging.warning(f"({i+1}/{len(keys_to_download)}) Could not find paper with key '{cite_key}' in JSON. Skipping.")
            continue

        logging.info(f"({i+1}/{len(keys_to_download)}) Downloading: {paper.get('title')}")
        download_url = get_pdf_url(paper)
        
        if not download_url:
            logging.warning(f" -> Failed. Could not find a PDF URL.")
            continue

        try:
            pdf_response = requests.get(download_url, headers=HTTP_HEADERS, timeout=30)
            pdf_response.raise_for_status() 

            content_type = pdf_response.headers.get('Content-Type', '')
            if 'application/pdf' in content_type:
                save_path = os.path.join(PDF_FOLDER, pdf_filename)
                with open(save_path, 'wb') as f:
                    f.write(pdf_response.content)
                logging.info(f" -> Successfully saved to {save_path}")
                new_files_added += 1
            else:
                logging.warning(f" -> Failed. Link was not a PDF (Content-Type: {content_type}): {download_url}")
            
            time.sleep(1) # Be polite

        except requests.RequestException as e:
            logging.warning(f" -> Download failed for {download_url}: {e}")
        except Exception as e:
            logging.error(f" -> An unexpected error occurred: {e}")

    return new_files_added > 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("No paper list provided. Exiting.")
        sys.exit(1)

    try:
        paper_keys = json.loads(sys.argv[1])
        if not isinstance(paper_keys, list):
            raise ValueError("Input is not a JSON list.")
    except Exception as e:
        logging.error(f"Could not parse input argument: {e}")
        sys.exit(1)
    
    try:
        logging.info("--- Starting Paper Download Script ---")
        # Sync repo to ensure paper URLs are up to date
        sync_repo() 
        new_files_were_added = download_specific_papers(paper_keys)

        if new_files_were_added:
            logging.info("New papers were downloaded.")
            if os.path.exists(FAISS_INDEX_PATH):
                logging.info(f"Deleting old FAISS index at '{FAISS_INDEX_PATH}' to force rebuild...")
                try:
                    shutil.rmtree(FAISS_INDEX_PATH)
                    logging.info("FAISS index deleted successfully.")
                except OSError as e:
                    logging.error(f"Error deleting FAISS index: {e}")
        else:
            logging.warning("No new files were successfully downloaded.")

        logging.info("--- Paper Download Script Finished ---")
        
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)