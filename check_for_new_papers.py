import os
import json
import subprocess
import logging
import sys

# --- Configuration ---
REPO_URL = "https://github.com/Guang000/Awesome-Dataset-Distillation.git"
CLONE_DIR = "temp_paper_repo"
PDF_FOLDER = "pdf_files"
ARTICLES_JSON_PATH = os.path.join(CLONE_DIR, "data", "articles.json")

# Setup logging to stderr so it doesn't interfere with stdout
logging.basicConfig(level=logging.INFO, format="[CheckScript] %(message)s", stream=sys.stderr)

def sync_repo():
    """Clones or pulls the latest version of the paper repo."""
    if os.path.exists(CLONE_DIR):
        logging.info(f"Pulling updates from {REPO_URL}...")
        subprocess.run(["git", "-C", CLONE_DIR, "pull"], check=True, capture_output=True, text=True)
    else:
        logging.info(f"Cloning {REPO_URL} into {CLONE_DIR}...")
        subprocess.run(["git", "clone", REPO_URL, CLONE_DIR], check=True, capture_output=True, text=True)
    logging.info("Repository sync successful.")

def find_missing_papers():
    """Finds papers in the JSON that are not in the PDF_FOLDER."""
    if not os.path.exists(ARTICLES_JSON_PATH):
        logging.error(f"Could not find {ARTICLES_JSON_PATH}.")
        return {"count": 0, "missing_keys": [], "error": "articles.json not found"}

    os.makedirs(PDF_FOLDER, exist_ok=True)
    
    try:
        existing_pdfs = set(os.listdir(PDF_FOLDER))
    except FileNotFoundError:
        existing_pdfs = set()
    
    logging.info(f"Found {len(existing_pdfs)} existing PDFs in '{PDF_FOLDER}'.")

    try:
        with open(ARTICLES_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {"count": 0, "missing_keys": [], "error": f"Failed to parse articles.json: {e}"}

    all_papers = []
    for section_list in data.get('main', {}).values():
        all_papers.extend(section_list)
    for section_list in data.get('applications', {}).values():
        all_papers.extend(section_list)

    logging.info(f"Found {len(all_papers)} total papers in articles.json.")

    missing_keys = []
    for paper in all_papers:
        cite_key = paper.get("cite")
        if not cite_key:
            continue
        
        pdf_filename = f"{cite_key}.pdf"
        if pdf_filename not in existing_pdfs:
            missing_keys.append(cite_key)

    logging.info(f"Found {len(missing_keys)} missing papers.")
    return {"count": len(missing_keys), "missing_keys": missing_keys, "error": None}

if __name__ == "__main__":
    try:
        sync_repo()
        report = find_missing_papers()
        # Print the final JSON report to stdout
        # This is what Streamlit will capture
        print(json.dumps(report))
    except Exception as e:
        # Print an error JSON to stdout
        print(json.dumps({"count": 0, "missing_keys": [], "error": str(e)}))