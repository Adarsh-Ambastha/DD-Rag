import streamlit as st
import subprocess
import os
import sys
import time # For RAG delays
import glob # For finding PDFs
import psutil # For Stop Button
import json # For JSON HANDLING

# --- RAG Imports ---
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- Constants ---
PDF_FOLDER = "pdf_files"
FAISS_INDEX_PATH = "faiss_index"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_CHAT_MODEL = "gemini-2.5-flash-preview-09-2025"


# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Chatbot with Research Papers",
    page_icon="💬",
    layout="wide"
)

st.title("💬 RAG Chatbot")
st.write("Chat with your research papers.")

# --- Session State for RAG Updater ---
if 'update_stage' not in st.session_state:
    # 'idle', 'checking', 'confirm', 'downloading'
    st.session_state.update_stage = 'idle'
if 'new_paper_info' not in st.session_state:
    # Will store {'count': N, 'missing_keys': [...]}
    st.session_state.new_paper_info = None
if 'update_process_pid' not in st.session_state:
    st.session_state.update_process_pid = None
if 'update_log_file' not in st.session_state:
    st.session_state.update_log_file = None
# --------------------------------------------------


# --- Helper function to clean error messages ---
def clean_error_message(e):
    """Encodes and decodes an exception string to replace surrogate characters."""
    return str(e).encode('utf-8', errors='replace').decode('utf-8')


# --- RAG Setup Functions ---
@st.cache_resource
def load_or_create_vector_store(pdf_folder_path, index_path, api_key):
    """Loads FAISS index if it exists, otherwise builds it from PDFs."""
    if not api_key:
        st.error("Gemini API Key is missing. Cannot build or load vector store.")
        return None
    try:
        model = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, google_api_key=api_key,
                                     temperature=0.3, convert_system_message_to_human=True)
        model.invoke("test")
        st.info("Gemini Chat API key validated.")
    except Exception as e:
        clean_e = clean_error_message(e)
        if "API_KEY_INVALID" in clean_e:
            st.error("The provided Gemini API Key is invalid. Please check your key.")
        else:
            st.error(f"Error validating Chat API key (may be invalid or disabled): {clean_e}")
        return None

    if os.path.exists(index_path):
        try:
            st.info("Loading existing FAISS index...")
            embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=api_key)
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            st.success("FAISS index loaded successfully.")
            return vector_store
        except Exception as e:
            clean_e = clean_error_message(e)
            st.error(f"Error loading FAISS index: {clean_e}. Rebuilding...")

    st.info(f"Building FAISS index from PDFs in '{pdf_folder_path}'...")
    # Ensure the PDF folder exists before trying to glob it
    os.makedirs(pdf_folder_path, exist_ok=True) 
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not pdf_files:
        st.error(f"No PDF files found in '{pdf_folder_path}'. Please add PDF files or run the updater.")
        return None
    
    all_docs = []
    # Add a progress bar for PDF loading
    st.info(f"Loading {len(pdf_files)} PDF(s)...")
    progress_bar = st.progress(0, text="Loading PDFs...")
    for i, pdf_path in enumerate(pdf_files):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            clean_e = clean_error_message(e)
            st.warning(f"Could not load PDF: {os.path.basename(pdf_path)}. Error: {clean_e}")
        progress_bar.progress((i + 1) / len(pdf_files), text=f"Loading {os.path.basename(pdf_path)}")
    
    progress_bar.empty() # Clear the progress bar
    if not all_docs:
        st.error("No documents could be loaded from PDFs.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(all_docs)

    # --- FIX: Clean text chunks to remove bad characters ---
    st.info(f"Cleaning {len(doc_splits)} document chunks...")
    cleaned_splits = []
    for doc in doc_splits:
        try:
            # Replace surrogate characters and other encoding errors with '?'
            clean_content = doc.page_content.encode('utf-8', errors='replace').decode('utf-8')
            doc.page_content = clean_content
            cleaned_splits.append(doc)
        except Exception as clean_e:
            # Clean the error *from cleaning* just in case
            safe_e_msg = clean_error_message(clean_e)
            st.warning(f"Could not clean a text chunk. Skipping. Error: {safe_e_msg}")
    # --- End of fix ---
    
    if not cleaned_splits:
        st.error("No document chunks available after cleaning. Cannot build index.")
        return None

    st.info(f"Creating embeddings using '{GEMINI_EMBEDDING_MODEL}'...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=api_key)
        # Use the cleaned list
        vector_store = FAISS.from_documents(cleaned_splits, embeddings) 
        st.info("Saving FAISS index...")
        vector_store.save_local(index_path)
        st.success("FAISS index built and saved successfully.")
        return vector_store
    except Exception as e:
        clean_e = clean_error_message(e)
        if "API_KEY_INVALID" in clean_e:
             st.error("The provided Gemini API Key is invalid (failed during embedding). Please check your key.")
        else:
            st.error(f"Error creating embeddings or FAISS index: {clean_e}")
        return None

def get_gemini_response(user_query, vector_store, api_key):
    """Performs RAG to get an answer from Gemini."""
    if vector_store is None:
        st.error("Vector store is not available. Please check PDF folder and API Key.")
        return "Vector store is not available. Please check PDF folder and API Key."
    if not api_key:
        st.error("Gemini API Key is missing. Cannot process query.")
        return "Gemini API Key is missing. Cannot process query."
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(user_query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt_template = f"""Context from relevant documents:
{context}
---
Based on the context above and your general knowledge, answer the following question.
You are an expert assistant specializing in Data Distillation and Machine Learning. Your answer should be clear, concise, and helpful.
- If the question can be answered from the context, please synthesize the answer from it.
- If the question is a coding request (e.g., "write a python script..."), provide the code and explain it, using your general knowledge.
- If the question is a general reasoning question that builds on the papers (e.g., "how would method X compare to Y?"), use the context as a reference and your own knowledge to form a complete answer.
- If the question is unrelated to the context, answer it using your general knowledge and state that the information was not in the provided documents.
Question: {user_query}
Answer:"""
        model = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, google_api_key=api_key,
                                     temperature=0.3, convert_system_message_to_human=True)
        max_retries = 5
        delay = 1
        for attempt in range(max_retries):
            try:
                response = model.invoke(prompt_template)
                return response.content
            except Exception as e:
                clean_e = clean_error_message(e)
                if "429" in clean_e and attempt < max_retries - 1:
                    st.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                elif "API_KEY_INVALID" in clean_e:
                    st.error("The provided Gemini API Key is invalid. Please check your key.")
                    return "Sorry, your API key is invalid."
                else:
                    st.error(f"Error calling Gemini API: {clean_e}")
                    return "Sorry, I encountered an error trying to generate a response."
        return "Sorry, I could not get a response after multiple retries."
    except Exception as e:
        clean_e = clean_error_message(e)
        st.error(f"An error occurred during RAG: {clean_e}")
        return "Sorry, an error occurred while processing your request."
# --- End of RAG Functions ---


# =============================================================================
# --- RAG Chatbot Code ---
# =============================================================================

st.header("💬 Chat with Data Distillation Papers")

# --- NEW: Manual Update Button (Multi-Step w/ Live Logs) ---
st.sidebar.divider()
st.sidebar.subheader("RAG Settings")

# --- This is the main UI logic for the multi-step update ---

# Stage 1: IDLE
# Show "Check for New Papers" button
if st.session_state.update_stage == 'idle':
    if st.sidebar.button("🔄 Check for New Papers"):
        st.session_state.update_stage = 'checking'
        st.rerun()

# Stage 2: CHECKING
# Run the 'check_for_new_papers.py' script and stream its logs
elif st.session_state.update_stage == 'checking':
    st.subheader("Checking for New Papers...")
    log_placeholder = st.empty()
    
    # Stop button for the "check" phase
    if st.button("⏹️ Cancel Check", disabled=(st.session_state.update_process_pid is None)):
        try:
            parent = psutil.Process(st.session_state.update_process_pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            st.warning("Check terminated by user.")
        except psutil.NoSuchProcess:
            st.info("Check process already finished.")
        
        st.session_state.update_stage = 'idle'
        st.session_state.update_process_pid = None
        st.session_state.update_log_file = None
        st.rerun()
    
    # --- Process starting logic (runs once) ---
    if st.session_state.update_process_pid is None:
        check_script_path = os.path.join(os.path.dirname(__file__), "check_for_new_papers.py")
        log_file_path = "paper_check.log" # Log to a file
        st.session_state.update_log_file = log_file_path
        
        if not os.path.exists(check_script_path):
            st.error(f"Error: 'check_for_new_papers.py' not found.")
            st.session_state.update_stage = 'idle'
        else:
            try:
                # FIX: Ensure log file is opened with utf-8
                log_file = open(log_file_path, 'w', encoding='utf-8')
                process = subprocess.Popen(
                    [sys.executable, "-u", check_script_path],
                    stdout=log_file, # Script prints logs AND final JSON to stdout
                    stderr=log_file, # Also capture errors
                    text=True,
                    encoding='utf-8', # FIX: Specify encoding for the process
                    cwd=os.getcwd()
                )
                st.session_state.update_process_pid = process.pid
                st.rerun() # Re-run to start monitoring
            except Exception as e:
                clean_e = clean_error_message(e)
                st.error(f"Failed to start check process: {clean_e}")
                st.session_state.update_stage = 'idle'

    # --- Process monitoring logic (runs in a loop) ---
    else:
        log_path = st.session_state.update_log_file
        log_content = ""
        try:
            if log_path and os.path.exists(log_path):
                # FIX: Read log file with utf-8
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
            log_placeholder.code(log_content, language="log")

            proc = psutil.Process(st.session_state.update_process_pid)
            
            # If still running, refresh
            if proc.status() == psutil.STATUS_RUNNING:
                st.warning("Checking repo... (Log updates automatically)")
                time.sleep(1) # Refresh rate
                st.rerun()
            
            # If finished, parse the log file
            else:
                st.info("Check complete. Parsing results...")
                # Read the log file one last time to get final output
                if log_path and os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                log_placeholder.code(log_content, language="log")

                # The check script prints JSON as its last line
                last_line = log_content.strip().splitlines()[-1]
                try:
                    report = json.loads(last_line)
                    if report.get("error"):
                        st.error(f"Error checking for papers: {report['error']}")
                        st.session_state.update_stage = 'idle'
                    elif report.get("count", 0) > 0:
                        st.session_state.new_paper_info = report
                        st.session_state.update_stage = 'confirm'
                    else:
                        st.success("Your paper database is already up to date!")
                        st.session_state.update_stage = 'idle'
                except json.JSONDecodeError:
                    st.error("Check script failed. Could not parse result. Log is shown above.")
                    st.session_state.update_stage = 'idle'
                
                # Clean up and rerun to show next stage
                st.session_state.update_process_pid = None
                st.session_state.update_log_file = None
                st.rerun()
        
        except psutil.NoSuchProcess:
            # Same logic as "if finished" block
            st.info("Check complete. Parsing results...")
            if log_path and os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
            log_placeholder.code(log_content, language="log")
            
            last_line = log_content.strip().splitlines()[-1]
            try:
                report = json.loads(last_line)
                if report.get("error"):
                    st.error(f"Error checking for papers: {report['error']}")
                    st.session_state.update_stage = 'idle'
                elif report.get("count", 0) > 0:
                    st.session_state.new_paper_info = report
                    st.session_state.update_stage = 'confirm'
                else:
                    st.success("Your paper database is already up to date!")
                    st.session_state.update_stage = 'idle'
            except json.JSONDecodeError:
                st.error("Check script failed. Could not parse result. Log is shown above.")
                st.session_state.update_stage = 'idle'

            st.session_state.update_process_pid = None
            st.session_state.update_log_file = None
            st.rerun()
        
        except Exception as e:
            clean_e = clean_error_message(e)
            st.error(f"An error occurred while monitoring the check process: {clean_e}")
            st.session_state.update_stage = 'idle'
            st.session_state.update_process_pid = None
            st.session_state.update_log_file = None

# Stage 3: CONFIRM
# Show the results from the check and ask for confirmation
elif st.session_state.update_stage == 'confirm':
    info = st.session_state.new_paper_info
    count = info['count']
    
    st.info(f"Found {count} new paper(s). Do you want to download them?")
    col1, col2 = st.columns(2)
    if col1.button("Yes, Download Now"):
        st.session_state.update_stage = 'downloading'
        st.rerun()
    if col2.button("Cancel"):
        st.session_state.update_stage = 'idle'
        st.session_state.new_paper_info = None
        st.rerun()

# Stage 4: DOWNLOADING
# Run the 'download_papers.py' script and stream its logs
elif st.session_state.update_stage == 'downloading':
    st.subheader("Downloading New Papers...")
    log_placeholder = st.empty()
    
    # Stop button for the "download" phase
    if st.button("⏹️ Stop Update", disabled=(st.session_state.update_process_pid is None)):
        try:
            parent = psutil.Process(st.session_state.update_process_pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            st.warning("Update terminated by user.")
        except psutil.NoSuchProcess:
            st.info("Process already finished.")
        
        st.session_state.update_stage = 'idle'
        st.session_state.new_paper_info = None
        st.session_state.update_process_pid = None
        st.session_state.update_log_file = None
        st.rerun()
    
    # --- Process starting logic (runs once) ---
    if st.session_state.update_process_pid is None:
        info = st.session_state.new_paper_info
        if not info or 'missing_keys' not in info:
            st.error("Error: No paper list found. Aborting.")
            st.session_state.update_stage = 'idle'
        else:
            paper_list_json = json.dumps(info['missing_keys'])
            log_file_path = "paper_download.log" # Log to a file
            st.session_state.update_log_file = log_file_path
            
            download_script_path = os.path.join(os.path.dirname(__file__), "download_papers.py")

            if not os.path.exists(download_script_path):
                st.error(f"Error: 'download_papers.py' not found.")
                st.session_state.update_stage = 'idle'
            else:
                try:
                    # FIX: Ensure log file is opened with utf-8
                    log_file = open(log_file_path, 'w', encoding='utf-8')
                    process = subprocess.Popen(
                        [sys.executable, "-u", download_script_path, paper_list_json],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8', # FIX: Specify encoding for the process
                        cwd=os.getcwd()
                    )
                    st.session_state.update_process_pid = process.pid
                    st.rerun()
                except Exception as e:
                    clean_e = clean_error_message(e)
                    st.error(f"Failed to start download process: {clean_e}")
                    st.session_state.update_stage = 'idle'

    # --- Process monitoring logic (runs in a loop) ---
    else:
        log_path = st.session_state.update_log_file
        log_content = ""
        try:
            if log_path and os.path.exists(log_path):
                # FIX: Read log file with utf-8
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
            log_placeholder.code(log_content, language="log")

            proc = psutil.Process(st.session_state.update_process_pid)
            if proc.status() == psutil.STATUS_RUNNING:
                st.warning("Downloading papers... (Log updates automatically)")
                time.sleep(2)
                st.rerun()
            else:
                st.success("✅ Download complete! Reloading RAG database...")
                st.cache_resource.clear()
                st.session_state.update_stage = 'idle'
                st.session_state.new_paper_info = None
                st.session_state.update_process_pid = None
                st.session_state.update_log_file = None
                time.sleep(2)
                st.rerun()
        
        except psutil.NoSuchProcess:
            st.success("✅ Download complete! Reloading RAG database...")
            if log_path and os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
            log_placeholder.code(log_content, language="log")
            
            st.cache_resource.clear()
            st.session_state.update_stage = 'idle'
            st.session_state.new_paper_info = None
            st.session_state.update_process_pid = None
            st.session_state.update_log_file = None
            time.sleep(2)
            st.rerun()
        
        except Exception as e:
            clean_e = clean_error_message(e)
            st.error(f"An error occurred while monitoring the download process: {clean_e}")
            st.session_state.update_stage = 'idle'
            st.session_state.new_paper_info = None
            st.session_state.update_process_pid = None
            st.session_state.update_log_file = None

# --- END: Multi-Step Update UI ---


# --- Main Chatbot UI (only runs if not updating) ---
if st.session_state.update_stage == 'idle':
    st.write("Ask questions about the research papers provided.")

    # --- UPDATED: Get API Key from Streamlit Secrets ---
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        st.error("GEMINI_API_KEY secret not found.")
        st.warning("Please add your Gemini API Key to your Streamlit app's secrets.")
        st.info("For more info, see: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")
        st.stop()

    if not api_key:
        st.error("Your GEMINI_API_KEY secret is empty.")
        st.stop()
    
    # Store the key in session state just to be consistent, though passing it is fine
    if 'GEMINI_API_KEY' not in st.session_state:
        st.session_state.GEMINI_API_KEY = api_key
    # --- End of UPDATED API Key Logic ---

    
    vector_store = load_or_create_vector_store(
        PDF_FOLDER, 
        FAISS_INDEX_PATH, 
        st.session_state.GEMINI_API_KEY  # Use the key from session state
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the papers..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response = get_gemini_response(
                prompt, 
                vector_store, 
                st.session_state.GEMINI_API_KEY  # Use the key from session state
            )
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})