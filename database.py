import os
import chromadb
from pypdf import PdfReader
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
DB_PATH = "./ned_db"
COLLECTION_NAME = "ned_scholarships"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600      # Size of each text piece
CHUNK_OVERLAP = 100   # Overlap to maintain context between pieces

def get_vector_collection():
    """Initializes and returns the ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

def ingest_pdfs(folder_path):
    """Phase 1: Extracts text from PDFs and stores them in the database."""
    collection = get_vector_collection()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    # List all PDF files in the target folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf") or f.endswith(".PDF")]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}.")
        return

    print(f"Starting ingestion of {len(pdf_files)} files...")

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        reader = PdfReader(path)
        
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or len(text.strip()) < 20:
                continue # Skip empty or image-only pages
            
            # Create chunks with specified overlap
            chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]
            
            for c_idx, chunk_text in enumerate(chunks):
                # Use .upsert to prevent duplicate IDs if script is re-run
                collection.upsert(
                    documents=[chunk_text],
                    metadatas=[{"source": filename, "page": page_idx + 1}],
                    ids=[f"{filename}_p{page_idx}_c{c_idx}"]
                )
        print(f"  ✓ Processed: {filename}")

    print(f"\nSuccess: Total chunks in database: {collection.count()}")

if __name__ == "__main__":
    ingest_pdfs("raw_data")