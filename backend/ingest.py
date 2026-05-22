import os
import glob
import sys
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types 
from google.genai.errors import ClientError

from langchain_core.embeddings import Embeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

def run_multi_doc_ingestion():
    print("Step 1: Initializing configuration guardrails...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("'GEMINI_API_KEY' not found in .env file")
        sys.exit(1)
    data_dir = "data"
    output_directory = "faiss_f1_regulations_vectors"

    pdf_pattern = os.path.join(data_dir, "SECTION *.pdf")
    pdf_files = glob.glob(pdf_pattern)

    if not pdf_files:
        print("REGULATIONS NOT FOUND`/`INCORRECT NAMING CONVENTION")
        sys.exit(1)
    
    print("Valid .env values and files found.")
    all_processed_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        separators = ["\n\n", "\n", " ", ""]
    )

    print("Data extraction and tagging...")
    for file_path in sorted(pdf_files):
        file_name = os.path.basename(file_path)

        try:
            section_letter = file_name.split(" ")[1].replace(".pdf", "").upper()
        except IndexError:
            section_letter = "UNKNOWN"

        loader = PyPDFLoader(file_path)
        pages = loader.load()
        chunks = text_splitter.split_documents(pages)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["uid"] = f"{section_letter}_{idx}"
            chunk.metadata["section"] = section_letter
            chunk.metadata["document_name"] = file_name
        all_processed_chunks.extend(chunks)

    total_extracted = len(all_processed_chunks)
    print(f"Text Segmenting Complete. Total text segments created: {total_extracted}")

    print("\nGoogle Gemini Embeddings engine process beginning now")
    client = genai.Client(api_key=api_key)
    model_name = "gemini-embedding-2"

    class NativeGeminiEmbeddings(Embeddings):
        def embed_documents(self, texts):
            max_retries = 8
            base_delay = 15
            formatted_contents = [
                types.Content(parts=[types.Part.from_text(text=t)]) for t in texts
            ]
            for attempt in range(max_retries):
                try:
                    response = client.models.embed_content(
                        model=model_name,
                        contents=formatted_contents
                    )
                    return [e.values for e in response.embeddings]
                except ClientError as e:
                    if e.code == 429 and attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"\n⚠️ Rate limit hit. Backing off for {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    raise e

        def embed_query(self, text):
            response = client.models.embed_content(model=model_name, contents=text)
            if hasattr(response.embeddings[0], 'values'):
                return response.embeddings[0].values
            return response.embeddings[0]

    embeddings = NativeGeminiEmbeddings()

    existing_uids = set()
    vector_db = None

    if os.path.exists(os.path.join(output_directory, "index.faiss")):
        print(f"\nFound existing local vector database folder. Reading checkpoints...")
        try:
            vector_db = FAISS.load_local(output_directory, embeddings, allow_dangerous_deserialization=True)
            for doc_id, doc in vector_db.docstore._dict.items():
                if "uid" in doc.metadata:
                    existing_uids.add(doc.metadata["uid"])
            print(f"Loaded database successfully. {len(existing_uids)} fragments already exist in index.")
        except Exception as e:
            print(f"Could not read existing index safely ({e}). Falling back to clean build.")
            vector_db = None

    remaining_chunks = [c for c in all_processed_chunks if c.metadata["uid"] not in existing_uids]
    total_remaining = len(remaining_chunks)

    if total_remaining == 0:
        print("\nAll document fragments are already vectorized and stored. Ingestion skipped.")
        sys.exit(0)

    print(f"\nDelta check complete: Skipping {len(existing_uids)} fragments. Embedding remaining {total_remaining} targets.")
    print("\nVectorizing documents via FAISS:")
    
    BATCH_SIZE = 5
    COOL_DOWN_TIME = 10

    if vector_db is None:
        first_batch = remaining_chunks[0:BATCH_SIZE]
        print(f"Initializing fresh vector index with chunks 0 to {len(first_batch)}...")
        vector_db = FAISS.from_documents(first_batch, embeddings)
        vector_db.save_local(output_directory)
        start_idx = BATCH_SIZE
        time.sleep(COOL_DOWN_TIME)
    else:
        start_idx = 0

    for i in range(start_idx, total_remaining, BATCH_SIZE):
        batch = remaining_chunks[i : i + BATCH_SIZE]
        print(f"Processing batch: chunks {i} to {min(i + BATCH_SIZE, total_remaining)} of {total_remaining}...")
        
        vector_db.add_documents(batch)
        
        vector_db.save_local(output_directory)
        
        print(f"Cooling down for {COOL_DOWN_TIME} seconds...")
        time.sleep(COOL_DOWN_TIME)

    print(f"\nVector database successfully locked into local folder: {output_directory}")
    print("Ingestion Process Completed Successfully. F1 Knowledge Base now ready.")

if __name__ == "__main__":
    run_multi_doc_ingestion()