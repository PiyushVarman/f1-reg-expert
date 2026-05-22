import os
import sys
from dotenv import load_dotenv

from google import genai
from google.genai import types

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def initialize_qa_system():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env file.")
        sys.exit(1)
        
    output_directory = "faiss_f1_regulations_vectors"
    if not os.path.exists(output_directory):
        print(f"Vector directory '{output_directory}' not found. Run ingest.py first!")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    model_name = "gemini-embedding-2"

    class NativeGeminiEmbeddings(Embeddings):
        def embed_documents(self, texts):
            formatted_contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
            response = client.models.embed_content(model=model_name, contents=formatted_contents)
            return [e.values for e in response.embeddings]

        def embed_query(self, text):
            response = client.models.embed_content(model=model_name, contents=text)
            if hasattr(response.embeddings[0], 'values'):
                return response.embeddings[0].values
            return response.embeddings[0]

    embeddings = NativeGeminiEmbeddings()

    print("Loading local F1 Regulations vector database...")
    vector_db = FAISS.load_local(output_directory, embeddings, allow_dangerous_deserialization=True)
    return client, vector_db

def ask_f1_expert(query, client, vector_db):
    print(f"\nSearching index for: '{query}'...")
    docs = vector_db.similarity_search(query, k=4)
    
    context_segments = []
    for doc in docs:
        source_sec = doc.metadata.get('section', 'UNKNOWN')
        source_doc = doc.metadata.get('document_name', 'UNKNOWN')
        context_segments.append(f"[Source: {source_doc} | Section: {source_sec}]\n{doc.page_content}")
        
    context_text = "\n\n---\n\n".join(context_segments)

    system_instruction = (
        "You are an expert FIA Formula 1 Race Steward and technical regulations authority. "
        "Your job is to answer user queries accurately based ONLY on the provided regulation context snippets. "
        "If the answer cannot be confidently deduced from the context, state that you do not have sufficient "
        "regulatory text to answer. Always reference the Section letter or source document if available."
    )

    user_prompt = f"CONTEXT REGULATIONS:\n{context_text}\n\nUSER QUESTION:\n{query}"

    print("Generating response from Gemini...")
    response = client.models.generate_content(
        model='gemini-2.5-flash', # Fast and highly intelligent for RAG tasks
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2, # Low temperature keeps answers factual and strictly tied to context
        )
    )
    
    return response.text, docs

if __name__ == "__main__":
    client, vector_db = initialize_qa_system()
    
    print("\n--- F1 Regulations RAG System Active ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_query = input("Ask about F1 Regulations: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        if not user_query.strip():
            continue
            
        answer, sources = ask_f1_expert(user_query, client, vector_db)
        print("\n=== ANSWER ===")
        print(answer)
        print("==============\n")