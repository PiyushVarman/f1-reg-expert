import os
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI(title="RegExpert F1 Core API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is completely missing from .env file.")

client = genai.Client(api_key=api_key)
vector_db = None

class NativeGeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        formatted_contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
        response = client.models.embed_content(model="gemini-embedding-2", contents=formatted_contents)
        return [e.values for e in response.embeddings]

    def embed_query(self, text):
        response = client.models.embed_content(model="gemini-embedding-2", contents=text)
        if hasattr(response.embeddings[0], 'values'):
            return response.embeddings[0].values
        return response.embeddings[0]

@app.on_event("startup")
def load_vector_db():
    global vector_db
    db_path = "faiss_f1_regulations_vectors"
    if not os.path.exists(db_path):
        print(f"❌ Error: Vector path '{db_path}' missing! Did you run ingest.py?")
        return
        
    print("🏎️  Loading Formula 1 Regulations into memory...")
    vector_db = FAISS.load_local(db_path, NativeGeminiEmbeddings(), allow_dangerous_deserialization=True)
    print("✓ Vector DB successfully locked and hot in RAM!")

class ChatRequest(BaseModel):
    message: str

async def generate_f1_stream(query: str) -> AsyncGenerator[str, None]:
    """Retrieves relevant chunks and yields a real-time token stream from Gemini."""
    if vector_db is None:
        yield "System Error: Vector Database was not loaded properly on boot up."
        return

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
        "Please don't give the .pdf names of these documents, just refer to them as 'Section '"
    )
    user_prompt = f"CONTEXT REGULATIONS:\n{context_text}\n\nUSER QUESTION:\n{query}"

    try:
        response_stream = client.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )
        
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
                await asyncio.sleep(0.01)
                
    except Exception as e:
        yield f"\n[Backend Error occurred processing streaming chunk: {str(e)}]"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be blank.")
        
    return StreamingResponse(
        generate_f1_stream(request.message), 
        media_type="text/plain"
    )