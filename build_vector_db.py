import json
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

# ==========================================
# 1. LOAD ENV VARIABLES
# ==========================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# ==========================================
# 2. INIT PINECONE
# ==========================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("college-rag")

# ==========================================
# 3. EMBEDDING MODEL
# ==========================================
# MUST match the one used in rag_agent.py
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# 4. LOAD JSON DATA
# ==========================================
try:
    with open("faculty_bio.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: 'faculty_bio.json' not found.")
    exit()

print(f"Processing {len(data)} faculty profiles...")

# ==========================================
# 5. PROCESS + UPSERT (BATCHED)
# ==========================================
batch = []
batch_size = 50  # improves performance

for i, item in enumerate(data):
    name = item.get('name', f"id_{i}")

    rich_text = (
        f"Faculty Name: {name}. "
        f"Date of Joining NMIT: {item.get('date_of_joining', 'Not Specified')}. "
        f"Education Qualification: {item.get('education', 'Not Specified')}. "
        f"Experience: {item.get('experience', 'Not Specified')}. "
        f"Areas of Interest: {item.get('areas_of_interest', 'Not Specified')}. "
        f"Research Work: {item.get('research_work', 'Not Specified')}."
    )

    # Create embedding
    vector = embeddings.embed_query(rich_text)

    # Add to batch
    batch.append(
        (
            str(name),  # id must be string
            vector,
            {
                "text": rich_text,
                "name": name
            }
        )
    )

    # Upload batch
    if len(batch) == batch_size:
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i // batch_size + 1}")
        batch = []

# Upload remaining
if batch:
    index.upsert(vectors=batch)
    print("Uploaded final batch")

print("✅ Data successfully uploaded to Pinecone!")