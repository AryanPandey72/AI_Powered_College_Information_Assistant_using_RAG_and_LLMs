import json
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. SETUP CHROMA DB (Local Storage)
# ==========================================
# This creates/opens a folder named 'college_chroma_db' in your project directory
client = chromadb.PersistentClient(path="college_chroma_db")

# We use the standard open-source embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ==========================================
# 2. RESET COLLECTION (Clean Slate)
# ==========================================
try:
    client.delete_collection(name="faculty_profiles")
except:
    pass

collection = client.create_collection(
    name="faculty_profiles",
    embedding_function=sentence_transformer_ef
)

# ==========================================
# 3. LOAD JSON DATA
# ==========================================
try:
    with open("faculty_bio.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: 'faculty_bio.json' not found. Please create it first.")
    exit()

documents = []
metadatas = []
ids = []

print(f"Processing {len(data)} faculty profiles...")

for idx, item in enumerate(data):
    # ==========================================
    # 4. CONSTRUCT RICH TEXT (The Searchable Content)
    # ==========================================
    # We stitch all the fields into one descriptive paragraph.
    # The AI reads this paragraph to find answers.
    rich_text = (
        f"Faculty Name: {item.get('name', 'Unknown')}. "
        f"Date of Joining NMIT: {item.get('date_of_joining', 'Not Specified')}. "
        f"Education Qualification: {item.get('education', 'Not Specified')}. "
        f"Experience: {item.get('experience', 'Not Specified')}. "
        f"Areas of Interest: {item.get('areas_of_interest', 'Not Specified')}. "
        f"Research Work: {item.get('research_work', 'Not Specified')}."
    )
    
    documents.append(rich_text)
    
    # Metadata helps us filter results if needed later
    metadatas.append({"name": item.get('name', 'Unknown')})
    
    # Unique ID for each record
    ids.append(str(idx))

# ==========================================
# 5. SAVE TO VECTOR DATABASE
# ==========================================
if documents:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("Success! Faculty profiles have been indexed in 'college_chroma_db'.")
    print("You can now ask questions like 'When did Madhura mam join?' or 'Who knows Blockchain?'.")
else:
    print("Warning: No data found in JSON to index.")