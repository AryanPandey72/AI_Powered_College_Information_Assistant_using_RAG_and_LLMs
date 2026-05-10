import os
import sys
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

from db_connector import execute_query, get_all_faculty_names

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY not found. Check your .env file.")

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Pinecone Setup
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("college-rag")

# ==========================================
# 2. LOAD NAME LISTS
# ==========================================
sql_names_list = get_all_faculty_names()
vector_names_list = []
try:
    with open("faculty_bio.json", "r") as f:
        data = json.load(f)
        vector_names_list = [item['name'] for item in data]
except:
    print("Warning: Could not load faculty_bio.json.")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def resolve_names(user_question):
    resolved = {"sql": None, "vector": None}
    
    for name in sql_names_list:
        if name.lower() in user_question.lower():
            resolved["sql"] = name
            break
            
    search_term = resolved["sql"] if resolved["sql"] else user_question
    for full_name in vector_names_list:
        name_parts = full_name.lower().replace(".", " ").split()
        if any(part in search_term.lower() for part in name_parts if len(part) > 3):
            resolved["vector"] = full_name
            break
            
    return resolved

def query_vector_db(question, filter_name=None):
    try:
        query_vector = embedding_model.embed_query(question)

        results = index.query(
            vector=query_vector,
            top_k=2,
            include_metadata=True
        )

        if not results["matches"]:
            return "No bio information found."

        texts = []
        for match in results["matches"]:
            text = match["metadata"].get("text", "")
            if filter_name and filter_name.lower() not in text.lower():
                continue
            texts.append(text)

        if texts:
            return "\n".join(texts)
        return "No bio information found."

    except Exception as e:
        return f"Vector DB Error: {e}"

# ==========================================
# 4. THE BRAINS (PROMPTS & CHAINS)
# ==========================================

# --- NEW: MEMORY REWRITER ---
rewrite_system = """
Given a chat history and the latest user question, which might reference context in the chat history (like pronouns or implied subjects), formulate a standalone question which can be understood completely without the chat history. 
Do NOT answer the question, just reformulate it if needed. If it already makes sense on its own, return it exactly as is.
"""
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("human", "History:\n{history}\n\nLatest Question: {question}")
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# --- ROUTER ---
router_system = """
Classify the question into one of three types:
1. "SQL" - For Classes, Time, Rooms, Days, or Student Projects.
2. "VECTOR" - For Bio, Research, Experience, Qualifications.
3. "BOTH" - If the user asks for BOTH (e.g. "Research AND Classes").

Reply ONLY with "SQL", "VECTOR", or "BOTH".
"""
router_prompt = ChatPromptTemplate.from_messages([("system", router_system), ("human", "{question}")])
router_chain = router_prompt | llm | StrOutputParser()

# --- SQL WRITER ---
sql_system = """
You are a SQL Expert. Convert the User Question into a MySQL query.

### DATABASE SCHEMA
1. faculty_schedule (faculty_name, day_of_week, start_time, end_time, room_number, subject_name)
2. final_year_project (mentor_name, project_title, student_names)

### CRITICAL RULES
1. **Preserve Filters:** If the user says "Monday", add `WHERE day_of_week = 'Monday'`.
2. **Time Logic (CRITICAL):** - If the user asks if someone is free "at" a specific time (e.g. "at 9:10"), DO NOT check for equality.
   - You MUST check if that time falls *between* the start and end time.
   - Syntax: `WHERE '09:10:00' BETWEEN start_time AND end_time`
3. **Current Date:** "Today" -> `DAYNAME(CURDATE())`.
4. **Ignore Bio:** If the question mentions "Research" or "Experience", IGNORE those parts.

### EXAMPLES
- Q: "Classes for Madhura today?"
  A: SELECT * FROM faculty_schedule WHERE faculty_name LIKE '%Madhura%' AND day_of_week = DAYNAME(CURDATE());

- Q: "Does Madhura have any classes at 9:10 AM on Tuesday?"
  A: SELECT * FROM faculty_schedule WHERE faculty_name LIKE '%Madhura%' AND day_of_week = 'Tuesday' AND '09:10:00' BETWEEN start_time AND end_time;

Return ONLY the raw SQL query.
"""
sql_prompt = ChatPromptTemplate.from_messages([("system", sql_system), ("human", "{question}")])
sql_chain = sql_prompt | llm | StrOutputParser()

# --- FINAL ANSWER GENERATOR (UPDATED FOR STRICT CONCISENESS) ---
final_system = """
You are a concise, highly efficient college assistant. Synthesize the answer from the provided data.

### DATA SOURCES:
1. **Database Data:** Includes the SQL Query used and the Result.
2. **Bio Data:** Text about research/history.

### STRICT RULES FOR ANSWERING:
1. **NO MATH OR SECONDS:** You are strictly forbidden from calculating time differences, mentioning seconds, or explaining time math.
2. **DIRECT YES/NO:** If the user asks a Yes/No question (e.g., "Is she free at 1:40?"), your very first word MUST be "Yes" or "No."
3. **IF THEY ARE FREE:** If the SQL result is empty for a specific time check, it means they are free. Say exactly: "Yes, [Name] is free at [Time]." Do NOT list their other classes for the day unless specifically asked.
4. **IF THEY ARE BUSY:** State exactly what class they have at that time and in which room. Keep it to one concise sentence.
5. **CONCISENESS:** Do not use filler words. Do not explain how you searched the database. Be direct, accurate, and polite.
"""
final_prompt = ChatPromptTemplate.from_messages([
    ("system", final_system), 
    ("human", "Question: {question}\nData: {context}")
])
final_chain = final_prompt | llm | StrOutputParser()

# ==========================================
# 5. MAIN LOGIC
# ==========================================
def ask_college_bot(user_question, chat_history=[]):
    # 1. Handle Chat Memory (Rewrite question if necessary)
    history_str = ""
    if chat_history:
        # Get the last 4 messages to provide recent context without overloading
        recent_history = chat_history[-4:] 
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    if history_str:
        standalone_question = rewrite_chain.invoke({
            "history": history_str, 
            "question": user_question
        }).strip()
        print(f"\n[DEBUG] Rewrote '{user_question}' -> '{standalone_question}'")
    else:
        standalone_question = user_question

    # 2. Proceed with the Standalone Question
    names = resolve_names(standalone_question)
    sql_target = names["sql"]      
    vector_target = names["vector"] 
    
    try:
        strategy = router_chain.invoke({"question": standalone_question}).strip()
    except:
        strategy = "BOTH" 
    
    context_data = []

    if strategy in ["SQL", "BOTH"]:
        target = sql_target if sql_target else "the faculty"
        enhanced_q = f"{standalone_question} (Refer to {target})"
        
        query = sql_chain.invoke({"question": enhanced_q}).replace("```sql", "").replace("```", "").strip()
        
        if "SELECT" in query.upper():
            sql_result = execute_query(query)
            context_data.append(f"Database Data (Query: {query}) -> Result: {sql_result}")
        else:
             context_data.append("Database Data: No relevant SQL generated.")

    if strategy in ["VECTOR", "BOTH"]:
        bio_result = query_vector_db(standalone_question, filter_name=vector_target)
        context_data.append(f"Bio Data: {bio_result}")

    # 3. Generate Final Output
    final_response = final_chain.invoke({
        "question": standalone_question,
        "context": "\n\n".join(context_data)
    })
    
    return final_response

if __name__ == "__main__":
    print("Hi, I am your Personal AI Assistant!")
    print("Ask me about Faculty Schedules, Projects, or your teachers.")

    # Simple terminal memory for local testing
    local_history = []

    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Bot: Signing Off! Have a great day!")
                break
            
            if not user_input.strip():
                continue

            print("Bot: Thinking...", end="\r")
            
            # Pass local history to the bot
            response = ask_college_bot(user_input, chat_history=local_history)
            
            # Update local history
            local_history.append({"role": "user", "content": user_input})
            local_history.append({"role": "assistant", "content": response})
            
            sys.stdout.write("\033[K") 
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\nBot: Signing Off. Goodbye!")
            break
        except Exception as e:
            print(f"Bot: Something went wrong ({e})")
