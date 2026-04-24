import os
import sys
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from db_connector import execute_query, get_all_faculty_names

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Connect to Local FAISS Vector Database
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
try:
    # allow_dangerous_deserialization is safe here since we generated the local index
    vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Warning: Could not load 'faiss_index'. ({e})")
    vector_db = None

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
    if not vector_db:
        return "No bio information found (Database missing)."
        
    filter_dict = {"name": filter_name} if filter_name else None
    docs = vector_db.similarity_search(question, k=2, filter=filter_dict)
    
    if docs:
        return "\n".join([doc.page_content for doc in docs])
    return "No bio information found."

# ==========================================
# 4. THE BRAINS (PROMPTS)
# ==========================================

router_system = """
Classify the question into one of three types:
1. "SQL" - For Classes, Time, Rooms, Days, or Student Projects.
2. "VECTOR" - For Bio, Research, Experience, Qualifications.
3. "BOTH" - If the user asks for BOTH (e.g. "Research AND Classes").

Reply ONLY with "SQL", "VECTOR", or "BOTH".
"""
router_prompt = ChatPromptTemplate.from_messages([("system", router_system), ("human", "{question}")])
router_chain = router_prompt | llm | StrOutputParser()

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

final_system = """
You are a helpful assistant. Synthesize the answer from the provided data.

### DATA SOURCES:
1. **Database Data:** Includes the SQL Query used and the Result.
2. **Bio Data:** Text about research/history.

### RULES:
1. Combine facts naturally.
2. **Confidence:** If the SQL Query returned a row for a specific time check (like 9:10 AM), confirm clearly that the faculty has a class at that time.
3. **Accuracy:** Ensure the bio details match the person asked about.
In the answer don't talk about the query process; just provide the final information.
"""
final_prompt = ChatPromptTemplate.from_messages([
    ("system", final_system), 
    ("human", "Question: {question}\nData: {context}")
])
final_chain = final_prompt | llm | StrOutputParser()

# ==========================================
# 5. MAIN LOGIC
# ==========================================
def ask_college_bot(user_question):
    names = resolve_names(user_question)
    sql_target = names["sql"]      
    vector_target = names["vector"] 
    
    try:
        strategy = router_chain.invoke({"question": user_question}).strip()
    except:
        strategy = "BOTH" 
    
    context_data = []

    if strategy in ["SQL", "BOTH"]:
        target = sql_target if sql_target else "the faculty"
        enhanced_q = f"{user_question} (Refer to {target})"
        
        query = sql_chain.invoke({"question": enhanced_q}).replace("```sql", "").replace("```", "").strip()
        
        if "SELECT" in query.upper():
            sql_result = execute_query(query)
            context_data.append(f"Database Data (Query: {query}) -> Result: {sql_result}")
        else:
             context_data.append("Database Data: No relevant SQL generated.")

    if strategy in ["VECTOR", "BOTH"]:
        bio_result = query_vector_db(user_question, filter_name=vector_target)
        context_data.append(f"Bio Data: {bio_result}")

    final_response = final_chain.invoke({
        "question": user_question,
        "context": "\n\n".join(context_data)
    })
    
    return final_response

if __name__ == "__main__":
    print("Hi, I am your Personal AI Assistant!")
    print("Ask me about Faculty Schedules, Projects, or your teachers.")

    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Bot: Signing Off! Have a great day!")
                break
            
            if not user_input.strip():
                continue

            print("Bot: Thinking...", end="\r")
            response = ask_college_bot(user_input)
            
            sys.stdout.write("\033[K") 
            print(f"Bot: {response}\n")
            print("Type 'exit' or 'quit' to stop.")
            
        except KeyboardInterrupt:
            print("\nBot: Signing Off. Goodbye!")
            break
        except Exception as e:
            print(f"Bot: Something went wrong ({e})")
