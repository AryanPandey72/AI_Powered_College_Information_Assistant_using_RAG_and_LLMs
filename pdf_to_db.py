import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from db_connector import execute_query, get_db_connection

# Load environment variables
load_dotenv()

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_timetable_pdf(pdf_path):
    print(f"Uploading {pdf_path} to OpenAI...")

    # 2. UPLOAD FILE
    try:
        uploaded_file = client.files.create(
            file=open(pdf_path, "rb"),
            purpose="user_data"
        )
        print("Upload successful! Generating SQL...")

    except Exception as e:
        print(f"\nFailed to upload file: {e}")
        return None

    # 3. PROMPT
    prompt = """
    You are a database expert. Analyze this college timetable PDF.

    There are two main parts:
    1. A schedule grid with Days, Times, and Subject Initials/Codes.
    2. A mapping table at the bottom linking those Initials/Codes to the full Subject Name, Faculty Name, and Room/Lab.

    Your task is to map the grid to the table and generate valid MySQL `INSERT INTO` statements for the `faculty_schedule` table.

    SCHEMA:
    CREATE TABLE faculty_schedule (
        id INT AUTO_INCREMENT PRIMARY KEY,
        faculty_name VARCHAR(100) NOT NULL,
        day_of_week ENUM('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday') NOT NULL,
        start_time TIME NOT NULL,
        end_time TIME NOT NULL,
        subject_name VARCHAR(100),
        class_section VARCHAR(50), 
        room_number VARCHAR(20)
    );

    RULES:
    - Write the queries strating from Monday to Saturday and don't mix them up. Also they should be in incremental order of time starting from earliest time slot to the latest time slot for each day.
    - Times MUST be in 'HH:MM:SS' format (e.g., '09:00:00', '13:30:00').
    - STRICT TIME MAPPING: Do not aggressively merge adjacent time slots. Only merge them into a single row if it is clearly a continuous Lab or Project block.
    - If a subject has a prefix like "LA/PD", ignore the "LA/PD" part. Map the base subject code to the table and assign it ONLY to that time slot and don't merge the timings with any other classes.
    - Handle merged cells or split labs (e.g., B1/B2). If a lab has two teachers, create separate rows for each teacher.
    - If a cell says 'BREAK' or 'LUNCH', then strictly look at their timings and never merge that with any other classes.
    - Clean up names (e.g., 'Mrs. Aruna TM' -> 'Mrs. Aruna T M').
    - For `class_section`, look at the top of the document (e.g., "CLASS: VI(A) SEMESTER") and format it cleanly (e.g., '6th Sem A') to use for every row.
    - Do NOT include the `id` column in your INSERT statements.
    - A1, A2, B1, B2, etc. are just section indicators for labs. They do NOT go into the subject name. They can be ignored in the subject name.
    - Return ONLY the raw SQL queries separated by semicolons. Do not include markdown blocks like ```sql or any explanations.
    - Example of an INSERT statement: INSERT INTO faculty_schedule (faculty_name, day_of_week, start_time, end_time, subject_name, class_section, room_number) VALUES ('Mrs. Kruthika C G', 'Wednesday', '09:00:00', '09:55:00', 'Professional Elective - Core Java', '5th Sem A', '185');
    - If the subject_name is Tutorial, append "Tutorial" to the subject name and append "Tutor" in the faculty_name column.
    - If the subject_name is "Technical Training" or "Aptitude Practice", use "Unknown" in the faculty name column.
    - If the subject name is like "22CSL68 A1-L3, A2-L4, A3-L10", treat it like AI,A2,A3 as batches and L1,L2,L3 as Labs i.e, the room number and 22CSL68 as the subject code. Here i have presented only an example for any kind of entry with same format treat it like that. For the faculty name assignment in this case if the faculty name in the mapping Table is like "A1-PRN+APM / A2-JS+UR / A3-TS+BPV" then assign the faculty name to the respective batch and lab. 
      For example for A1-L3 assign the faculty name as "PRN+APM" and for A2-L4 assign the faculty name as "JS+UR" and for A3-L10 assign the faculty name as "APM+JS". If the mapping table is not in that format then assign the faculty name as whatever written in the mapping table for the particular subject code.
    - If in the mapping table the faculty name is "All Faculty" for any particular class, then create separate INSERT INTO statements for each faculty member mentioned in the mapping table not including shortforms of the names.
    """

    # 4. CALL MODEL
    try:
        response = client.responses.create(
            model="gpt-5.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_file",
                            "file_id": uploaded_file.id
                        }
                    ]
                }
            ],
            temperature=0.0
        )

        print("\n[DEBUG] Response received")

        sql_queries = response.output_text.strip()

        # Clean markdown if any
        if sql_queries.startswith("```sql"):
            sql_queries = sql_queries[6:]
        if sql_queries.endswith("```"):
            sql_queries = sql_queries[:-3]

        print("\n--- GENERATED SQL ---")
        print(sql_queries)
        print("---------------------\n")

        return sql_queries

    except Exception as e:
        print(f"Error generating content: {e}")
        return None

    finally:
        # Cleanup uploaded file
        try:
            client.files.delete(uploaded_file.id)
        except:
            pass


def execute_generated_sql(sql_script):
    """Executes the batch of INSERT statements."""
    if not sql_script:
        return

    print("Executing SQL in database...")
    conn = get_db_connection()

    if conn:
        cursor = conn.cursor()
        try:
            queries = [q.strip() for q in sql_script.split(';') if q.strip()]
            for q in queries:
                cursor.execute(q)

            conn.commit()
            print(f"Successfully inserted {len(queries)} rows into the database!")

        except Exception as e:
            print(f"SQL Execution Error: {e}")
            conn.rollback()

        finally:
            cursor.close()
            conn.close()


# ==========================================
# EXECUTE SCRIPT
# ==========================================
if __name__ == "__main__":
    folder_path = "timetables"

    if os.path.exists(folder_path):
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        for i, filename in enumerate(pdf_files):
            pdf_path = os.path.join(folder_path, filename)
            print(f"\n================ Processing: {filename} ================")

            generated_sql = process_timetable_pdf(pdf_path)

            if generated_sql:
                user_confirm = input(f"Insert records from {filename}? (y/n): ")
                if user_confirm.lower() == 'y':
                    execute_generated_sql(generated_sql)
                else:
                    print("Skipped insertion.")

            # Rate limit safety
            if i < len(pdf_files) - 1:
                print("\nWaiting 10 seconds before next file...")
                time.sleep(10)

    else:
        print(f"Error: Could not find '{folder_path}' folder.")