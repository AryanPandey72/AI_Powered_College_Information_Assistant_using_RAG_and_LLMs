import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables from .env file (for local testing)
load_dotenv()

def get_db_connection():
    """Establishes a connection to the Aiven MySQL database."""
    try:
        # Fetching credentials securely from environment variables
        db_host = os.getenv("DB_HOST")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")
        db_port = os.getenv("DB_PORT", "3306") # Default to 3306 if not specified

        if not db_password or not db_host:
            print("Error: Database credentials are not properly set in the environment.")
            return None

        connection = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=db_port
        )
        
        if connection.is_connected():
            return connection
            
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def execute_query(query):
    """Executes a Read-Only SQL query and returns the results as a list of dictionaries."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True) 
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as e:
            return f"SQL Error: {e}"
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    return "Connection Error"

def get_all_faculty_names():
    """Fetches all unique faculty names from both schedule and project tables."""
    conn = get_db_connection()
    names = set()
    
    if conn:
        cursor = conn.cursor()
        try:
            # 1. Get names from Schedule
            cursor.execute("SELECT DISTINCT faculty_name FROM faculty_schedule")
            for row in cursor.fetchall():
                if row[0]: # Check if not None
                    names.add(row[0])
            
            # 2. Get names from Projects
            cursor.execute("SELECT DISTINCT mentor_name FROM final_year_project")
            for row in cursor.fetchall():
                if row[0]: # Check if not None
                    names.add(row[0])
                    
        except Error as e:
            print(f"Error fetching names: {e}")
            
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
                
    return list(names)