#database_manager.py
import sqlite3
from datetime import datetime
import hashlib
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def init_database():
    """Initialize the SQLite database with all required tables"""
    conn = sqlite3.connect('medicareai.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')

    # Chat history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,  -- 'user' or 'bot'
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        session_id TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')

    # CBC reports table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cbc_reports (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        report_name TEXT NOT NULL,
        report_data TEXT NOT NULL,  -- Store the extracted text from PDF
        analysis_result TEXT,       -- Store the AI analysis
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')

    # User sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')

    # Email logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS email_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        email_type TEXT NOT NULL,
        content TEXT NOT NULL,
        status TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    # Disease prediction table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS disease_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        prediction_type TEXT NOT NULL,
        result TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id)  -- Changed from users(id) to users(user_id)
    )
    ''')

    conn.commit()
    conn.close()

class DatabaseManager:
    def __init__(self, db_file='medicareai.db',openai_api_key=None):
        self.db_file = db_file
        self.chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)
        init_database()
        
    def hash_password(self, password: str) -> str:
        """
        Hash a password using SHA-256
        
        Args:
            password (str): The plain text password to hash
            
        Returns:
            str: The hashed password
        """
        if not isinstance(password, str):
            raise ValueError("Password must be a string")
        
        # Create SHA-256 hash of the password
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

   
    def get_connection(self):
        return sqlite3.connect(self.db_file)

    def add_user(self, username, email, password_hash):
        """Add a new user to the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
                ''', (username, email, password_hash))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_user(self, username):
        """Retrieve user information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT user_id, username, email, password_hash
            FROM users
            WHERE username = ?
            ''', (username,))
            return cursor.fetchone()

    def start_session(self, user_id):
        """Start a new chat session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO user_sessions (user_id)
            VALUES (?)
            ''', (user_id,))
            conn.commit()
            return cursor.lastrowid

    def end_session(self, session_id):
        """End a chat session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE user_sessions
            SET end_time = CURRENT_TIMESTAMP
            WHERE session_id = ?
            ''', (session_id,))
            conn.commit()

    def add_chat_message(self, user_id, session_id, role, content):
        """Add a chat message to history"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO chat_history (user_id, session_id, role, content)
            VALUES (?, ?, ?, ?)
            ''', (user_id, session_id, role, content))
            conn.commit()
            return cursor.lastrowid

    def get_chat_history(self, user_id, limit=10):
        """Get recent chat history for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT role, content, timestamp
            FROM chat_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (user_id, limit))
            return cursor.fetchall()

    def save_cbc_report(self, user_id, report_name, report_data, analysis_result):
        """Save a CBC report and its analysis"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO cbc_reports (user_id, report_name, report_data, analysis_result)
            VALUES (?, ?, ?, ?)
            ''', (user_id, report_name, report_data, analysis_result))
            conn.commit()
            return cursor.lastrowid

    def get_user_cbc_reports(self, user_id):
        """Get all CBC reports for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT report_id, report_name, analysis_result, upload_date
            FROM cbc_reports
            WHERE user_id = ?
            ORDER BY upload_date DESC
            ''', (user_id,))
            return cursor.fetchall()
        

    def get_email_logs(self, user_id, limit=10):
        """Get recent email logs for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT email_type, content, status, timestamp
            FROM email_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (user_id, limit))
            return cursor.fetchall()
    
    def save_disease_prediction(self, user_id, prediction_type, result, timestamp):
        """
        Save a disease prediction result to the database.
        
        Args:
            user_id (int): The ID of the user
            prediction_type (str): Type of prediction (Hemophilia, Anemia, etc.)
            result (str): The prediction result text
            timestamp (datetime): When the prediction was made
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO disease_predictions 
                    (user_id, prediction_type, result, timestamp) 
                    VALUES (?, ?, ?, ?)
                ''', (user_id, prediction_type, result, timestamp))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                print(f"Error saving prediction: {e}")
                return None
                       
    def get_disease_predictions(self, user_id):
        """Get all disease predictions for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT prediction_type, result, timestamp
            FROM disease_predictions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            ''', (user_id,))
            return cursor.fetchall()


    def analyze_cbc_report(self, user_id, report_name, report_data):
        """
        Analyze CBC report data using AI interpretation
        
        Args:
            user_id (int): The ID of the user submitting the report
            report_name (str): Name of the report file
            report_data (str): The extracted text content from the report
            
        Returns:
            dict: AI analysis and interpretation of the CBC report
        """
        # Generate AI interpretation
        prompt = f"""
            You are MediCareAI, a helpful, empathetic, and highly intelligent virtual doctor. 
            Analyze the following CBC report text and provide insights. 
            Mention any detected abnormalities and their implications. 
            Suggest recommendations if needed.

            CBC Report Text:
            {report_data}
            """
        
        # Get AI interpretation
        response = self.chat.invoke(prompt)
        ai_analysis = response.content
        
        # Save to database
        report_id = self.save_cbc_report(
            user_id,
            report_name,
            report_data,
            ai_analysis
        )
        
        return {
            'report_id': report_id,
            'analysis': ai_analysis,
            'timestamp': datetime.now().isoformat()
        } 