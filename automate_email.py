#automate_email.py
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import schedule
import time
import threading
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class EmailService:
    def __init__(self, db_manager, openai_api_key):
        self.db = db_manager
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email = "tech.swa239@gmail.com"
        self.password = "lzuu pynw dlqj yxsu"
        self.chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)
        
        # Health tip prompt template
        self.tip_prompt = PromptTemplate(
            input_variables=["username"],
            template=(
                """Generate a personalized daily health tip for {username}. 
                The tip should be concise, practical, and easy to implement. 
                Include a friendly greeting and sign-off.
                Keep the entire message under 200 words."""
            )
        )
    
    def generate_health_tip(self, username):
        """Generate a personalized health tip using OpenAI"""
        formatted_prompt = self.tip_prompt.format(username=username)
        response = self.chat.invoke(formatted_prompt)
        return response.content
    
    def send_email(self, recipient_email, subject, body):
        """Send an email to a specific recipient"""
        try:
            message = MIMEMultipart()
            message["From"] = self.email
            message["To"] = recipient_email
            message["Subject"] = subject
            
            message.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(message)
                
            return True
        except Exception as e:
            print(f"Error sending email to {recipient_email}: {str(e)}")
            return False
    
    def send_health_tips(self):
        """Send health tips to all users"""
        print(f"Sending health tips at {datetime.now()}")
        
        # Get all users from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, username, email FROM users")
            users = cursor.fetchall()
        
        for user_id, username, email in users:
            # Generate personalized health tip
            tip = self.generate_health_tip(username)
            
            # Send email
            subject = f"Your Daily Health Tip - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
            success = self.send_email(email, subject, tip)
            
            # Log the email in database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO email_logs (user_id, email_type, content, status)
                VALUES (?, ?, ?, ?)
                ''', (user_id, 'health_tip', tip, 'sent' if success else 'failed'))
                conn.commit()
    
    def schedule_emails(self):
        """Schedule emails to be sent at specific times"""
        schedule.every().day.at("12:00").do(self.send_health_tips)
        schedule.every().day.at("18:00").do(self.send_health_tips)
        schedule.every().day.at("00:00").do(self.send_health_tips)
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def start(self):
        """Start the email service"""
        self.schedule_emails()
        scheduler_thread = threading.Thread(target=self.run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()