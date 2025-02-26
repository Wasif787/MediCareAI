#main.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pyttsx3
import speech_recognition as sr
import pdfplumber
from datetime import datetime
import hashlib
from database_manager import DatabaseManager
from automate_email import EmailService
from models import predict_disease  

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key is not set. Please check your .env file.")

# Initialize the OpenAI Chat Model with the API key
chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

class MediCareAI:
    def __init__(self):
        self.db = DatabaseManager()
        self.current_user = None
        self.current_session = None
        self.email_service = EmailService(self.db, openai_api_key)
        self.email_service.start()
        
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def login(self):
        while True:
            print("\nWelcome to MediCareAI!")
            print("1. Login")
            print("2. Register")
            print("3. Exit")
            
            choice = input("Choose an option (1-3): ").strip()
            
            if choice == "1":
                username = input("Username: ").strip()
                password = input("Password: ").strip()
                
                user = self.db.get_user(username)
                if user and user[3] == self.hash_password(password):
                    self.current_user = {"id": user[0], "username": user[1], "email": user[2]}
                    self.current_session = self.db.start_session(self.current_user["id"])
                    print(f"Welcome back, {username}!")
                    return True
                print("Invalid username or password.")
                
            elif choice == "2":
                username = input("Choose username: ").strip()
                email = input("Enter email: ").strip()
                password = input("Choose password: ").strip()
                
                user_id = self.db.add_user(username, email, self.hash_password(password))
                if user_id:
                    print("Registration successful! Please login.")
                else:
                    print("Username or email already exists.")
                    
            elif choice == "3":
                return False
    
    def handle_query(self, user_input):
        # Add user message to database
        self.db.add_chat_message(
            self.current_user["id"],
            self.current_session,
            "user",
            user_input
        )
        
        # Get recent chat history for context
        history = self.db.get_chat_history(self.current_user["id"])
        history_text = "\nRecent conversation history:\n"
        for role, content, timestamp in history[::-1]:  # Reverse to get chronological order
            speaker = "User" if role == "user" else "MediCareAI"
            history_text += f"{speaker}: {content}\n"
        
        # Format the prompt with user input and conversation history
        formatted_prompt = prompt.format(
            user_input=user_input,
            conversation_history=history_text
        )
        
        # Generate the response
        response = chat.invoke(formatted_prompt)
        
        # Add bot response to database
        self.db.add_chat_message(
            self.current_user["id"],
            self.current_session,
            "bot",
            response.content
        )
        
        return response.content
    
    def analyze_cbc_report(self, file_path):
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()

            if not text.strip():
                return "The PDF report is empty or could not be read. Please check the file."

            prompt = f"""
            You are MediCareAI, a helpful, empathetic, and highly intelligent virtual doctor. 
            Analyze the following CBC report text and provide insights. 
            Mention any detected abnormalities and their implications. 
            Suggest recommendations if needed.

            CBC Report Text:
            {text}
            """

            response = chat.invoke(prompt)
            
            # Save the report and analysis in the database
            report_name = os.path.basename(file_path)
            self.db.save_cbc_report(
                self.current_user["id"],
                report_name,
                text,
                response.content
            )
            
            return response.content

        except FileNotFoundError:
            return "Error: The specified file was not found."
        except Exception as e:
            return f"Error processing the file: {str(e)}"
    
    def display_history_menu(self):
        while True:
            print("\nHistory Options:")
            print("1. View Recent Chat History")
            print("2. View CBC Report History")
            print("3. View Disease Prediction History")
            print("4. Return to Main Menu")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                history = self.db.get_chat_history(self.current_user["id"], 10)
                if not history:
                    print("\nNo chat history available.")
                else:
                    print("\n=== Chat History ===")
                    for role, content, timestamp in history[::-1]:
                        speaker = "You" if role == "user" else "MediCareAI"
                        print(f"[{timestamp}] {speaker}: {content}\n")
                        
            elif choice == "2":
                reports = self.db.get_user_cbc_reports(self.current_user["id"])
                if not reports:
                    print("\nNo CBC reports found.")
                else:
                    print("\n=== CBC Reports History ===")
                    for report_id, name, analysis, date in reports:
                        print(f"\nReport: {name}")
                        print(f"Date: {date}")
                        print("Analysis Summary:", analysis[:200] + "..." if len(analysis) > 200 else analysis)
            
            elif choice == "3":
                predictions = self.db.get_disease_predictions(self.current_user["id"])
                if not predictions:
                    print("\nNo disease predictions found.")
                else:
                    print("\n=== Disease Prediction History ===")
                    for pred_type, result, timestamp in predictions:
                        print(f"\nPrediction Type: {pred_type}")
                        print(f"Date: {timestamp}")
                        print("Results:")
                        print(result)
                        print("-" * 50)
                        
            elif choice == "4":
                break
    
    def predict_diseases(self):
        while True:
            print("\nDisease Prediction System")
            print("=" * 30)
            print("1. Hemophilia Prediction")
            print("2. Anemia Prediction")
            print("3. Thalassemia Prediction")
            print("4. Fibrinogen Prediction")
            print("5. Return to Main Menu")
            
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "5":
                    break
                    
                if choice not in ["1", "2", "3", "4"]:
                    print("Invalid choice. Please select a number between 1 and 5.")
                    continue
                    
                pdf_path = input("Enter the path to your CBC PDF file: ").strip()
                
                # Save prediction in database
                result = predict_disease(pdf_path, int(choice))
                if result is not None:
                    prediction_type = {
                        "1": "Hemophilia",
                        "2": "Anemia",
                        "3": "Thalassemia",
                        "4": "Fibrinogen"
                    }[choice]
                    
                    # Convert probabilities to string for storage
                    result_str = "\n".join([f"Class {i} probability: {prob:.4f}" 
                                          for i, prob in enumerate(result[0])])
                    
                    # Save prediction in database
                    self.db.save_disease_prediction(
                        self.current_user["id"],
                        prediction_type,
                        result_str,
                        datetime.now()
                    )
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")                

    def run(self):
        if not self.login():
            print("Goodbye!")
            return

        while True:
            print("\nPlease select an option:")
            print("1. Text Interaction")
            print("2. Voice Interaction")
            print("3. Upload CBC Report for Analysis (PDF)")
            print("4. Disease Prediction")
            print("5. History Options")
            print("6. Logout")

            choice = input("Enter your choice (1-6): ").strip()

            if choice == "1":
                print("\nText Interaction Mode")
                while True:
                    user_input = input("\nYou: ")
                    if user_input.lower() == "exit":
                        break
                    response = self.handle_query(user_input)
                    print(f"{response}")

            elif choice == "2":
                print("\nVoice Interaction Mode")
                while True:
                    user_input = listen()
                    if "exit" in user_input.lower():
                        break
                    print(f"\nYou: {user_input}")
                    response = self.handle_query(user_input)
                    print(f"{response}")
                    speak(response)

            elif choice == "3":
                print("\nCBC Report Analysis")
                file_path = input("Enter the file path of the CBC report (PDF): ").strip()
                result = self.analyze_cbc_report(file_path)
                print(f"\nAnalysis Result:\n{result}")

            elif choice == "4":
                self.predict_diseases()

            elif choice == "5":
                self.display_history_menu()

            elif choice == "6":
                if self.current_session:
                    self.db.end_session(self.current_session)
                self.current_user = None
                self.current_session = None
                print("Logged out successfully.")
                if not self.login():
                    break

# Keep the original prompt template
prompt = PromptTemplate(
    input_variables=["conversation_history", "user_input"],
    template=(
        """You are MediCareAI, a helpful, empathetic, and highly intelligent virtual doctor.
Your goal is to accurately diagnose medical conditions by asking the user thoughtful and relevant questions, but only one question at a time.
Once you gather sufficient information, provide concise yet detailed responses, including:

1. Disease information,
2. A suggested prescription when necessary,
3. Tailored lifestyle recommendations, and
4. Personalized dietary advice.

Ensure your responses are in a kind, professional tone and show genuine care for the user's well-being.
Begin by asking questions to better understand the user's symptoms, history, and concerns before providing advice.
Always prioritize safety, suggesting consultation with a human doctor for serious or unclear cases.

{conversation_history}
User Query: {user_input}"""
    )
)

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."

if __name__ == "__main__":
    medicareai = MediCareAI()
    medicareai.run()