#streamlit.py
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
from audio_recorder_streamlit import audio_recorder
import io
import soundfile as sf
import time
import os
import pyttsx3
import speech_recognition as sr
import base64
from dotenv import load_dotenv
import openai

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Constants
API_URL = "http://127.0.0.1:8000"
PAGES = ["Login", "Register", "Dashboard", "Chat", "Voice Interface", "CBC Analysis", "Disease Prediction", "Email Preferences"]

# Session state initialization
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"

def get_recommendation(prediction_data, disease_name):
    """Sends the prediction results to OpenAI API and returns a next-step recommendation."""
    try:
        if disease_name == "Thalassemia":
            prompt = f"""
            Based on the CBC report analysis, here are the probabilities for different conditions under {disease_name}:

            {', '.join([f"{k}: {v*100:.3f}" for k, v in prediction_data.items()])}

            As a medical assistant AI, provide a concise next-step recommendation for the patient. 
            - If all probabilities are low, reassure the patient that their results appear normal.
            - If any condition has a significant probability, suggest whether they should consult a specialist, take further tests, or monitor their symptoms.
            - If probabilities are ambiguous, advise continued health monitoring and consulting a doctor if symptoms appear.
            - Keep the response medically accurate, avoiding unnecessary concern if no risk is detected.
            - Keep the recommendation within 3-4 sentences.
            """
        elif disease_name == "Anemia":
            prompt = f"""
            You are a medical assistant AI. Based on the CBC report analysis, the following findings were observed:

            {', '.join([f"{k}: {v*100:.3f}" for k, v in prediction_data.items()])}

            Please provide a concise next-step recommendation based on the findings:
            - If the probability of a condition is high (above 0.7), suggest seeing a specialist or further testing.
            - If the probability is moderate (0.3-0.7), recommend monitoring symptoms or performing additional tests.
            - If the probability is low (below 0.3), suggest continuing regular check-ups or monitoring health without immediate action.

            Ensure the response is brief and only covers the relevant condition(s) with clear advice based on the probabilities. Be empathetic but direct.
            """
        elif disease_name == "Hemophilia":
            prompt = f"""
            You are a medical assistant AI. Based on the CBC report analysis, the following APTT findings were observed:

            {', '.join([f"Class {k}: {v*100:.3f}" for k, v in prediction_data.items()])}

            Please provide a concise next-step recommendation based on the findings:
            - If Class 0 (Low APTT) is detected with high probability, suggest consultation with a hematologist for thrombosis risk evaluation and possible anticoagulant therapy.
            - If Class 1 (Normal APTT) is detected with high probability, reassure the patient that their clotting factor is normal, but regular health check-ups are recommended.
            - If Class 2 (High APTT) is detected with high probability, suggest further investigation for hemophilia or other clotting disorders and consultation with a hematologist for potential treatment options.

            Ensure the response is brief, addressing only the relevant condition, and provide actionable advice.
            """
        elif disease_name == "Fibrinogen":
            prompt = f"""
            You are a medical assistant AI. Based on the CBC report analysis, the following class probabilities were observed:

            {', '.join([f"Class {k}: {v*100:.3f}" for k, v in prediction_data.items()])}

            Please provide a concise next-step recommendation based on the findings:
            - If Class 1 (Normal) is detected with high probability, reassure the patient that their health status is normal and no immediate action is necessary. Regular health check-ups are recommended.
            - If Class 2 (Infections/Inflammation - Sepsis, Pneumonia) is detected with high probability, suggest consulting with a healthcare provider immediately for further diagnostic tests and treatment options.
            - If Class 3 (Bleeding/Clotting Disorders - Hemophilia) is detected with high probability, recommend a consultation with a hematologist for further investigation and appropriate management, such as potential treatment or genetic counseling.

            Make sure the response is brief and addresses only the relevant condition, providing clear, actionable next steps.
            """

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a medical assistant AI."},
                      {"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error getting recommendation: {str(e)}"
    

# Utility functions
def make_api_request(endpoint, method="GET", data=None, files=None):
    headers = {}
    if st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    
    try:
        if method == "GET":
            response = requests.get(f"{API_URL}{endpoint}", headers=headers)
        else:
            response = requests.post(f"{API_URL}{endpoint}", headers=headers, data=data, files=files)
        
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def login_page():
    st.title("MediCareAI Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            response = requests.post(
                f"{API_URL}/token",
                data={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                st.session_state.access_token = response.json()["access_token"]
                st.session_state.current_page = "Dashboard"
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    if st.button("Don't have an account? Register"):
        st.session_state.current_page = "Register"
        st.rerun()

def register_page():
    st.title("MediCareAI Registration")
    
    with st.form("register_form"):
        username = st.text_input("Username", help="Username must be at least 6 characters long")
        email = st.text_input("Email", help="Please enter a valid email address (e.g., user@example.com)")
        password = st.text_input("Password", type="password", help="Password must be at least 6 characters long")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            # Input validation
            validation_errors = []
            
            # Username validation
            if not username:
                validation_errors.append("Username is required.")
            elif len(username) < 6:
                validation_errors.append("Username must be at least 6 characters long.")
            
            # Email validation
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not email:
                validation_errors.append("Email is required.")
            elif not re.match(email_pattern, email):
                validation_errors.append("Please enter a valid email address.")
            
            # Password validation
            if not password:
                validation_errors.append("Password is required.")
            elif len(password) < 6:
                validation_errors.append("Password must be at least 6 characters long.")
            elif password != confirm_password:
                validation_errors.append("Passwords do not match.")
            
            # Display validation errors if any
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                try:
                    # Make the API request
                    response = requests.post(
                        f"{API_URL}/users",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({
                            "username": username,
                            "email": email,
                            "password": password
                        })
                    )
                    
                    # Check response status code
                    if response.status_code == 200:
                        st.success("Registration successful! Please login.")
                        time.sleep(2)  # Give user time to read success message
                        st.session_state.current_page = "Login"
                        st.rerun()
                    else:
                        # Parse error message from response
                        error_data = response.json()
                        if response.status_code == 400:
                            if "Username or email already exists" in error_data.get("detail", ""):
                                st.error("This email or username is already registered. Please use a different one.")
                            else:
                                st.error(error_data.get("detail", "Registration failed. Please try again."))
                        elif response.status_code == 422:
                            st.error("Invalid email format. Please check your email address.")
                        else:
                            st.error(f"Registration failed: {error_data.get('detail', 'Unknown error occurred.')}")
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}. Please try again later.")
                except json.JSONDecodeError:
                    st.error("Server response error. Please try again later.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
    
    # Add some spacing
    st.write("")
    
    # Login link with better styling
    col1, col2, col3 = st.columns([0.0001, 2, 1])
    with col2:
        if st.button("Already have an account? Login", type="secondary"):
            st.session_state.current_page = "Login"
            st.rerun()

def dashboard_page():
    st.title("MediCareAI Dashboard")
    
    # Initialize session state for pagination
    if 'chat_page' not in st.session_state:
        st.session_state.chat_page = 1
    
    ITEMS_PER_PAGE = 6  # Number of chat messages per page
    
    # Get user's history
    chat_history = make_api_request("/history/chat")
    cbc_reports = make_api_request("/history/cbc-reports")
    predictions = make_api_request("/history/predictions")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Chat Sessions", len(chat_history.get("history", [])) if chat_history else 0)
    with col2:
        st.metric("CBC Reports", len(cbc_reports.get("reports", [])) if cbc_reports else 0)
    with col3:
        st.metric("Disease Predictions", len(predictions.get("predictions", [])) if predictions else 0)
    
    # Display chat history with pagination
    if chat_history and chat_history.get("history"):
        st.subheader("Conversation History")
        
        # Calculate total pages
        total_messages = len(chat_history["history"])
        total_pages = (total_messages + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        
        # Display chat messages for current page
        start_idx = (st.session_state.chat_page - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_messages)
        
        # Create expandable container for each day's messages
        messages_by_date = {}
        for msg in chat_history["history"][start_idx:end_idx]:
            date = datetime.strptime(msg[2], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            if date not in messages_by_date:
                messages_by_date[date] = []
            messages_by_date[date].append(msg)
        
        # Display messages grouped by date
        for date in sorted(messages_by_date.keys(), reverse=True):
            with st.expander(f"Messages from {date}", expanded=True):
                for msg in messages_by_date[date]:
                    with st.chat_message(msg[0]):
                        st.write(f"{msg[1]}")
                        st.caption(f"Time: {msg[2]}")
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.chat_page > 1:
                if st.button("← Previous"):
                    st.session_state.chat_page -= 1
                    st.rerun()
        
        with col2:
            st.write(f"Page {st.session_state.chat_page} of {total_pages}")
        
        with col3:
            if st.session_state.chat_page < total_pages:
                if st.button("Next →"):
                    st.session_state.chat_page += 1
                    st.rerun()
        
        # Option to reset to first page
        if st.session_state.chat_page > 1:
            if st.button("Back to Latest Messages"):
                st.session_state.chat_page = 1
                st.rerun()

def chat_page():
    st.title("Chat with MediCareAI")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What do you want to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = make_api_request(
            "/chat",
            method="POST",
            data=json.dumps({"message": prompt})
        )
        
        if response:
            with st.chat_message("assistant"):
                st.markdown(response["response"])
            st.session_state.messages.append({"role": "assistant", "content": response["response"]})

def disease_prediction_page():
    st.title("Disease Prediction")
    
    # Create a mapping of disease types with descriptions
    disease_types = {
        "1": {
            "name": "Hemophilia",
            "description": "Predicts APTT levels: Low (thrombosis), Normal, or High (Hemophilia)",
            "classes": ["Low APTT (thrombosis)", "Normal APTT", "High APTT (Hemophilia)"]
        },
        "2": {
            "name": "Anemia",
            "description": "Classifies different types of anemia based on blood cell characteristics",
            "classes": [
                "Normocytic,Normochromic",
                "Normocytic",
                "Microcytosis",
                "Hypochromia",
                "Macrocytosis",
                "Macrocytic",
                "Isopoikilocytosis"
            ]
        },
        "3": {
            "name": "Thalassemia",
            "description": "Identifies different types of thalassemia conditions",
            "classes": [
                "Normal",
                "Alpha Thalassemia",
                "Beta Thalassemia",
                "Delta Beta Thalassemia",
                "Hereditary Persistence of Fetal Hemoglobin"
            ]
        },
        "4": {
            "name": "Fibrinogen",
            "description": "Analyzes fibrinogen levels and related conditions",
            "classes": ["Normal", "Infections/Inflammation", "Bleeding/Clotting Disorders"]
        }
    }
    
    # Disease type selection with descriptions
    selected_type = st.selectbox(
        "Select Disease Type",
        list(disease_types.keys()),
        format_func=lambda x: f"{disease_types[x]['name']} - {disease_types[x]['description']}"
    )
    
    # File upload - restricted to PDF only
    uploaded_file = st.file_uploader(
        "Upload CBC Report (PDF only)", 
        type=["pdf"],
        help="Please upload a Complete Blood Count (CBC) report in PDF format"
    )
    
    if uploaded_file and st.button("Predict"):
        try:
            with st.spinner("Analyzing CBC report..."):
                files = {"file": uploaded_file}
                response = make_api_request(
                    f"/predict-disease/{selected_type}",
                    method="POST",
                    files=files
                )
            
            if response and "probabilities" in response:
                st.success("Prediction Complete!")

                # Extract class names and probabilities
                classes = disease_types[selected_type]["classes"]
                probabilities = response["probabilities"]
                prediction_data = {classes[i]: probabilities[i] for i in range(len(classes))}

                # Generate Next-Step Recommendation
                recommendation = get_recommendation(prediction_data, disease_types[selected_type]["name"])

                # Visualization
                df = pd.DataFrame({"Class": classes, "Probability": probabilities})
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = px.bar(
                        df,
                        x="Class",
                        y="Probability",
                        title=f"{disease_types[selected_type]['name']} Prediction Results",
                        labels={"Probability": "Probability", "Class": "Classification"},
                        text=df["Probability"].apply(lambda x: f"{x:.2f}")  # Changed to 2 decimal points
                    )
                    fig.update_traces(
                        textposition="outside", 
                        marker_color="rgb(0, 123, 255)",
                        texttemplate='%{text}'  # Ensures exact text values are displayed
                    )
                    fig.update_layout(
                        showlegend=False, 
                        xaxis_tickangle=-45, 
                        yaxis_range=[0, 1]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Detailed Results")
                    for class_name, prob in prediction_data.items():
                        st.metric(label=class_name, value=f"{prob:.2f}")  # Changed to 2 decimal points

                # Display Next-Step Recommendation
                st.subheader("🔍 Next Step Recommendation")
                st.info(recommendation)

                # Save to history
                st.caption(f"Prediction made at: {response.get('timestamp', datetime.now())}")

            else:
                st.error("Failed to get prediction results. Please try again.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def cbc_analysis_page():
    st.title("CBC Report Analysis")
    
    uploaded_file = st.file_uploader("Upload CBC Report", type=["pdf"])
    
    if uploaded_file and st.button("Analyze"):
        files = {"file": uploaded_file}
        response = make_api_request(
            "/analyze-cbc",
            method="POST",
            files=files
        )
        
        if response:
            st.success("Analysis Complete!")
            
            # Format the analysis results as text
            analysis_text = "CBC Report Analysis Results\n"
            analysis_text += "=" * 30 + "\n\n"
            
            for category, details in response["analysis"].items():
                analysis_text += f"{category.upper()}\n"
                analysis_text += "-" * len(category) + "\n"
                if isinstance(details, dict):
                    for key, value in details.items():
                        analysis_text += f"{key}: {value}\n"
                else:
                    analysis_text += f"{details}\n"
                analysis_text += "\n"
            
            # Display formatted text
            st.text_area("Analysis Results", analysis_text, height=400)
            
            # Create download button for the report
            st.download_button(
                label="📥 Download as txt",
                data=analysis_text,
                file_name="cbc_analysis_report.txt",
                mime="text/plain"
            )
            
            # Add an option to download as PDF
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from io import BytesIO
            
            def create_pdf_report(analysis_text):
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Add title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=16,
                    spaceAfter=30
                )
                story.append(Paragraph("CBC Report Analysis Results", title_style))
                
                # Add content
                for line in analysis_text.split('\n'):
                    if line.isupper():  # Category headers
                        style = styles['Heading2']
                    else:
                        style = styles['Normal']
                    if line.strip():  # Skip empty lines
                        story.append(Paragraph(line, style))
                        story.append(Spacer(1, 6))
                
                doc.build(story)
                return buffer.getvalue()
            
            pdf_report = create_pdf_report(analysis_text)
            st.download_button(
                label="📥 Download as PDF",
                data=pdf_report,
                file_name="cbc_analysis_report.pdf",
                mime="application/pdf"
            )

def voice_interface_page():
    st.title("Voice Conversation with MediCareAI")
    
    # Initialize session state for conversation
    if "voice_conversation" not in st.session_state:
        st.session_state.voice_conversation = []
    
    # Initialize TTS engine
    engine = pyttsx3.init()
    
    # Display conversation history
    st.subheader("Conversation History")
    for message in st.session_state.voice_conversation:
        with st.chat_message(message["role"]):
            st.write(f"🎵 Voice Message: {message['content']}")
    
    # Record audio
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=44100,
        text="Conversation active..."
    )
    
    if audio_bytes:
        # Process the recorded audio
        try:
            # Save temporary audio file
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_bytes)
            
            # Convert speech to text
            recognizer = sr.Recognizer()
            with sr.AudioFile("temp_audio.wav") as source:
                audio = recognizer.record(source)
                user_message = recognizer.recognize_google(audio)
            
            # Add user message to conversation
            st.session_state.voice_conversation.append({
                "role": "user",
                "content": user_message
            })
            
            # Get AI response
            with st.spinner("MediCareAI is thinking..."):
                response = make_api_request(
                    "/chat",
                    method="POST",
                    data=json.dumps({"message": user_message})
                )
            
            if response:
                ai_message = response["response"]
                
                # Add AI response to conversation
                st.session_state.voice_conversation.append({
                    "role": "assistant",
                    "content": ai_message
                })
                
                # Convert AI response to speech and play it automatically
                engine.save_to_file(ai_message, "temp_response.mp3")
                engine.runAndWait()
                
                # Create a container for the AI response
                with st.container():
                    st.markdown("**MediCareAI's Response:**")
                    st.write(ai_message)
                    
                    # Auto-play the response using HTML audio element with autoplay
                    st.markdown(
                        f"""
                        <audio autoplay>
                            <source src="data:audio/mp3;base64,{base64.b64encode(open('temp_response.mp3', 'rb').read()).decode()}" type="audio/mp3">
                        </audio>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Clean up temporary files
                if os.path.exists("temp_response.mp3"):
                    os.remove("temp_response.mp3")
                    
        except Exception as e:
            st.error(f"Error processing voice: {str(e)}")
        finally:
            # Clean up temporary audio file
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
    
    # Clear conversation button
    if st.session_state.voice_conversation:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.voice_conversation = []
            st.rerun()

    # Add usage instructions
    with st.expander("📝 How to use voice conversation"):
        st.markdown("""
        1. Click the microphone icon to start recording
        2. Speak clearly into your microphone
        3. Recording will automatically stop after you finish speaking
        4. Wait for MediCareAI to process and respond
        5. MediCareAI will automatically speak its response
        6. Repeat the process for a natural conversation
        
        **Note:** Make sure your browser has permission to access your microphone.
        """)

def email_preferences_page():
    st.title("Email Preferences")
    
    with st.form("email_preferences"):
        daily_tips = st.checkbox("Receive Daily Health Tips")
        weekly_summary = st.checkbox("Receive Weekly Health Summary")
        appointment_reminders = st.checkbox("Receive Appointment Reminders")
        submit = st.form_submit_button("Update Preferences")
        
        if submit:
            preferences = {
                "daily_tips": daily_tips,
                "weekly_summary": weekly_summary,
                "appointment_reminders": appointment_reminders
            }
            
            response = make_api_request(
                "/email-preferences",
                method="POST",
                data=json.dumps(preferences)
            )
            
            if response:
                st.success("Preferences updated successfully!")
    
    if st.button("Send Test Health Tip"):
        response = make_api_request("/trigger-health-tip", method="POST")
        if response:
            st.success("Health tip sent successfully!")

# Main app
def main():
    st.set_page_config(
        page_title="MediCareAI",
        page_icon="🏥",
        layout="wide"
    )
    
    # Sidebar navigation (only show when logged in)
    if st.session_state.access_token:
        st.sidebar.title("Navigation")
        selected_page = st.sidebar.selectbox("Go to", PAGES[2:])  # Skip login/register
        if st.sidebar.button("Logout"):
            st.session_state.access_token = None
            st.session_state.current_page = "Login"
            st.rerun()
    else:
        selected_page = st.session_state.current_page
    
    # Page routing
    if selected_page == "Login" or (not st.session_state.access_token and st.session_state.current_page == "Login"):
        login_page()
    elif selected_page == "Register" or (not st.session_state.access_token and st.session_state.current_page == "Register"):
        register_page()
    elif st.session_state.access_token:
        if selected_page == "Dashboard":
            dashboard_page()
        elif selected_page == "Chat":
            chat_page()
        elif selected_page == "Disease Prediction":
            disease_prediction_page()
        elif selected_page == "CBC Analysis":
            cbc_analysis_page()
        elif selected_page == "Voice Interface":
            voice_interface_page()
        elif selected_page == "Email Preferences":
            email_preferences_page()
    else:
        st.warning("Please login to access this page")
        login_page()

if __name__ == "__main__":
    main()