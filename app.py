#app.py
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import jwt
from jwt import PyJWTError
import os
from database_manager import DatabaseManager
from automate_email import EmailService
from models import predict_disease
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import pyttsx3
from fastapi.responses import FileResponse
import speech_recognition as sr
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="MediCareAI API", description="API for MediCareAI healthcare system", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db = DatabaseManager()
email_service = EmailService(db, openai_api_key)
chat = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY is not set in environment variables")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatMessage(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    prediction_type: str
    probabilities: List[float]
    timestamp: datetime

# Helper functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception
    
    user = db.get_user(username)
    if user is None:
        raise credentials_exception
    return {"id": user[0], "username": user[1], "email": user[2]}

# Error handler for common exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to MediCareAI API"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = db.get_user(form_data.username)
        if not user or user[3] != db.hash_password(form_data.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = create_access_token(data={"sub": user[1]})
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/users", response_model=User)
async def create_user(user: UserCreate):
    try:
        user_id = db.add_user(user.username, user.email, db.hash_password(user.password))
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already exists"
            )
        return {
            "id": user_id,
            "username": user.username,
            "email": user.email,
            "created_at": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/chat")
async def chat_endpoint(
    message: ChatMessage,
    current_user: dict = Depends(get_current_user)
):
    # Create a new session if needed
    session_id = db.start_session(current_user["id"])
    
    # Get chat history
    history = db.get_chat_history(current_user["id"])
    history_text = "\nRecent conversation history:\n"
    for role, content, timestamp in history[::-1]:
        speaker = "User" if role == "user" else "MediCareAI"
        history_text += f"{speaker}: {content}\n"
    
    # Generate response using the same prompt template from main.py
    prompt = PromptTemplate(
        input_variables=["conversation_history", "user_input"],
        template=(
            """You are MediCareAI, a helpful, empathetic, and highly intelligent virtual doctor.
            Your goal is to accurately diagnose medical conditions by asking the user thoughtful and relevant questions.
            Once you gather sufficient information, provide concise yet detailed responses, including:
            1. Disease information,
            2. A suggested prescription when necessary,
            3. Tailored lifestyle recommendations, and
            4. Personalized dietary advice.
            {conversation_history}
            User Query: {user_input}"""
        )
    )
    
    formatted_prompt = prompt.format(
        conversation_history=history_text,
        user_input=message.message
    )
    
    response = chat.invoke(formatted_prompt)
    
    # Save the conversation
    db.add_chat_message(current_user["id"], session_id, "user", message.message)
    db.add_chat_message(current_user["id"], session_id, "bot", response.content)
    
    return {"response": response.content}

@app.post("/analyze-cbc")
async def analyze_cbc(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        # Read binary content
        content = await file.read()
        
        # Save temporarily for PyPDF2 to read
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        
        # Extract text from PDF
        from PyPDF2 import PdfReader
        reader = PdfReader(temp_path)
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text()
        
        # Get AI analysis
        result = db.analyze_cbc_report(
            current_user["id"],
            file.filename,
            text_content
        )
        
        return JSONResponse(
            content={
                "analysis": result
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict-disease/{disease_type}")
async def predict_disease_endpoint(
    disease_type: int,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if disease_type not in [1, 2, 3, 4]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid disease type. Must be between 1 and 4"
        )
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Make prediction
        result = predict_disease(temp_path, disease_type)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error processing the file"
            )
        
        # Map disease type to name
        disease_names = {
            1: "Hemophilia",
            2: "Anemia",
            3: "Thalassemia",
            4: "Fibrinogen"
        }
        
        # Save prediction
        prediction_type = disease_names[disease_type]
        result_str = "\n".join([f"Class {i} probability: {prob:.4f}" 
                               for i, prob in enumerate(result[0])])
        
        db.save_disease_prediction(
            current_user["id"],
            prediction_type,
            result_str,
            datetime.now()
        )
        
        return PredictionResponse(
            prediction_type=prediction_type,
            probabilities=result[0].tolist(),
            timestamp=datetime.now()
        )
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/history/chat")
async def get_chat_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    history = db.get_chat_history(current_user["id"], limit)
    return {"history": history}

@app.get("/history/cbc-reports")
async def get_cbc_reports(
    current_user: dict = Depends(get_current_user)
):
    reports = db.get_user_cbc_reports(current_user["id"])
    return {"reports": reports}

@app.get("/history/predictions")
async def get_predictions(
    current_user: dict = Depends(get_current_user)
):
    predictions = db.get_disease_predictions(current_user["id"])
    return {"predictions": predictions}

# Voice Interaction Endpoints
@app.post("/speech-to-text")
async def convert_speech_to_text(
    audio_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    recognizer = sr.Recognizer()
    try:
        # Save audio file temporarily
        temp_path = f"temp_{audio_file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Convert speech to text
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            
        return {"text": text}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

import asyncio
from fastapi.background import BackgroundTasks  # Corrected import

@app.post("/text-to-speech")
async def convert_text_to_speech(
    text: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    engine = pyttsx3.init()
    temp_path = f"temp_speech_{current_user['id']}.mp3"
    
    try:
        # Configure the engine
        engine.setProperty('rate', 150)    # Speaking rate
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Save to file
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        # Add a small delay to ensure file is written
        await asyncio.sleep(1)
        
        # Verify file exists
        if not os.path.exists(temp_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate audio file"
            )
        
        # Add cleanup task to background tasks
        background_tasks.add_task(os.remove, temp_path)
        
        # Return the file
        return FileResponse(
            path=temp_path,
            media_type="audio/mp3",
            filename="speech.mp3"
        )
    except Exception as e:
        # Clean up if there's an error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Email Service Endpoints
@app.post("/email-preferences")
async def update_email_preferences(
    preferences: dict,
    current_user: dict = Depends(get_current_user)
):
    # Update user's email preferences in database
    return {"message": "Email preferences updated successfully"}

@app.post("/trigger-health-tip")
async def send_health_tip(
    current_user: dict = Depends(get_current_user)
):
    tip = email_service.generate_health_tip(current_user["username"])
    success = email_service.send_email(
        current_user["email"],
        "Your Today's Health Tip",
        tip
    )
    return {"success": success, "message": "Health tip sent successfully"}

@app.get("/email-history")
async def get_email_history(
    current_user: dict = Depends(get_current_user)
):
    history = db.get_email_logs(current_user["id"])
    return {"history": history}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)