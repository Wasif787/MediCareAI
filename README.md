# MediCareAI

MediCareAI is an intelligent healthcare platform that combines AI-powered diagnostics with a user-friendly interface to provide accessible medical assistance. The system offers disease prediction, CBC report analysis, real-time medical chat assistance, and voice interaction capabilities to help users understand their health conditions and receive guidance.

## Features

### Disease Prediction
- **Multiple Disease Models**: Predict and analyze four different types of blood disorders:
  - Hemophilia (APTT levels)
  - Anemia (Classification of different types)
  - Thalassemia (Types and severity)
  - Fibrinogen (Levels and related conditions)
- **Visual Results**: Interactive charts and detailed probability breakdowns
- **AI-Powered Recommendations**: Personalized next steps based on prediction results

### CBC Report Analysis
- **PDF Processing**: Upload and analyze Complete Blood Count reports
- **Comprehensive Analysis**: Detailed breakdown of blood parameters and their significance
- **Exportable Reports**: Download analysis as TXT or PDF for future reference

### AI Medical Assistant
- **Natural Language Chat**: Conversational interface to discuss health concerns
- **Medical Knowledge**: Provides disease information, prescription suggestions, lifestyle recommendations, and dietary advice
- **Conversation History**: Save and review past conversations

### Voice Interface
- **Voice Command Recognition**: Speak directly to the AI assistant
- **Text-to-Speech Responses**: Listen to AI-generated responses
- **Seamless Interaction**: Natural conversation flow for hands-free usage

### User Management
- **Secure Authentication**: JWT-based user authentication
- **Personal Health Dashboard**: Track prediction history and past conversations
- **Email Notifications**: Customizable health tips and reminders

## Technology Stack

### Backend
- **FastAPI**: High-performance API framework
- **JWT Authentication**: Secure token-based authentication
- **OpenAI Integration**: GPT-4 powered medical responses
- **PyTorch**: ML models for disease prediction
- **Speech Recognition**: Voice command processing
- **PyPDF2**: PDF document analysis

### Frontend
- **Streamlit**: Interactive web interface
- **Plotly**: Data visualization and charts
- **Pandas**: Data processing and management
- **Audio Processing**: Recording and playback capabilities

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Environment Setup
1. Clone the repository
```bash
git clone https://github.com/Wasif787/medicare-ai.git
cd medicare-ai
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
Create a `.env` file with the following variables:
```
JWT_SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_openai_api_key
```

### Running the Application
1. Start the FastAPI backend
```bash
uvicorn app:app --reload
```

2. In a separate terminal, start the Streamlit frontend
```bash
streamlit run streamlit.py
```

3. Access the application at `http://localhost:8501`

## Usage Guide

### First-time Setup
1. Register a new account using the registration form
2. Log in with your credentials
3. Navigate through the sidebar menu to access different features

### Disease Prediction
1. Select the Disease Prediction page
2. Choose the disease type from the dropdown
3. Upload your CBC report in PDF format
4. Click "Predict" to see results and recommendations

### Chat with MediCareAI
1. Go to the Chat page
2. Type your medical question
3. Receive AI-generated medical advice

### Voice Interface
1. Navigate to the Voice Interface page
2. Click on the microphone icon and speak your question
3. Listen to the AI response

## Future Enhancements
- Mobile application support
- Integration with wearable health devices
- Expanded disease prediction models
- Multilingual support
- Telemedicine appointment scheduling

## License
[MIT License](LICENSE)

## Disclaimer
MediCareAI is designed as an aid to healthcare and should not replace professional medical advice. Always consult with a healthcare professional for medical concerns.
