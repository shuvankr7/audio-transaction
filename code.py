# Import required libraries first
import streamlit as st
import tempfile
import os
import numpy as np
import time

# Make sure set_page_config is the first Streamlit command

# Set environment variables
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
GROQ_API_KEY = "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"
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1024

# Define functions for the app
def load_whisper_model():
    """Load the Whisper model with proper import handling"""
    try:
        # Import inside function to avoid circular imports
        import whisper
        model = whisper.load_model("base")
        return model, None
    except ImportError:
        return None, "Whisper module not found. Please install it with 'pip install openai-whisper'"
    except Exception as e:
        return None, f"Error loading Whisper model: {str(e)}"

def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        # Import inside function
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
        return llm, None
    except Exception as e:
        return None, f"Error initializing RAG system: {str(e)}"

def process_transaction_message(message, llm):
    """Process transaction messages using LLM"""
    if llm is None:
        return "Error: RAG system is not initialized."
    system_prompt = (
        "Your input is a transaction message extracted from voice. Extract structured details likeAmount, Transaction Type, Bank Name, Card Type, paied to whom,marchent, Transaction Mode, Transaction Date, Reference Number, and tag."
        "Tag meaning which category of spending, if amazon then shopping etc, if zomato then eating"
        "Just give the json output, Don't say anything else , if there is no output then don't predict, say it is null"
        "If mode of payment is not mentioned, assume cash by default. "
        "If any field is missing, set it as null. "
        "Return only a JSON or a list of JSON objects."
        "as human giving input ,so input can be of few worlds and less structured gramatically and simple"
        "example 1: today I spent 500 at dominoze,you need to handle it carefully"
        "IF USER GIVES MULTIPLE ITEMS CORROSPONDING TO MULTIPLE PRICES THEN GENERATE LIST OF JESON CORROSPONDINGLY"
    )
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response.content if hasattr(response, 'content') else response

def init_session_state():
    """Initialize session state variables"""
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = None
    if 'rag_llm' not in st.session_state:
        st.session_state.rag_llm = None

# Initialize session state
init_session_state()

# Main app layout
st.markdown("<h1 style='text-align: center;'>🔊 Audio Transaction Processor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for loading models
with st.sidebar:
    st.header("Model Configuration")
    
    if st.button("Load Whisper Model"):
        with st.spinner("Loading Whisper model..."):
            model, error = load_whisper_model()
            if model:
                st.session_state.whisper_model = model
                st.success("✅ Whisper model loaded successfully!")
            else:
                st.error(f"❌ {error}")
    
    if st.button("Initialize LLM"):
        with st.spinner("Initializing LLM..."):
            llm, error = initialize_rag_system()
            if llm:
                st.session_state.rag_llm = llm
                st.success("✅ LLM initialized successfully!")
            else:
                st.error(f"❌ {error}")

# Main content area
st.markdown("### 🎤 Record Your Transaction")

# Check if models are loaded
models_ready = st.session_state.whisper_model is not None and st.session_state.rag_llm is not None
if not models_ready:
    st.warning("⚠️ Please load the Whisper model and initialize the LLM from the sidebar before proceeding.")

# Voice recording section - only enable if models are loaded
audio_recorder_disabled = not models_ready
audio_bytes = st.audio_recorder(
    text="Click to record", 
    recording_color="#e8b62c", 
    neutral_color="#6aa36f" if not audio_recorder_disabled else "#cccccc", 
    stop_recording_text="Click to stop recording",
    disabled=audio_recorder_disabled
)

if audio_bytes and models_ready:
    st.markdown("**🎵 Audio Preview:**")
    st.audio(audio_bytes, format="audio/wav")
    
    if st.button('🎤 Transcribe Audio'):
        try:
            with st.spinner("⏳ Transcribing audio... Please wait."):
                # Save audio bytes to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name
                
                # Transcribe using Whisper
                result = st.session_state.whisper_model.transcribe(tmp_file_path)
                transcription = result.get("text", "")
            
                # Clean up temp file
                os.unlink(tmp_file_path)
            
            if not transcription:
                st.error("❌ No transcription output. Please check your audio recording.")
            else:
                # Store transcription in session state
                st.session_state.transcription = transcription
                st.success("✅ Transcription successful!")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# If transcription exists, show editable text area
if 'transcription' in st.session_state:
    st.markdown("### ✏️ Edit Transcription Before Processing")
    edited_transcription = st.text_area("", st.session_state.transcription, height=200)
    
    if st.button('💼 Process Transaction Details'):
        with st.spinner("🤖 Processing transaction details..."):
            processed_result = process_transaction_message(edited_transcription, st.session_state.rag_llm)
            if processed_result:
                st.markdown("### 📊 Extracted Transaction Details")
                st.code(processed_result, language="json")
            else:
                st.error("❌ Failed to process transaction details.")
