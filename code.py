import streamlit as st
import tempfile
import os
import numpy as np
import wave
import time

# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import LangChain after environment variables are set
from langchain_groq import ChatGroq

# Default Groq API key (Ensure this is kept secure)
GROQ_API_KEY = "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"
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1024

# Load whisper model at startup
@st.cache_resource
def load_whisper_model():
    try:
        import whisper
        return whisper.load_model("base")
    except ImportError:
        st.error(" Whisper module not found. Please ensure it's installed correctly.")
        st.stop()
    except Exception as e:
        st.error(f" Error loading Whisper model: {str(e)}")
        st.stop()

# Initialize RAG system internally
def initialize_rag_system():
    try:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
    except Exception as e:
        st.error(f" Error initializing RAG system: {str(e)}")
        return None

# Load models on startup
whisper_model = load_whisper_model()
rag_llm = initialize_rag_system()

def process_transaction_message(message, llm):
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

def save_audio_data(audio_data, sample_rate):
    """Save audio data to a WAV file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        # Convert audio data to int16 format
        audio_data_int = (audio_data * 32767).astype(np.int16)
        
        # Write to WAV file
        with wave.open(tmp_file.name, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data_int.tobytes())
        
        return tmp_file.name

def main():
    st.markdown("<h1 style='text-align: center;'>🔊 Audio Transaction Processor</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Voice recording section
    st.markdown("### 🎤 Record Your Transaction")
    
    # Use Streamlit's audio recorder
    audio_bytes = st.audio_recorder(text="Click to record", 
                                   recording_color="#e8b62c", 
                                   neutral_color="#6aa36f", 
                                   stop_recording_text="Click to stop recording")
    
    if audio_bytes:
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
                    result = whisper_model.transcribe(tmp_file_path)
                    transcription = result.get("text", "")
                
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                
                if not transcription:
                    st.error(" No transcription output. Please check your audio recording.")
                    return
                
                # Store transcription in session state
                st.session_state.transcription = transcription
                
            except Exception as e:
                st.error(f" An error occurred: {str(e)}")
    
    # If transcription exists, show editable text area
    if 'transcription' in st.session_state:
        st.markdown("### ✏️ Edit Transcription Before Processing")
        edited_transcription = st.text_area("", st.session_state.transcription, height=200)
        
        if st.button(' Process Transaction Details'):
            with st.spinner("🤖 Processing transaction details..."):
                processed_result = process_transaction_message(edited_transcription, rag_llm)
                if processed_result:
                    st.markdown("###  Extracted Transaction Details")
                    st.code(processed_result, language="json")
                else:
                    st.error(" Failed to process transaction details.")

if __name__ == "__main__":
    main()
