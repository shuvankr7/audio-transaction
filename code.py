import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
import time

st.set_page_config(page_title="Audio Transaction Processor", page_icon="🎤", layout="wide")

# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        st.error("Whisper module not found. Please ensure it's installed correctly.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
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
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Load models on startup
whisper_model = load_whisper_model()
rag_llm = initialize_rag_system()

def process_transaction_message(message, llm):
    if llm is None:
        return "Error: RAG system is not initialized."
    system_prompt = (
        "Your input is a transaction message extracted from voice. Extract structured details like Amount, "
        "Transaction Type, Bank Name, Card Type, Paid To, Merchant, Transaction Mode, Transaction Date, "
        "Reference Number, and Tag. Tag means which category of spending, if Amazon then shopping, if Zomato then eating. "
        "Just return the JSON output. If there is no output, return 'null'. "
        "If mode of payment is not mentioned, assume cash by default. "
        "If any field is missing, set it as null. "
        "Return only a JSON or a list of JSON objects. "
        "As the input is from a human, it may be unstructured grammatically and simple. "
        "Example: 'Today I spent 500 at Domino's', you need to handle it carefully. "
        "If multiple items are mentioned, generate a list of JSON objects accordingly."
    )
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response.content if hasattr(response, 'content') else response

def main():
    st.markdown("<h1 style='text-align: center;'>🔊 Live Voice Transaction Processor</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Record live audio
    audio_bytes = st.audio_recorder()
    
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        with st.spinner("⏳ Transcribing audio... Please wait."):
            try:
                result = whisper_model.transcribe(tmp_file_path)
                transcription = result.get("text", "")
                os.unlink(tmp_file_path)
                
                if not transcription:
                    st.error("No transcription output. Please try again.")
                    return
                
                st.session_state.transcription = transcription
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # If transcription exists, show editable text area
    if 'transcription' in st.session_state:
        st.markdown("### ✏️ Edit Transcription Before Processing")
        edited_transcription = st.text_area("", st.session_state.transcription, height=200)
        
        if st.button('Process Transaction Details'):
            with st.spinner("🤖 Processing transaction details..."):
                processed_result = process_transaction_message(edited_transcription, rag_llm)
                if processed_result:
                    st.markdown("### Extracted Transaction Details")
                    st.code(processed_result, language="json")
                else:
                    st.error("Failed to process transaction details.")

if __name__ == "__main__":
    main()
