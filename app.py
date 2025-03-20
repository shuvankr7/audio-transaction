import streamlit as st
import os
import whisper
import time
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from langchain_groq import ChatGroq

# Set environment variables
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# API Key and Model Config
GROQ_API_KEY = "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"
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1024

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    try:
        model = whisper.load_model("base", device="cpu")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

# Initialize RAG System
@st.cache_resource
def initialize_rag_system():
    try:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

# Process Transaction Message
def process_transaction_message(message, llm):
    if llm is None:
        return "Error: RAG system is not initialized."
    
    system_prompt = (
        "Your input is a transaction message extracted from voice. Extract structured details like Amount, "
        "Transaction Type, Bank Name, Card Type, Paid to whom, Merchant, Transaction Mode, Transaction Date, "
        "Reference Number, and Category Tag."
        "Category Tag means spending category: if Amazon then Shopping, if Zomato then Eating, etc."
        "Return JSON output only. If the mode of payment is not mentioned, assume cash by default."
        "If any field is missing, set it as null."
        "Example: 'I spent 500 at Domino’s' should be parsed correctly."
    )
    
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response.content if hasattr(response, 'content') else response

# Streamlit UI
st.title("🎙️ Voice-Based Transaction Analyzer")

# WebRTC Audio Recorder
st.subheader("Record Your Transaction")
webrtc_ctx = webrtc_streamer(
    key="record-audio",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"video": False, "audio": True},
)

# When recording stops
if st.button("Stop Recording"):
    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if audio_frames:
            st.success("Recording complete!")

            # Save the recorded audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(b"".join(audio_frames[0].to_ndarray().tobytes()))
                temp_audio_path = tmpfile.name
            
            st.write("Transcribing audio...")
            whisper_model = load_whisper_model()

            if whisper_model:
                result = whisper_model.transcribe(temp_audio_path)
                transcription = result.get("text", "").strip()

                if transcription:
                    st.subheader("Transcription (Edit if needed)")
                    edited_text = st.text_area("Edit before processing:", transcription)
                    
                    # Auto-submit countdown (7s)
                    countdown = st.empty()
                    start_time = time.time()
                    while time.time() - start_time < 7:
                        countdown.markdown(f"**Auto-submitting in {7 - int(time.time() - start_time)} seconds...**")
                        time.sleep(1)
                    
                    # Submit to Groq
                    st.subheader("Processing...")
                    rag_llm = initialize_rag_system()
                    processed_result = process_transaction_message(edited_text, rag_llm)

                    # Show final result
                    st.subheader("Extracted Transaction Details")
                    st.json(processed_result)
                else:
                    st.error("No transcription output.")
            else:
                st.error("Whisper model failed to load.")
        else:
            st.error("No audio recorded.")
    else:
        st.error("No audio receiver detected.")
