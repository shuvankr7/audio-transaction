import streamlit as st
import torch
import os
import whisper
import time
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorFactory
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
        "Your input is a transaction message extracted from voice. Extract structured details like Amount, Transaction Type, Bank Name, Card Type, Paid to whom, Merchant, Transaction Mode, Transaction Date, Reference Number, and Tag."
        "Tag means the category of spending (e.g., if Amazon, then shopping; if Zomato, then eating)."
        "Just return JSON output, no additional text. If there is no valid data, return 'null'."
        "Assume cash payment if mode is not mentioned. Set missing fields to null."
        "Example 1: 'today I spent 500 at Domino’s' should be handled correctly."
        "For multiple items with multiple prices, return a list of JSON objects."
    )
    
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response.content if hasattr(response, 'content') else response

# Streamlit UI
st.title("Voice-Based Transaction Analyzer")
st.sidebar.header("Settings")

# WebRTC Audio Capture
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": False, "audio": True},
)

if webrtc_ctx.audio_receiver:
    st.write("Listening... Speak now!")
    audio_frames = webrtc_ctx.audio_receiver.get_frames()
    
    if st.button("Process Recording"):
        st.write("Processing Audio...")
        
        # Convert audio data to numpy array
        audio_data = np.concatenate([frame.to_ndarray() for frame in audio_frames], axis=0)
        
        # Save as WAV file for Whisper
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(audio_data.tobytes())
        
        whisper_model = load_whisper_model()
        if whisper_model:
            result = whisper_model.transcribe(temp_file_path)
            transcription = result.get("text", "").strip()
            
            if transcription:
                st.session_state.transcription = transcription
            else:
                st.error("No transcription output.")
        else:
            st.error("Whisper model failed to load.")

# Display transcription and allow edits
if "transcription" in st.session_state:
    st.subheader("Transcription")
    text_input = st.text_area("Edit the transcription if needed", value=st.session_state.transcription, key="edited_text")
    final_text = text_input.strip()
    
    # Process extracted text
    with st.status("Processing transaction details...", expanded=True) as status:
        rag_llm = initialize_rag_system()
        processed_result = process_transaction_message(final_text, rag_llm)
        st.session_state.final_output = processed_result
        status.update(label="Processing complete!", state="complete", expanded=False)

# Display extracted transaction details
if "final_output" in st.session_state:
    st.subheader("Extracted Transaction Details")
    st.json(st.session_state.final_output)
