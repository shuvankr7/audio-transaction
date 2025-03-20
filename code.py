import streamlit as st
import torch
import os
import sounddevice as sd
import wave
import io
import whisper
import time
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

# Record Audio
def record_audio(duration=5, samplerate=44100):
    st.write("Recording... Speak now!")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='int16')
        sd.wait()
        
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

# Process Transaction Message
def process_transaction_message(message, llm):
    if llm is None:
        return "Error: RAG system is not initialized."
    
    system_prompt = (
        "Your input is a transaction message extracted from voice. Extract structured details like Amount, Transaction Type, Bank Name, Card Type, Paid To, Merchant, "
        "Transaction Mode, Transaction Date, Reference Number, and Category Tag. "
        "If mode of payment is not mentioned, assume cash by default. "
        "Return only a JSON object. "
        "Example: '100 on KFC' -> {\"Amount\": 100, \"Merchant\": \"KFC\", \"Transaction Mode\": \"Cash\"}"
    )
    
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response.content if hasattr(response, 'content') else response

# Streamlit UI
st.title("Voice-Based Transaction Analyzer")
st.sidebar.header("Settings")
duration = st.sidebar.slider("Recording Duration (seconds)", 3, 10, 5)

# Initialize session state for transcription and timer
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "editing_done" not in st.session_state:
    st.session_state.editing_done = False
if "final_output" not in st.session_state:
    st.session_state.final_output = None  # Store final extracted JSON

if st.button("Start Recording"):
    audio_data = record_audio(duration)
    if audio_data:
        st.success("Recording complete!")
        
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(audio_data)

        st.write("Transcribing audio...")
        whisper_model = load_whisper_model()
        
        if whisper_model:
            result = whisper_model.transcribe(temp_file_path)
            st.session_state.transcription = result.get("text", "").strip()
            st.session_state.editing_done = False  # Reset editing state
            st.session_state.final_output = None  # Reset final output
            
            if not st.session_state.transcription:
                st.error("No transcription output.")
        else:
            st.error("Whisper model failed to load.")

# If transcription exists, show editing area
if st.session_state.transcription:
    st.subheader("Transcription")
    text_input = st.text_area("Edit the transcription if needed", value=st.session_state.transcription, key="edited_text")

    # Dynamic countdown timer
    countdown_placeholder = st.empty()
    go_pressed = st.button("Go")

    if not st.session_state.editing_done:
        for i in range(7, -1, -1):  # Countdown from 7 to 0
            countdown_placeholder.write(f"**Auto-submitting in {i} seconds...**")
            time.sleep(1)
            if go_pressed:
                break  # Stop countdown if "Go" is pressed

        st.session_state.editing_done = True  # Mark editing as done

    final_text = st.session_state.edited_text
    st.write(f"Final Input: {final_text}")

    # Show "Loading..." before processing
    with st.status("Processing transaction details...", expanded=True) as status:
        rag_llm = initialize_rag_system()
        processed_result = process_transaction_message(final_text, rag_llm)
        st.session_state.final_output = processed_result  # Store result in session
        status.update(label="Processing complete!", state="complete", expanded=False)

# Display the final extracted transaction details
if st.session_state.final_output:
    st.subheader("Extracted Transaction Details")
    st.json(st.session_state.final_output)
