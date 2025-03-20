import streamlit as st 
import torch
import os
import wave
import io
import whisper
import time
from langchain_groq import ChatGroq
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np

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

# WebRTC Audio Processing
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = io.BytesIO()
    
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_data = frame.to_ndarray()
        self.audio_buffer.write(audio_data.tobytes())
        return frame
    
    def get_audio_data(self):
        return self.audio_buffer.getvalue()

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

# WebRTC for voice recording
st.write("Press Start to record your voice.")
audio_processor = AudioProcessor()
webrtc_ctx = webrtc_streamer(
    key="speech_recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=lambda: audio_processor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx.audio_receiver:
    st.success("Recording complete!")
    audio_data = audio_processor.get_audio_data()
    
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
    text_input = st.text_area("Edit the transcription if needed (Auto-submits in 7s)", value=st.session_state.transcription, key="edited_text")

    # Timer for 7 seconds
    if not st.session_state.editing_done:
        start_time = time.time()
        go_pressed = st.button("Go")

        while time.time() - start_time < 7:
            if go_pressed:
                break  # User pressed Go before 7 seconds
            time.sleep(1)

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
