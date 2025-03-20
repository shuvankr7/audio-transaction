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
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

# Import webrtc-related components
def setup_audio_recorder():
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
        import av
        
        # Use session state to store audio frames
        if 'audio_frames' not in st.session_state:
            st.session_state.audio_frames = []
        
        def video_frame_callback(frame):
            # Return unchanged video frame
            return frame
        
        def audio_frame_callback(frame):
            # Store audio frames in session state
            if frame is not None:
                sound = frame.to_ndarray()
                st.session_state.audio_frames.append(sound)
            return frame
        
        # Setup webrtc with audio only
        ctx = webrtc_streamer(
            key="voice-recorder",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            video_frame_callback=video_frame_callback,
            audio_frame_callback=audio_frame_callback,
            async_processing=True,
        )
        
        return ctx
    except ImportError:
        st.error("❌ streamlit-webrtc not installed. Please install with 'pip install streamlit-webrtc'")
        return None

def save_audio_frames_to_file():
    """Save collected audio frames to a WAV file"""
    if 'audio_frames' not in st.session_state or not st.session_state.audio_frames:
        return None
    
    try:
        import soundfile as sf
        import numpy as np
        
        # Concatenate all audio frames
        audio_data = np.concatenate(st.session_state.audio_frames, axis=0)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Save to WAV
        sf.write(temp_file.name, audio_data, 48000)  # Most webrtc audio is 48kHz
        
        return temp_file.name
    except Exception as e:
        st.error(f"❌ Error saving audio: {str(e)}")
        return None

# Initialize session state
init_session_state()

# Main app layout
st.markdown("<h1 style='text-align: center;'>🔊 Audio Transaction Processor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for loading models
with st.sidebar:
    st.header("Model Configuration")
    
    if st.button("Load Whisper Model", key="load_whisper"):
        with st.spinner("Loading Whisper model..."):
            model, error = load_whisper_model()
            if model:
                st.session_state.whisper_model = model
                st.success("✅ Whisper model loaded successfully!")
            else:
                st.error(f"❌ {error}")
    
    if st.button("Initialize LLM", key="init_llm"):
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

# Voice recording section with webrtc
st.markdown("Click the 'START' button below to begin recording your transaction:")

# Get the webrtc context
webrtc_ctx = setup_audio_recorder()

# Add a save button below the webrtc component
if webrtc_ctx is not None:
    if webrtc_ctx.state.playing and len(st.session_state.get('audio_frames', [])) > 0:
        st.info("🔴 Recording in progress... Speak your transaction details and then click 'Stop' followed by 'Save Recording'")
    
    if not webrtc_ctx.state.playing and len(st.session_state.get('audio_frames', [])) > 0:
        if st.button("Save Recording", key="save_recording"):
            with st.spinner("Saving audio..."):
                audio_file = save_audio_frames_to_file()
                if audio_file:
                    st.session_state.audio_file = audio_file
                    st.success("✅ Recording saved successfully!")

# If audio is saved, show transcribe button
if 'audio_file' in st.session_state and st.session_state.audio_file and models_ready:
    st.markdown("**🎵 Audio recorded successfully!**")
    
    if st.button('🎤 Transcribe Audio', key="transcribe_audio"):
        try:
            with st.spinner("⏳ Transcribing audio... Please wait."):
                # Transcribe using Whisper
                result = st.session_state.whisper_model.transcribe(st.session_state.audio_file)
                transcription = result.get("text", "")
            
            if not transcription:
                st.error("❌ No transcription output. Please check your audio recording.")
            else:
                # Store transcription in session state
                st.session_state.transcription = transcription
                st.success("✅ Transcription successful!")
                
        except Exception as e:
            st.error(f"❌ An error occurred during transcription: {str(e)}")

# If transcription exists, show editable text area
if 'transcription' in st.session_state:
    st.markdown("### ✏️ Edit Transcription Before Processing")
    edited_transcription = st.text_area("", st.session_state.transcription, height=200)
    
    if st.button('💼 Process Transaction Details', key="process_transaction"):
        with st.spinner("🤖 Processing transaction details..."):
            processed_result = process_transaction_message(edited_transcription, st.session_state.rag_llm)
            if processed_result:
                st.markdown("### 📊 Extracted Transaction Details")
                st.code(processed_result, language="json")
            else:
                st.error("❌ Failed to process transaction details.")
