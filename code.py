# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
# import av
# import numpy as np
# import wave
# import os

# # Title
# st.title("🎙 Live Audio Recorder")

# # Custom Audio Processor Class
# class AudioProcessor(AudioProcessorBase):
#     def __init__(self) -> None:
#         self.audio_frames = []

#     def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#         audio_data = frame.to_ndarray()
#         self.audio_frames.append(audio_data)
#         return frame

# # WebRTC Streamer
# webrtc_ctx = webrtc_streamer(
#     key="audio_recorder",
#     mode=WebRtcMode.SENDRECV,
#     media_stream_constraints={"video": False, "audio": True},
#     async_processing=True,
#     audio_processor_factory=AudioProcessor,
# )

# # Stop & Save Button
# if st.button("Stop and Save Recording"):
#     if webrtc_ctx and webrtc_ctx.audio_processor:
#         audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_frames, axis=0)

#         # Save as a WAV file
#         output_filename = "recorded_audio.wav"
#         with wave.open(output_filename, "wb") as wf:
#             wf.setnchannels(1)
#             wf.setsampwidth(2)  # 16-bit PCM
#             wf.setframerate(44100)
#             wf.writeframes(audio_data.tobytes())

#         st.success(f"✅ Audio saved as {output_filename}")
#     else:
#         st.warning("⚠️ No audio recorded!")

# # Play the saved audio file if exists
# if os.path.exists("recorded_audio.wav"):
#     st.audio("recorded_audio.wav", format="audio/wav")



    
                       
import torch
import streamlit as st
import tempfile
import os
import sounddevice as sd
import wave
import io
from langchain_groq import ChatGroq
import time
import sounddevice as sd
st.title("Audio Device Check")

try:
    devices = sd.query_devices()
    st.write("Detected audio devices:", devices)
except Exception as e:
    st.error(f"Error accessing audio devices: {str(e)}")


# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default Groq API key (Ensure this is kept secure)
GROQ_API_KEY = "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"
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1024

# Load whisper model at startup - important to import here, not at the top of the file
@st.cache_resource
def load_whisper_model():
    try:
        # Import whisper inside the function to avoid circular imports
        import whisper
        return whisper.load_model("base")
    except ImportError:
        st.error("OpenAI Whisper module not found. Install it using: pip install git+https://github.com/openai/whisper.git")
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

# Record audio function
def record_audio(duration=5, samplerate=44100, device=None):
    """Records audio from the user."""
    st.write("Recording... Speak now!")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='int16', device=device)
        sd.wait()
        
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

def process_transaction_message(message, llm):
    if llm is None:
        return "Error: RAG system is not initialized."
    system_prompt = (
        "Your input is a transaction message extracted from voice. Extract structured details like "
        "Amount, Transaction Type, Bank Name, Card Type, Paid To, Merchant, Transaction Mode, Transaction Date, Reference Number, and Category Tag. "
        "Tag meaning which category of spending, if Amazon then shopping etc, if Zomato then food. "
        "If mode of payment is not mentioned, assume cash by default. "
        "If any field is missing, set it as null. "
        "Return only a JSON or a list of JSON objects. "
        "As input comes from human speech, it may be grammatically simple or incomplete. "
        "Example: 'today I spent 500 at dominoes' should be processed correctly. "
        "If user mentions multiple items with corresponding prices, generate a list of JSON objects accordingly."
    )
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response.content if hasattr(response, 'content') else response

def get_available_devices():
    """Returns a list of available audio input devices"""
    devices = sd.query_devices()
    input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    return input_devices, devices

def main():
    st.set_page_config(page_title="Audio Transaction Processor", page_icon="🎤", layout="wide")
    
    st.markdown("<h1 style='text-align: center;'>🎧 Audio Transaction Processor</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models on startup - moved inside main() to avoid circular imports
    with st.spinner("Loading models..."):
        whisper_model = load_whisper_model()
        rag_llm = initialize_rag_system()
    
    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
    
    with tab1:
        # Get available input devices
        input_devices, devices_info = get_available_devices()
        
        # Display device selection if devices are available
        if input_devices:
            device_names = [f"Device {i}: {devices_info[i]['name']}" for i in input_devices]
            selected_device = st.selectbox("Select input device:", device_names)
            device_index = input_devices[device_names.index(selected_device)]
            
            duration = st.slider("Recording duration (seconds):", 1, 10, 5)
            
            if st.button('🎤 Record Audio'):
                with st.spinner("Recording..."):
                    audio_data = record_audio(duration=duration, device=device_index)
                    if audio_data:
                        tmp_file_path = "temp_audio.wav"
                        with open(tmp_file_path, "wb") as f:
                            f.write(audio_data)
                        st.success("Recording complete!")
        else:
            st.error("No audio input devices found.")
    
    with tab2:
        uploaded_file = st.file_uploader("📂 Upload an audio file", type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'])
        
        if uploaded_file is not None:
            st.markdown("**🎵 Audio Preview:**")
            st.audio(uploaded_file, format="audio/*")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
                # Store the path for processing
                st.session_state.audio_file_path = tmp_file_path
    
    # Process either recorded or uploaded audio
    audio_file_path = None
    
    if os.path.exists("temp_audio.wav"):
        st.markdown("**🎵 Audio Preview:**")
        st.audio("temp_audio.wav", format="audio/wav")
        audio_file_path = "temp_audio.wav"
    elif 'audio_file_path' in st.session_state:
        audio_file_path = st.session_state.audio_file_path
    
    if audio_file_path and st.button('🎤 Transcribe Audio'):
        try:
            with st.spinner("⏳ Transcribing audio... Please wait."):
                result = whisper_model.transcribe(audio_file_path)
                transcription = result.get("text", "")
            
            if not transcription:
                st.error("No transcription output. Please check your audio file.")
            else:
                st.session_state.transcription = transcription
                
                # Clean up files after processing
                if audio_file_path == "temp_audio.wav" and os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                elif 'audio_file_path' in st.session_state:
                    os.unlink(st.session_state.audio_file_path)
                    del st.session_state.audio_file_path
                    
        except Exception as e:
            st.error(f"An error occurred during transcription: {str(e)}")
    
    # If transcription exists, show editable text area
    if 'transcription' in st.session_state:
        st.markdown("### ✏️ Edit Transcription Before Processing")
        edited_transcription = st.text_area("", st.session_state.transcription, height=200)
        
        if st.button('💼 Process Transaction Details'):
            with st.spinner("🤖 Processing transaction details..."):
                processed_result = process_transaction_message(edited_transcription, rag_llm)
                if processed_result:
                    st.markdown("### 📊 Extracted Transaction Details")
                    st.code(processed_result, language="json")
                else:
                    st.error("Failed to process transaction details.")

if __name__ == "__main__":
    main()
