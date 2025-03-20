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



    
                       
import streamlit as st
import tempfile
import os
import sounddevice as sd
import wave
import io
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

def record_audio(duration=5, samplerate=44100, device=None):
    """Records audio from the user."""
    st.write("Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='int16', device=device)
    sd.wait()
    
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    
    return audio_buffer.getvalue()

def process_transaction_message(message, llm):
    if llm is None:
        return "Error: RAG system is not initialized."
    system_prompt = (
        "Your input is a transaction message extracted from voice. Extract structured details like Amount, Transaction Type, Bank Name, "
        "Card Type, Paid To, Merchant, Transaction Mode, Transaction Date, Reference Number, and Category Tag. "
        "If mode of payment is not mentioned, assume cash by default. If any field is missing, set it as null. "
        "Return only a JSON or a list of JSON objects. "
        "Example: 'Today I spent 500 at Domino's' should be categorized correctly."
    )
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response.content if hasattr(response, 'content') else response

def main():
    st.markdown("<h1 style='text-align: center;'>🎧 Audio Transaction Processor</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Record audio instead of file upload
    if st.button('🎤 Record Audio'):
        with st.spinner("Recording..."):
            audio_data = record_audio()
            tmp_file_path = "temp_audio.wav"
            with open(tmp_file_path, "wb") as f:
                f.write(audio_data)
        st.success("Recording complete!")
    
    # Process transcription
    if os.path.exists("temp_audio.wav"):
        st.markdown("**🎵 Audio Preview:**")
        st.audio("temp_audio.wav", format="audio/wav")
        
        if st.button('🎤 Transcribe Audio'):
            try:
                with st.spinner("⏳ Transcribing audio... Please wait."):
                    result = whisper_model.transcribe("temp_audio.wav")
                    transcription = result.get("text", "")
                
                if not transcription:
                    st.error("No transcription output. Please check your audio file.")
                    return
                
                st.session_state.transcription = transcription
                os.remove("temp_audio.wav")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
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
                    st.error("Failed to process transaction details.")

if __name__ == "__main__":
    main()
