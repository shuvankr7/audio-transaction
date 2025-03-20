import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import soundfile as sf
import os

st.title("🎙 Live Audio Recorder")

# Ensure a directory exists for storing recordings
SAVE_DIR = "recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

# WebRTC Configuration with STUN server for better connectivity
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# Define the audio frame processor
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_data = frame.to_ndarray()
        self.frames.append(audio_data)
        return frame


# Create an instance of the processor
audio_processor = AudioProcessor()


# Function to process incoming audio
def process_audio(frame: av.AudioFrame) -> av.AudioFrame:
    return audio_processor.recv(frame)


# Initialize WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": False, "audio": True},
    audio_frame_callback=process_audio,  # Fix: Using audio frame callback
)

# Save Recording Button
if st.button("🛑 Stop and Save Recording"):
    if audio_processor.frames:
        audio_np = np.concatenate(audio_processor.frames, axis=0)
        save_path = os.path.join(SAVE_DIR, "recorded_audio.wav")
        sf.write(save_path, audio_np, samplerate=48000)
        st.success(f"✅ Audio saved at: {save_path}")
    else:
        st.warning("⚠️ No audio recorded!")
