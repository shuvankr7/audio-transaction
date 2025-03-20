import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import soundfile as sf
import os
import time

# Title
st.title("🎙️ Live Voice Recorder with Streamlit")

# Directory to save audio files
SAVE_DIR = "recorded_audio"
os.makedirs(SAVE_DIR, exist_ok=True)

# Define callback class for audio recording
class AudioRecorder:
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        # Convert WebRTC audio frame to numpy array
        audio = frame.to_ndarray()
        self.audio_frames.append(audio)
        return frame

# Initialize audio recorder
audio_recorder = AudioRecorder()

# Streamlit WebRTC component
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": False, "audio": True},
    audio_receiver_size=1024,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True
)

# Check if recording is active
if webrtc_ctx.state.playing:
    st.write("🎤 Recording... Speak into your microphone!")

    # Save recorded audio when recording stops
    if st.button("⏹ Stop & Save Recording"):
        if len(audio_recorder.audio_frames) > 0:
            # Convert recorded frames to numpy array
            audio_data = np.concatenate(audio_recorder.audio_frames, axis=0)
            
            # Define audio file path
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join(SAVE_DIR, f"recording_{timestamp}.wav")
            
            # Save the recorded audio
            sf.write(file_path, audio_data, samplerate=48000)
            st.success(f"✅ Recording saved: `{file_path}`")
        else:
            st.error("⚠️ No audio recorded!")

