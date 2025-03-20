import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import soundfile as sf
import os

st.title("🎙 Live Audio Recorder")

# Define WebRTC config
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Directory to save recordings
SAVE_DIR = "recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_data = frame.to_ndarray()
        self.frames.append(audio_data)
        return frame

# Create WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=False,  # Fix for async issue
    processor_factory=AudioProcessor
)

# Button to save recording
if webrtc_ctx and webrtc_ctx.state.playing:
    if st.button("🛑 Stop and Save Recording"):
        processor = webrtc_ctx.processor
        if processor and processor.frames:
            audio_np = np.concatenate(processor.frames, axis=0)
            save_path = os.path.join(SAVE_DIR, "recorded_audio.wav")
            sf.write(save_path, audio_np, samplerate=48000)
            st.success(f"✅ Audio saved at: {save_path}")
        else:
            st.warning("⚠️ No audio recorded!")
