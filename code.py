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

# WebRTC Configuration (STUN server added)
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Audio Processor Class
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_data = frame.to_ndarray()
        self.frames.append(audio_data)
        return frame

# Initialize WebRTC
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=False,
)

# Ensure processor exists before accessing it
if webrtc_ctx and webrtc_ctx.state.playing:
    processor = webrtc_ctx.processor
    if processor is None:
        processor = AudioProcessor()  # Fix: Create an instance of the processor if None
        webrtc_ctx.processor = processor

    # Stop and Save Button
    if st.button("🛑 Stop and Save Recording"):
        if processor.frames:
            audio_np = np.concatenate(processor.frames, axis=0)
            save_path = os.path.join(SAVE_DIR, "recorded_audio.wav")
            sf.write(save_path, audio_np, samplerate=48000)
            st.success(f"✅ Audio saved at: {save_path}")
        else:
            st.warning("⚠️ No audio recorded!")
