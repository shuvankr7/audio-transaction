import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
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

# Class to process audio frames
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frame."""
        audio_array = frame.to_ndarray()
        self.audio_frames.append(audio_array)
        return frame  # Return the frame so WebRTC can process it

# Instantiate Audio Processor
audio_processor = AudioProcessor()

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    async_processing=True,
    processor_factory=lambda: audio_processor,
)

# Show status message
if webrtc_ctx.state.playing:
    st.write("🎤 **Recording... Speak into your microphone!**")

# Stop and Save Button
if st.button("⏹ Stop & Save Recording"):
    if len(audio_processor.audio_frames) > 0:
        # Convert recorded frames to numpy array
        audio_data = np.concatenate(audio_processor.audio_frames, axis=0)

        # Define audio file path
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(SAVE_DIR, f"recording_{timestamp}.wav")

        # Save the recorded audio
        sf.write(file_path, audio_data, samplerate=48000)
        st.success(f"✅ **Recording saved:** `{file_path}`")
    else:
        st.error("⚠️ **No audio recorded!** Make sure the microphone is capturing sound.")
