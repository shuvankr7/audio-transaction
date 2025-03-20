import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import wave
import os

# Title
st.title("🎙 Live Audio Recorder")

# Function to process audio frames
def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
    audio = frame.to_ndarray()
    return av.AudioFrame.from_ndarray(audio, format="s16")

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,  # Fixes 'No audio receiver available' issue
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)

# Initialize session state for storing audio
if "audio_data" not in st.session_state:
    st.session_state["audio_data"] = []

# Record Button
if webrtc_ctx and webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    for frame in audio_frames:
        st.session_state["audio_data"].append(frame.to_ndarray())

# Stop & Save Button
if st.button("Stop and Save Recording"):
    if len(st.session_state["audio_data"]) > 0:
        audio_data = np.concatenate(st.session_state["audio_data"], axis=0)

        # Save as a WAV file
        output_filename = "recorded_audio.wav"
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(44100)
            wf.writeframes(audio_data.tobytes())

        st.success(f"✅ Audio saved as {output_filename}")

        # Clear recorded audio data
        st.session_state["audio_data"] = []
    else:
        st.warning("⚠️ No audio recorded!")

# Play the saved audio file if exists
if os.path.exists("recorded_audio.wav"):
    st.audio("recorded_audio.wav", format="audio/wav")
