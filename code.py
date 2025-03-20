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

# Streamlit WebRTC component
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True
)

# Check if recording is active
if webrtc_ctx.state.playing:
    st.write("🎤 Recording... Speak into your microphone!")

    # Stop & Save button
    if st.button("⏹ Stop & Save Recording"):
        if webrtc_ctx.audio_receiver:
            try:
                # Collect audio data from receiver
                audio_frames = []
                while True:
                    frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                    audio_frames.append(frame.to_ndarray())

            except Exception as e:
                pass  # This will stop when there are no more frames

            if len(audio_frames) > 0:
                # Convert recorded frames to numpy array
                audio_data = np.concatenate(audio_frames, axis=0)
                
                # Define audio file path
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_path = os.path.join(SAVE_DIR, f"recording_{timestamp}.wav")
                
                # Save the recorded audio
                sf.write(file_path, audio_data, samplerate=48000)
                st.success(f"✅ Recording saved: `{file_path}`")
            else:
                st.error("⚠️ No audio recorded!")
        else:
            st.error("⚠️ No audio receiver available!")

