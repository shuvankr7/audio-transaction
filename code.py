import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import soundfile as sf
import os

# Title
st.title("🎙️ Live Voice Recorder")

# Function to process audio frames
def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
    audio = frame.to_ndarray()
    audio_data.append(audio)
    return frame

# Storage for audio data
audio_data = []

# Streamlit WebRTC streamer for audio recording
webrtc_ctx = webrtc_streamer(
    key="record_audio",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": False, "audio": True},
    audio_receiver_size=1024,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

# Save and Download
if st.button("Stop and Save Recording"):
    if len(audio_data) > 0:
        # Convert recorded data to a NumPy array
        recorded_audio = np.concatenate(audio_data, axis=0)

        # Save as a WAV file
        output_file = "recorded_audio.wav"
        sf.write(output_file, recorded_audio, samplerate=48000)

        # Provide download link
        with open(output_file, "rb") as f:
            st.download_button("Download Recording", f, file_name="recorded_audio.wav", mime="audio/wav")

        # Clean up the file
        os.remove(output_file)
        st.success("Recording saved and ready for download!")
    else:
        st.error("No audio recorded yet. Please try again.")

