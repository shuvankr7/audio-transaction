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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import wave
import os

st.title("🎙 Live Audio Recorder")

# Custom Audio Processor to Store Audio Frames
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_frames = []
        self.sample_rate = 44100  # Standard audio sample rate

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_data = frame.to_ndarray()
        self.audio_frames.append(audio_data)
        return frame

# Initialize WebRTC Audio Streamer
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
    audio_processor_factory=AudioProcessor,
)

# Stop & Save Button
if st.button("Stop and Save Recording"):
    if webrtc_ctx and webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.audio_frames:
        audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_frames, axis=0)

        # Save the recorded audio as a WAV file
        output_filename = "recorded_audio.wav"
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit PCM format
            wf.setframerate(44100)  # 44.1 kHz sample rate
            wf.writeframes(audio_data.astype(np.int16).tobytes())

        st.success(f"✅ Audio saved successfully as '{output_filename}'")
        st.audio(output_filename, format="audio/wav")  # Play saved audio
    else:
        st.warning("⚠️ No audio recorded! Please try again.")

# Debugging: Show message if WebRTC is not initialized
if webrtc_ctx is None:
    st.error("❌ WebRTC initialization failed! Try reloading the app.")

