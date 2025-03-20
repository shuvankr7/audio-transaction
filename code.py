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
import sounddevice as sd
import numpy as np
import wavio as wv
import tempfile

def record_audio(duration, fs=44100):
    """Records audio from the microphone."""
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        return recording, fs
    except sd.PortAudioError as e:
        st.error(f"PortAudio error: {e}")
        return None, None

def save_audio(recording, fs, filename):
    """Saves the recorded audio to a WAV file."""
    wv.write(filename, recording, fs)

def main():
    st.title("Audio Recorder")

    duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

    if st.button("Start Recording"):
        recording, fs = record_audio(duration)
        if recording is not None:
            st.success("Recording complete!")

            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_filename = temp_audio.name
                save_audio(recording, fs, temp_filename)

            # Display audio player and download button
            st.audio(temp_filename, format="audio/wav")
            with open(temp_filename, "rb") as f:
                st.download_button(
                    label="Download Recording",
                    data=f,
                    file_name="recorded_audio.wav",
                    mime="audio/wav",
                )
        else:
            st.error("Recording failed.")

if __name__ == "__main__":
    main()
