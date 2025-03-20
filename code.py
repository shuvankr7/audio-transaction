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
import streamlit_webrtc as webrtc
import whisper
import av  # Required by streamlit-webrtc

def transcribe_audio(audio_bytes):
    """Transcribes audio using OpenAI Whisper."""
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    model = whisper.load_model("base")
    result = model.transcribe("temp_audio.wav")
    return result["text"]

def main():
    st.title("Live Audio Recording and Transcription")

    webrtc_streamer = webrtc.webrtc_streamer(
        key="audio-recorder",
        mode=webrtc.WebRtcMode.SENDRECV,
        audio_receiver_size=1024,  # Adjust buffer size as needed
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_streamer.audio_receiver:
        try:
            audio_frames = webrtc_streamer.audio_receiver.get_frames()
            if audio_frames:
                audio_bytes = b"".join([frame.to_ndarray().tobytes() for frame in audio_frames])

                # Transcribe the recorded audio
                transcription = transcribe_audio(audio_bytes)
                st.write("Transcription:")
                st.write(transcription)

        except av.error.BlockingIOError:
            # Handle potential blocking errors
            pass
        except Exception as e:
             st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
