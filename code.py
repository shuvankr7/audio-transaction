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
import pyaudio
import wave
import os

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OUTPUT_FILENAME = "recorded_audio.wav"

def record_audio(duration):
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    st.write("Recording...")
    frames = []
    
    # Record for specified duration
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded audio
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return OUTPUT_FILENAME

def main():
    st.title("Live Audio Recorder")
    
    # Recording duration input
    duration = st.slider("Select recording duration (seconds)", 
                        min_value=1, 
                        max_value=30, 
                        value=5)
    
    # Record button
    if st.button("Start Recording"):
        with st.spinner("Recording audio..."):
            audio_file = record_audio(duration)
        
        st.success("Recording completed!")
        
        # Download button
        with open(audio_file, 'rb') as file:
            st.download_button(
                label="Download Recording",
                data=file,
                file_name=OUTPUT_FILENAME,
                mime="audio/wav"
            )
    
    # Instructions
    st.write("""
    Instructions:
    1. Select desired recording duration using the slider
    2. Click 'Start Recording' to begin
    3. Speak into your microphone
    4. Wait for recording to complete
    5. Download the recording
    """)

if __name__ == "__main__":
    main()
