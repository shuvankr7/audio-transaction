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
import streamlit.components.v1 as components

# Streamlit app
st.title("Live Audio Recorder (Browser-Based)")

# Recording duration input
duration = st.slider("Select recording duration (seconds)", 
                     min_value=1, 
                     max_value=30, 
                     value=5)

# HTML and JavaScript code for audio recording
audio_recorder_html = f"""
<script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let recordingTimeout;

    // Function to start recording
    function startRecording() {{
        if (isRecording) return;
        
        navigator.mediaDevices.getUserMedia({{ audio: true }})
            .then(stream => {{
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {{
                    audioChunks.push(event.data);
                }};
                
                mediaRecorder.onstop = () => {{
                    const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    // Create a download link
                    const link = document.createElement('a');
                    link.href = audioUrl;
                    link.download = 'recorded_audio.wav';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    // Stop all tracks to release the microphone
                    stream.getTracks().forEach(track => track.stop());
                }};
                
                mediaRecorder.start();
                isRecording = true;
                document.getElementById('status').innerText = 'Recording...';
                
                // Stop recording after the specified duration
                recordingTimeout = setTimeout(() => {{
                    stopRecording();
                }}, {duration * 1000});
            }})
            .catch(err => {{
                document.getElementById('status').innerText = 'Error: ' + err.message;
            }});
    }}

    // Function to stop recording
    function stopRecording() {{
        if (!isRecording) return;
        
        mediaRecorder.stop();
        isRecording = false;
        clearTimeout(recordingTimeout);
        document.getElementById('status').innerText = 'Recording completed! File downloaded.';
    }}
</script>

<button onclick="startRecording()">Start Recording</button>
<p id="status">Click the button to start recording.</p>
"""

# Render the HTML/JS component
if st.button("Show Recording Interface"):
    components.html(audio_recorder_html, height=150)

# Instructions
st.write("""
Instructions:
1. Select desired recording duration using the slider.
2. Click 'Show Recording Interface' to display the recording button.
3. Click 'Start Recording' to begin.
4. Speak into your microphone.
5. Wait for the recording to complete (it will stop automatically after the selected duration).
6. The recording will be downloaded automatically as 'recorded_audio.wav'.
""")
