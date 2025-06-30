from TTS.api import TTS

print("Loading XTTS v2 model...")
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')

# Replace with your actual reference file path
reference_voice = "voice_samples\shadow_slave_jessica.mp3"

print("Testing voice cloning...")
tts.tts_to_file(
    text="Hello, this is a test of voice cloning for my audiobook project. The voice should sound natural and expressive.",
    speaker_wav=reference_voice,
    language="en",
    file_path="voice_clone_test.wav"
)

print("Voice cloning test completed: voice_clone_test.wav")