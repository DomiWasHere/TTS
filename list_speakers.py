from TTS.api import TTS

print("Loading XTTS v2 model...")
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')

print("Available speakers:")
if hasattr(tts, 'speakers') and tts.speakers:
    for i, speaker in enumerate(tts.speakers):
        print(f"  {i}: {speaker}")
else:
    print("  No built-in speakers found - this model requires voice cloning")

print("\nModel info:")
print(f"  Model name: {tts.model_name}")
print(f"  Is multi-speaker: {tts.is_multi_speaker}")
print(f"  Is multi-lingual: {tts.is_multi_lingual}")