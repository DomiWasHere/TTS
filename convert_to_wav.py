#!/usr/bin/env python3
"""
Convert MP3 to WAV for better XTTS compatibility
Optimized for Python 3.11
"""
import os
import librosa
import soundfile as sf
from pydub import AudioSegment

def convert_with_librosa(input_path, output_path):
    """Convert using librosa (recommended for TTS)"""
    try:
        print(f"Converting {input_path} with librosa...")
        # Load audio at 22050 Hz (XTTS preferred sample rate)
        audio, sr = librosa.load(input_path, sr=22050)
        
        # Save as WAV
        sf.write(output_path, audio, sr)
        print(f"✓ Conversion successful: {output_path}")
        
        # Check file info
        duration = len(audio) / sr
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Channels: 1 (mono)")
        
        return True
    except Exception as e:
        print(f"✗ Librosa conversion failed: {e}")
        return False

def convert_with_pydub(input_path, output_path):
    """Fallback conversion using pydub"""
    try:
        print(f"Converting {input_path} with pydub...")
        audio = AudioSegment.from_mp3(input_path)
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(22050)  # 22050 Hz
        
        # Export as WAV
        audio.export(output_path, format="wav")
        print(f"✓ Conversion successful: {output_path}")
        
        # Check file info
        duration = len(audio) / 1000.0  # pydub uses milliseconds
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: 22050 Hz")
        print(f"  Channels: 1 (mono)")
        
        return True
    except Exception as e:
        print(f"✗ Pydub conversion failed: {e}")
        return False

def main():
    """Convert your MP3 files to WAV"""
    
    # Your file paths
    input_files = [
        "voice_samples/shadow_slave_jessica.mp3",
        "C:/Users/Domi/Desktop/TTS/voice_samples/shadow_slave_jessica.mp3"
    ]
    
    for input_path in input_files:
        if os.path.exists(input_path):
            print(f"Found input file: {input_path}")
            
            # Create output path
            output_path = input_path.replace('.mp3', '.wav')
            
            # Try librosa first (better for TTS)
            if convert_with_librosa(input_path, output_path):
                print(f"✓ Ready for TTS: {output_path}")
                break
            else:
                # Fallback to pydub
                if convert_with_pydub(input_path, output_path):
                    print(f"✓ Ready for TTS: {output_path}")
                    break
                else:
                    print(f"✗ Both conversion methods failed for {input_path}")
        else:
            print(f"File not found: {input_path}")
    
    print("\nConversion complete!")

if __name__ == "__main__":
    main()