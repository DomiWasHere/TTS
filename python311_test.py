#!/usr/bin/env python3
"""
Test script for Python 3.11 with Coqui TTS
Run with: py -3.11 python311_test.py
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*")

def check_versions():
    """Check installed versions"""
    print(f"Python version: {sys.version}")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not installed")
        return False
    
    try:
        import tokenizers
        print(f"Tokenizers version: {tokenizers.__version__}")
    except ImportError:
        print("Tokenizers not installed")
        return False
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed")
        return False
    
    return True

def test_tts_import():
    """Test TTS import"""
    try:
        from TTS.api import TTS
        print("✓ TTS import successful")
        return True
    except Exception as e:
        print(f"✗ TTS import failed: {e}")
        return False

def test_model_loading():
    """Test XTTS model loading with error handling"""
    try:
        # Suppress the GPT2InferenceModel warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*")
            warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
            
            from TTS.api import TTS
            print("Loading XTTS v2 model...")
            
            # Try loading with explicit device
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            tts = tts.to(device)
            
            print(f"✓ Model loaded successfully on {device}")
            return tts
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        
        # Check if it's the specific generate error
        if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
            print("\nThis is the known compatibility issue.")
            print("Try these solutions:")
            print("1. pip install transformers==4.30.0")
            print("2. pip install transformers==4.35.0")
            print("3. Use a virtual environment with Python 3.10")
        
        return None

def test_voice_clone_simple(tts):
    """Simple voice cloning test"""
    # Look for your audio file
    possible_paths = [
        "voice_samples/shadow_slave_jessica.wav",
        "voice_samples/shadow_slave_jessica.mp3",
        "C:/Users/Domi/Desktop/TTS/voice_samples/shadow_slave_jessica.mp3",
        "C:/Users/Domi/Desktop/TTS/voice_samples/shadow_slave_jessica.wav"
    ]
    
    reference_file = None
    for path in possible_paths:
        if os.path.exists(path):
            reference_file = path
            break
    
    if not reference_file:
        print("✗ No reference audio file found")
        print("Please ensure your audio file exists at one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    
    print(f"Using reference file: {reference_file}")
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            tts.tts_to_file(
                text="This is a test with Python 3.11 and the latest setup.",
                speaker_wav=reference_file,
                language="en",
                file_path="test_python311.wav"
            )
        
        print("✓ Voice cloning test successful!")
        print("✓ Output saved as: test_python311.wav")
        return True
        
    except Exception as e:
        print(f"✗ Voice cloning failed: {e}")
        return False

def main():
    print("=== Python 3.11 Coqui TTS Test ===\n")
    
    # Check versions
    if not check_versions():
        print("Please install missing dependencies")
        return
    
    print("-" * 50)
    
    # Test TTS import
    if not test_tts_import():
        return
    
    # Test model loading
    tts = test_model_loading()
    if not tts:
        return
    
    # Test voice cloning
    success = test_voice_clone_simple(tts)
    
    if success:
        print("\n🎉 All tests passed! Ready for audiobook creation.")
    else:
        print("\n❌ Voice cloning test failed.")

if __name__ == "__main__":
    main()