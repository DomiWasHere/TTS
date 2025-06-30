#!/usr/bin/env python3
"""
High-Quality Audiobook Creator for Python 3.11
Run with: py -3.11 audiobook_creator_py311.py
"""
import os
import re
import torch
import warnings
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize
import nltk

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")
warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*")

# Download required NLTK data - Updated for newer NLTK versions
def download_nltk_data():
    """Download required NLTK data with fallback for different versions"""
    tokenizers_to_try = ['punkt_tab', 'punkt']
    
    for tokenizer in tokenizers_to_try:
        try:
            nltk.data.find(f'tokenizers/{tokenizer}')
            print(f"✅ NLTK tokenizer '{tokenizer}' found")
            return tokenizer
        except LookupError:
            print(f"📥 Downloading NLTK {tokenizer} tokenizer...")
            try:
                nltk.download(tokenizer, quiet=True)
                print(f"✅ Successfully downloaded {tokenizer}")
                return tokenizer
            except Exception as e:
                print(f"⚠️ Failed to download {tokenizer}: {e}")
                continue
    
    raise RuntimeError("Could not download any NLTK tokenizer")

# Initialize NLTK
current_tokenizer = download_nltk_data()

class AudiobookCreator:
    def __init__(self, reference_voice_path):
        self.reference_voice = reference_voice_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎙️  Initializing AudiobookCreator on {self.device}")
        
        # Verify reference file exists
        if not os.path.exists(reference_voice_path):
            raise FileNotFoundError(f"Reference voice file not found: {reference_voice_path}")
        
        # Initialize TTS with error suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("📥 Loading XTTS v2 model...")
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            print("✅ Model loaded successfully!")
    
    def preprocess_text_for_emotion(self, text):
        """Enhanced text preprocessing for emotional storytelling"""
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Add emotional pauses and emphasis
        text = re.sub(r'\.(?=\s[A-Z])', '... ', text)  # Longer pause after sentences
        text = re.sub(r'!(?=\s)', '! ', text)  # Emphasis after exclamations
        text = re.sub(r'\?(?=\s)', '? ', text)  # Pause after questions
        text = re.sub(r',(?=\s)', ', ', text)  # Brief pause after commas
        text = re.sub(r';(?=\s)', '; ', text)  # Medium pause after semicolons
        text = re.sub(r':(?=\s)', ': ', text)  # Medium pause after colons
        
        # Handle dialogue with better expression
        text = re.sub(r'"([^"]*)"', r'"\1"', text)
        
        # Add emphasis to certain words for better storytelling
        text = re.sub(r'\b(suddenly|whispered|screamed|gasped|breathed)\b', r'\1...', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def split_into_optimal_chunks(self, text, max_chars=800):
        """Split text into chunks optimized for TTS quality"""
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            print(f"⚠️ NLTK tokenization failed: {e}")
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            if len(current_chunk + sentence) <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single sentence is too long, split by clauses
                if len(sentence) > max_chars:
                    clauses = re.split(r'[,;:]', sentence)
                    temp_chunk = ""
                    for clause in clauses:
                        if len(temp_chunk + clause) <= max_chars:
                            temp_chunk += clause + ", "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = clause + ", "
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_audio_with_settings(self, text, output_path, speed=0.95, temperature=0.7):
        """Create audio with enhanced settings for storytelling"""
        try:
            processed_text = self.preprocess_text_for_emotion(text)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try with enhanced settings first
                try:
                    self.tts.tts_to_file(
                        text=processed_text,
                        speaker_wav=self.reference_voice,
                        language="en",
                        file_path=output_path,
                        split_sentences=True,
                        speed=speed,
                        temperature=temperature  # Controls expressiveness
                    )
                    return True
                except TypeError:
                    # If temperature parameter is not supported, try without it
                    self.tts.tts_to_file(
                        text=processed_text,
                        speaker_wav=self.reference_voice,
                        language="en",
                        file_path=output_path,
                        split_sentences=True,
                        speed=speed
                    )
                    return True
                    
        except Exception as e:
            # Try with basic settings if enhanced settings fail
            try:
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=self.reference_voice,
                    language="en",
                    file_path=output_path,
                    split_sentences=True
                )
                return True
            except Exception as e2:
                print(f"❌ Error creating audio: {e2}")
                return False
    
    def create_chapter_with_quality(self, chapter_text, chapter_name, output_dir="audiobook_output", 
                                  speed=0.95, add_chapter_intro=True):
        """Create a high-quality chapter with intro and proper pacing"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📖 Creating chapter: {chapter_name}")
        
        # Optional chapter introduction
        if add_chapter_intro:
            intro_text = f"Chapter {chapter_name.replace('_', ' ')}"
            intro_path = os.path.join(output_dir, f"{chapter_name}_intro.wav")
            
            print("🎭 Creating chapter introduction...")
            if self.create_audio_with_settings(intro_text, intro_path, speed=0.8):
                try:
                    intro_audio = AudioSegment.from_wav(intro_path)
                    # Add silence after intro
                    intro_audio += AudioSegment.silent(duration=1500)  # 1.5 second pause
                except Exception as e:
                    print(f"⚠️ Could not load intro audio: {e}")
                    intro_audio = None
            else:
                intro_audio = None
        else:
            intro_audio = None
        
        # Split chapter into chunks
        chunks = self.split_into_optimal_chunks(chapter_text)
        print(f"📝 Processing {len(chunks)} chunks...")
        
        # Create audio for each chunk
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(output_dir, f"{chapter_name}_chunk_{i:03d}.wav")
            
            print(f"🎵 Processing chunk {i+1}/{len(chunks)}...")
            if self.create_audio_with_settings(chunk, chunk_file, speed):
                chunk_files.append(chunk_file)
            else:
                print(f"⚠️  Failed to create chunk {i+1}, skipping...")
        
        if not chunk_files:
            print(f"❌ No chunks created for {chapter_name}")
            return False
        
        # Combine all audio
        final_path = os.path.join(output_dir, f"{chapter_name}.wav")
        if self.combine_chapter_audio(chunk_files, final_path, intro_audio):
            # Clean up temporary files
            for chunk_file in chunk_files:
                try:
                    os.remove(chunk_file)
                except:
                    pass
            
            if intro_audio and add_chapter_intro:
                try:
                    os.remove(intro_path)
                except:
                    pass
            
            print(f"✅ Chapter completed: {final_path}")
            return True
        
        return False
    
    def combine_chapter_audio(self, chunk_files, output_path, intro_audio=None):
        """Combine audio files with professional spacing and normalization"""
        try:
            combined = AudioSegment.empty()
            
            # Add intro if provided
            if intro_audio:
                combined += intro_audio
            
            # Add each chunk with proper spacing
            for i, file_path in enumerate(chunk_files):
                if os.path.exists(file_path):
                    try:
                        audio = AudioSegment.from_wav(file_path)
                        
                        # Normalize volume
                        audio = normalize(audio)
                        
                        # Add to combined audio
                        combined += audio
                        
                        # Add natural pause between chunks (except last)
                        if i < len(chunk_files) - 1:
                            # Shorter pause for same paragraph, longer for paragraph breaks
                            pause_duration = 800  # 0.8 seconds
                            combined += AudioSegment.silent(duration=pause_duration)
                    except Exception as e:
                        print(f"⚠️ Could not process audio file {file_path}: {e}")
                        continue
            
            # Final normalization and export
            if len(combined) > 0:
                combined = normalize(combined)
                combined.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "22050"])
                return True
            else:
                print("❌ No audio content to combine")
                return False
            
        except Exception as e:
            print(f"❌ Error combining audio: {e}")
            return False

# Example usage and test
if __name__ == "__main__":
    print("🎬 High-Quality Audiobook Creator - Python 3.11")
    print("=" * 50)
    
    # Check for reference voice file
    reference_paths = [
        "voice_samples/shadow_slave_jessica.wav",
        "C:/Users/Domi/Desktop/TTS/voice_samples/shadow_slave_jessica.wav",
        "voice_samples/shadow_slave_jessica.mp3",  # Will suggest conversion
        "C:/Users/Domi/Desktop/TTS/voice_samples/shadow_slave_jessica.mp3"
    ]
    
    reference_voice = None
    for path in reference_paths:
        if os.path.exists(path):
            reference_voice = path
            break
    
    if not reference_voice:
        print("❌ Reference voice file not found!")
        print("Please ensure your reference audio exists at:")
        for path in reference_paths:
            print(f"  - {path}")
        print("\n💡 Tip: Use convert_to_wav.py to convert MP3 to WAV first")
        exit(1)
    
    if reference_voice.endswith('.mp3'):
        print("⚠️  Using MP3 file. For best quality, convert to WAV first:")
        print("   py -3.11 convert_to_wav.py")
    
    try:
        # Initialize creator
        creator = AudiobookCreator(reference_voice)
        
        # Test with a sample story
        sample_story = """
        Jessica stood at the edge of the shadow realm, her heart pounding with anticipation. 
        The ancient prophecy had spoken of this moment, but nothing could have prepared her for the reality.
        
        "You don't have to do this alone," whispered a familiar voice behind her.
        
        She turned, tears glistening in her eyes. "But I do. This is my destiny, my burden to bear."
        
        The shadows began to dance around her, responding to her emotions. Power coursed through her veins, 
        both exhilarating and terrifying. She had become something more than human, something the world had never seen.
        
        With a deep breath, she stepped forward into the unknown, ready to embrace whatever awaited her in the darkness.
        """
        
        # Create the sample chapter
        print("\n🎭 Creating sample audiobook chapter...")
        success = creator.create_chapter_with_quality(
            sample_story, 
            "Sample_Chapter_Jessica", 
            speed=0.95,  # Slightly slower for dramatic effect
            add_chapter_intro=True
        )
        
        if success:
            print("\n🎉 Sample chapter created successfully!")
            print("📁 Check the 'audiobook_output' folder for your audio file")
            print("\n💡 Tips for your full audiobook:")
            print("   - Adjust speed (0.8-1.2) for different moods")
            print("   - Use chapter introductions for professional feel")
            print("   - Process long books in chapters to manage memory")
        else:
            print("\n❌ Sample chapter creation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure your reference audio file is clear and uncompressed")
        print("   - Try converting MP3 to WAV format")
        print("   - Check that you have enough disk space")
        print("   - Make sure ffmpeg is installed for audio processing")