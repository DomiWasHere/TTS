#!/usr/bin/env python3
"""
Enhanced Multi-Audiobook Creator for Python 3.11
Supports multiple books from text files with chapter detection
Run with: py -3.11 multi_audiobook_creator.py
"""
import os
import re
import torch
import warnings
import json
from pathlib import Path
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize
import nltk

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")
warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*")

# Download required NLTK data
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

class MultiAudiobookCreator:
    def __init__(self, reference_voice_path):
        self.reference_voice = reference_voice_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎙️  Initializing MultiAudiobookCreator on {self.device}")
        
        # Verify reference file exists
        if not os.path.exists(reference_voice_path):
            raise FileNotFoundError(f"Reference voice file not found: {reference_voice_path}")
        
        # Initialize TTS with error suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("📥 Loading XTTS v2 model...")
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            print("✅ Model loaded successfully!")
    
    def parse_book_from_file(self, file_path):
        """Parse a book file and extract title, chapters"""
        print(f"📚 Parsing book file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract book title from filename or first line
        book_title = Path(file_path).stem
        
        # Try to find title in content (common patterns)
        title_patterns = [
            r'^Title:\s*(.+)$',
            r'^TITLE:\s*(.+)$', 
            r'^#\s*(.+)$',  # Markdown header
            r'^(.+)\n=+$',  # Underlined title
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                book_title = match.group(1).strip()
                # Remove the title line from content
                content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
                break
        
        # Clean up the content
        content = content.strip()
        
        # Chapter detection patterns (in order of preference)
        chapter_patterns = [
            r'(?:^|\n)(?:CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)(?:\s*[:\-\.]?\s*(.+?))?(?=\n|\r)',
            r'(?:^|\n)(?:CHAPTER|Chapter)\s+(.+?)(?=\n|\r)',
            r'(?:^|\n)(\d+)\.\s+(.+?)(?=\n|\r)',  # "1. Chapter Title"
            r'(?:^|\n)#{1,3}\s+(.+?)(?=\n|\r)',   # Markdown headers
            r'(?:^|\n)(.+?)\n=+(?=\n|\r)',       # Underlined chapters
            r'(?:^|\n)(.+?)\n-+(?=\n|\r)',       # Dashed underline
        ]
        
        chapters = []
        chapter_splits = None
        
        # Try each pattern until we find chapters
        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
            if len(matches) > 1:  # Need at least 2 chapters
                print(f"✅ Found {len(matches)} chapters using pattern")
                chapter_splits = matches
                break
        
        if chapter_splits:
            # Extract chapters based on splits
            for i, match in enumerate(chapter_splits):
                # Determine chapter title
                if len(match.groups()) >= 2 and match.group(2):
                    chapter_title = f"Chapter_{match.group(1)}_{match.group(2)}"
                else:
                    chapter_title = f"Chapter_{match.group(1) if match.group(1) else i+1}"
                
                # Clean chapter title for filename
                chapter_title = re.sub(r'[^\w\s\-_]', '', chapter_title)
                chapter_title = re.sub(r'\s+', '_', chapter_title)
                
                # Extract chapter content
                start_pos = match.end()
                end_pos = chapter_splits[i+1].start() if i+1 < len(chapter_splits) else len(content)
                chapter_content = content[start_pos:end_pos].strip()
                
                if chapter_content:  # Only add non-empty chapters
                    chapters.append({
                        'title': chapter_title,
                        'content': chapter_content
                    })
        else:
            # If no chapters found, split by paragraphs or length
            print("📝 No chapter markers found, splitting by content...")
            chunks = self.split_long_content(content)
            chapters = [{'title': f'Part_{i+1:02d}', 'content': chunk} 
                       for i, chunk in enumerate(chunks)]
        
        print(f"📖 Book parsed: '{book_title}' with {len(chapters)} chapters")
        return {
            'title': book_title,
            'chapters': chapters
        }
    
    def split_long_content(self, content, max_chapter_length=5000):
        """Split long content into manageable chapters"""
        paragraphs = content.split('\n\n')
        chapters = []
        current_chapter = ""
        
        for para in paragraphs:
            if len(current_chapter + para) > max_chapter_length and current_chapter:
                chapters.append(current_chapter.strip())
                current_chapter = para
            else:
                current_chapter += "\n\n" + para if current_chapter else para
        
        if current_chapter:
            chapters.append(current_chapter.strip())
        
        return chapters
    
    def preprocess_text_for_emotion(self, text):
        """Enhanced text preprocessing for emotional storytelling"""
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Add emotional pauses and emphasis
        text = re.sub(r'\.(?=\s[A-Z])', '... ', text)
        text = re.sub(r'!(?=\s)', '! ', text)
        text = re.sub(r'\?(?=\s)', '? ', text)
        text = re.sub(r',(?=\s)', ', ', text)
        text = re.sub(r';(?=\s)', '; ', text)
        text = re.sub(r':(?=\s)', ': ', text)
        
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
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
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
                
                try:
                    self.tts.tts_to_file(
                        text=processed_text,
                        speaker_wav=self.reference_voice,
                        language="en",
                        file_path=output_path,
                        split_sentences=True,
                        speed=speed,
                        temperature=temperature
                    )
                    return True
                except TypeError:
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
    
    def create_chapter_audio(self, chapter_text, chapter_name, book_output_dir, speed=0.95):
        """Create audio for a single chapter"""
        chapter_path = os.path.join(book_output_dir, f"{chapter_name}.wav")

        if os.path.exists(chapter_path):
            print(f"⏭️  Skipping existing chapter: {chapter_name}")
            return True
    

        print(f"🎵 Creating audio for: {chapter_name}")

        # Split chapter into chunks
        chunks = self.split_into_optimal_chunks(chapter_text)
        print(f"📝 Processing {len(chunks)} chunks...")
        
        # Create audio for each chunk
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(book_output_dir, f"{chapter_name}_chunk_{i:03d}.wav")
            
            if self.create_audio_with_settings(chunk, chunk_file, speed):
                chunk_files.append(chunk_file)
            else:
                print(f"⚠️ Failed to create chunk {i+1}")
        
        if not chunk_files:
            return False
        
        # Combine chunks into chapter
        chapter_path = os.path.join(book_output_dir, f"{chapter_name}.wav")
        success = self.combine_audio_files(chunk_files, chapter_path)
        
        # Clean up chunk files
        for chunk_file in chunk_files:
            try:
                os.remove(chunk_file)
            except:
                pass
        
        return success
    
    def combine_audio_files(self, audio_files, output_path):
        """Combine multiple audio files with proper spacing"""
        try:
            combined = AudioSegment.empty()
            
            for i, file_path in enumerate(audio_files):
                if os.path.exists(file_path):
                    try:
                        audio = AudioSegment.from_wav(file_path)
                        audio = normalize(audio)
                        combined += audio
                        
                        # Add pause between chunks
                        if i < len(audio_files) - 1:
                            combined += AudioSegment.silent(duration=800)
                    except Exception as e:
                        print(f"⚠️ Could not process {file_path}: {e}")
                        continue
            
            if len(combined) > 0:
                combined = normalize(combined)
                combined.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "22050"])
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error combining audio: {e}")
            return False
    
    def create_full_audiobook(self, book_data, output_base_dir="audiobooks", speed=0.95):
        """Create complete audiobook from parsed book data"""
        book_title = book_data['title']
        chapters = book_data['chapters']
        
        # Create book directory
        book_dir = os.path.join(output_base_dir, book_title)
        os.makedirs(book_dir, exist_ok=True)
        
        print(f"\n📚 Creating audiobook: {book_title}")
        print(f"📁 Output directory: {book_dir}")
        print(f"📖 Chapters to process: {len(chapters)}")
        
        successful_chapters = []
        failed_chapters = []
        
        # Process each chapter
        for i, chapter in enumerate(chapters):
            print(f"\n--- Chapter {i+1}/{len(chapters)}: {chapter['title']} ---")
            
            success = self.create_chapter_audio(
                chapter['content'], 
                chapter['title'], 
                book_dir, 
                speed
            )
            
            if success:
                successful_chapters.append(chapter['title'])
                print(f"✅ Chapter completed: {chapter['title']}")
            else:
                failed_chapters.append(chapter['title'])
                print(f"❌ Chapter failed: {chapter['title']}")
        
        # Create book summary
        summary = {
            'book_title': book_title,
            'total_chapters': len(chapters),
            'successful_chapters': len(successful_chapters),
            'failed_chapters': len(failed_chapters),
            'chapters': [ch['title'] for ch in chapters],
            'failed_list': failed_chapters
        }
        
        # Save summary
        summary_path = os.path.join(book_dir, 'audiobook_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 Audiobook Creation Complete!")
        print(f"📊 Success Rate: {len(successful_chapters)}/{len(chapters)} chapters")
        print(f"📁 Location: {book_dir}")
        
        if failed_chapters:
            print(f"⚠️  Failed chapters: {', '.join(failed_chapters)}")
        
        return book_dir, summary
    
    def process_multiple_books(self, book_files, output_dir="audiobooks", speed=0.95):
        """Process multiple book files"""
        results = []
        
        print(f"📚 Processing {len(book_files)} books...")
        
        for i, book_file in enumerate(book_files):
            print(f"\n{'='*60}")
            print(f"📖 Processing Book {i+1}/{len(book_files)}: {book_file}")
            print(f"{'='*60}")
            
            try:
                # Parse the book
                book_data = self.parse_book_from_file(book_file)
                
                # Create the audiobook
                book_dir, summary = self.create_full_audiobook(book_data, output_dir, speed)
                
                results.append({
                    'file': book_file,
                    'status': 'success',
                    'output_dir': book_dir,
                    'summary': summary
                })
                
            except Exception as e:
                print(f"❌ Failed to process {book_file}: {e}")
                results.append({
                    'file': book_file,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Create overall summary
        self.create_batch_summary(results, output_dir)
        
        return results
    
    def create_batch_summary(self, results, output_dir):
        """Create summary of batch processing"""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        batch_summary = {
            'total_books': len(results),
            'successful_books': len(successful),
            'failed_books': len(failed),
            'books': results,
            'timestamp': str(Path().cwd())
        }
        
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"🎉 BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Total Books: {len(results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"📄 Summary saved: {summary_path}")


# Example usage and configuration
if __name__ == "__main__":
    print("🎬 Multi-Audiobook Creator - Python 3.11")
    print("=" * 60)
    
    # Configuration
    BOOKS_FOLDER = "books"  # Folder containing your text files
    OUTPUT_FOLDER = "audiobooks"  # Where audiobooks will be saved
    VOICE_SPEED = 0.95  # Adjust speech speed (0.8-1.2)
    
    # Check for reference voice file
    reference_paths = [
        "voice_samples/shadow_slave_jessica.wav",
        "C:/Users/Domi/Desktop/TTS/voice_samples/shadow_slave_jessica.wav",
        "voice_samples/shadow_slave_jessica.mp3",
        "C:/Users/Domi/Desktop/TTS/voice_samples/shadow_slave_jessica.mp3"


"""         "voice_samples\donald_trump.mp3",
        "C:/Users/Domi/Desktop/TTS/voice_samples/donald_trump.mp3",
        "voice_samples\donald_trump.wav",
        "C:/Users/Domi/Desktop/TTS/voice_samples/donald_trump.wav" """
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
        exit(1)
    
    # Find book files
    book_extensions = ['.txt', '.md']
    book_files = []
    
    if os.path.exists(BOOKS_FOLDER):
        for ext in book_extensions:
            book_files.extend(Path(BOOKS_FOLDER).glob(f'*{ext}'))
    
    # Also check current directory
    for ext in book_extensions:
        book_files.extend(Path('.').glob(f'*{ext}'))
    
    book_files = [str(f) for f in book_files]
    
    if not book_files:
        print(f"❌ No book files found!")
        print(f"📁 Create a '{BOOKS_FOLDER}' folder and add your .txt or .md files")
        print(f"Or place book files in the current directory")
        
        # Create sample book file
        sample_book = """Title: Sample Adventure Story

Chapter 1: The Beginning

Jessica stood at the edge of the shadow realm, her heart pounding with anticipation. The ancient prophecy had spoken of this moment, but nothing could have prepared her for the reality.

"You don't have to do this alone," whispered a familiar voice behind her.

She turned, tears glistening in her eyes. "But I do. This is my destiny, my burden to bear."

Chapter 2: The Journey

The shadows began to dance around her, responding to her emotions. Power coursed through her veins, both exhilarating and terrifying. She had become something more than human, something the world had never seen.

With a deep breath, she stepped forward into the unknown, ready to embrace whatever awaited her in the darkness.

Chapter 3: The Resolution

Hours later, she emerged from the shadow realm transformed. The journey had changed her in ways she never imagined possible. She was ready to face the world with her newfound power and wisdom."""
        
        os.makedirs(BOOKS_FOLDER, exist_ok=True)
        sample_path = os.path.join(BOOKS_FOLDER, "sample_book.txt")
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_book)
        
        print(f"✅ Created sample book: {sample_path}")
        book_files = [sample_path]
    
    print(f"📚 Found {len(book_files)} book files:")
    for book_file in book_files:
        print(f"  - {book_file}")
    
    try:
        # Initialize creator
        creator = MultiAudiobookCreator(reference_voice)
        
        # Process all books
        results = creator.process_multiple_books(book_files, OUTPUT_FOLDER, VOICE_SPEED)
        
        print(f"\n🎉 All audiobooks processed!")
        print(f"📁 Check the '{OUTPUT_FOLDER}' folder for your audiobooks")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure your reference audio file is clear and uncompressed")
        print("   - Check that you have enough disk space")
        print("   - Make sure ffmpeg is installed for audio processing")