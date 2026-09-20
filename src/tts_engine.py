"""
Text-to-Speech Engine
Converts translated text to natural speech output
"""

import pyttsx3
import threading
import queue
from pathlib import Path


class TTSEngine:
    """Text-to-Speech engine for ISL translations"""
    
    def __init__(self, engine='pyttsx3', rate=150, volume=1.0, voice_index=1):
        """
        Initialize TTS engine
        
        Args:
            engine: TTS engine to use ('pyttsx3' or 'gtts')
            rate: Speaking rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice_index: Voice selection (0=male, 1=female typically)
        """
        self.engine_type = engine
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.engine = None
        self.gTTS = None
        self.pygame = None

        # Thread-safe queue for TTS requests
        self.tts_queue = queue.Queue()
        self.is_running = True
        
        # Initialize engine
        if engine == 'pyttsx3':
            self._init_pyttsx3()
        else:
            self._init_gtts()
        
        # Start TTS worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            
            if voices and len(voices) > self.voice_index:
                self.engine.setProperty('voice', voices[self.voice_index].id)
            
            print(f"✓ pyttsx3 TTS engine initialized")
            print(f"  Rate: {self.rate} WPM")
            print(f"  Volume: {self.volume}")
            print(f"  Available voices: {len(voices) if voices else 0}")
            
        except Exception as e:
            print(f"Error initializing pyttsx3: {e}")
            self.engine = None
    
    def _init_gtts(self):
        """Initialize gTTS engine"""
        try:
            from gtts import gTTS
            import pygame
            
            self.gTTS = gTTS
            pygame.mixer.init()
            self.pygame = pygame
            
            print(f"✓ gTTS engine initialized")
        except ImportError as e:
            print(f"Error: {e}")
            print("Please install: pip install gtts pygame")
            self.gTTS = None
    
    def speak(self, text, async_mode=True):
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            async_mode: If True, speaks in background; if False, blocks until complete
        """
        if not text or not text.strip():
            return
        
        if async_mode:
            # Add to queue for async processing
            self.tts_queue.put(text)
        else:
            # Speak immediately (blocking)
            self._speak_now(text)
    
    def _speak_now(self, text):
        """Speak text immediately (blocking)"""
        if self.engine_type == 'pyttsx3' and self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        
        elif self.engine_type == 'gtts' and self.gTTS:
            try:
                # Generate audio file
                tts = self.gTTS(text=text, lang='en', slow=False)
                temp_file = Path('temp_tts.mp3')
                tts.save(str(temp_file))
                
                # Play audio
                self.pygame.mixer.music.load(str(temp_file))
                self.pygame.mixer.music.play()
                
                while self.pygame.mixer.music.get_busy():
                    self.pygame.time.Clock().tick(10)
                
                # Cleanup
                if temp_file.exists():
                    temp_file.unlink()
            
            except Exception as e:
                print(f"TTS Error: {e}")
    
    def _process_queue(self):
        """Process TTS queue in background thread"""
        while self.is_running:
            try:
                # Get text from queue with timeout
                text = self.tts_queue.get(timeout=0.1)
                if text:
                    self._speak_now(text)
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Queue Error: {e}")
    
    def set_rate(self, rate):
        """Change speaking rate"""
        self.rate = rate
        if self.engine_type == 'pyttsx3' and self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """Change volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        if self.engine_type == 'pyttsx3' and self.engine:
            self.engine.setProperty('volume', self.volume)
    
    def set_voice(self, voice_index):
        """Change voice"""
        self.voice_index = voice_index
        if self.engine_type == 'pyttsx3' and self.engine:
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > voice_index:
                self.engine.setProperty('voice', voices[voice_index].id)
    
    def get_voices(self):
        """Get list of available voices"""
        if self.engine_type == 'pyttsx3' and self.engine:
            voices = self.engine.getProperty('voices')
            return [
                {'id': i, 'name': v.name, 'languages': v.languages}
                for i, v in enumerate(voices)
            ] if voices else []
        return []
    
    def stop(self):
        """Stop current speech"""
        if self.engine_type == 'pyttsx3' and self.engine:
            self.engine.stop()
        elif self.engine_type == 'gtts' and self.pygame:
            self.pygame.mixer.music.stop()
    
    def clear_queue(self):
        """Clear pending TTS requests"""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break
    
    def shutdown(self):
        """Shutdown TTS engine"""
        self.is_running = False
        self.clear_queue()
        
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        if self.engine_type == 'pyttsx3' and self.engine:
            self.engine.stop()
        
        print("TTS engine shutdown")


class EnhancedTTS(TTSEngine):
    """Enhanced TTS with additional features for news channels"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emphasis_words = set()
        self.pause_after_words = set()
    
    def add_emphasis(self, words):
        """
        Add words that should be emphasized
        
        Args:
            words: List of words or single word
        """
        if isinstance(words, str):
            words = [words]
        self.emphasis_words.update(word.lower() for word in words)
    
    def add_pause_after(self, words):
        """
        Add words that should have a pause after them
        
        Args:
            words: List of words or single word
        """
        if isinstance(words, str):
            words = [words]
        self.pause_after_words.update(word.lower() for word in words)
    
    def speak_news(self, headline, details=None):
        """
        Speak news in professional format
        
        Args:
            headline: News headline
            details: Optional details
        """
        # Speak headline with emphasis
        self.speak(headline, async_mode=False)
        
        # Pause between headline and details
        if details:
            import time
            time.sleep(0.5)
            self.speak(details, async_mode=False)
    
    def speak_with_ssml(self, text, ssml_tags=None):
        """
        Speak with SSML-like formatting
        (Simplified version - full SSML requires different engines)
        
        Args:
            text: Text to speak
            ssml_tags: Dictionary of formatting instructions
        """
        # This is a simplified version
        # For full SSML support, consider using cloud TTS services
        formatted_text = text
        
        if ssml_tags:
            # Apply basic formatting
            if 'rate' in ssml_tags:
                original_rate = self.rate
                self.set_rate(ssml_tags['rate'])
                self.speak(formatted_text)
                self.set_rate(original_rate)
                return
        
        self.speak(formatted_text)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("\n" + "="*60)
    print("TEXT-TO-SPEECH ENGINE TEST")
    print("="*60 + "\n")
    
    # Initialize TTS
    tts = EnhancedTTS(rate=150, volume=1.0)
    
    # List available voices
    voices = tts.get_voices()
    print(f"Available voices: {len(voices)}")
    for voice in voices[:3]:  # Show first 3
        print(f"  {voice['id']}: {voice['name']}")
    
    print("\n" + "-"*60)
    
    # Test speeches
    test_messages = [
        "Welcome to the Indian Sign Language translation system.",
        "This system converts sign language to speech in real-time.",
        "Breaking news: The weather forecast predicts rain tomorrow."
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\nTest {i}: {msg}")
        tts.speak(msg, async_mode=False)
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("TTS TEST COMPLETE")
    print("="*60 + "\n")
    
    # Shutdown
    tts.shutdown()
