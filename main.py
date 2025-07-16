import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langdetect import detect
import boto3
import subprocess
from io import BytesIO
import openai
import time
from urllib.parse import urlparse, parse_qs
import logging
import base64
import hashlib
import tempfile
import uuid

# Load .env variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FFMPEG_PATH = os.path.abspath("bin/ffmpeg")

# Streamlit config
st.set_page_config(page_title="YT Summary Chatbot", layout="centered")
st.title("🎥 YouTube Video Summarizer & Chatbot")

# Enhanced styling with audio player customization
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .question-box {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #2196F3;
        margin: 0.5rem 0;
    }
    .audio-controls {
        background-color: #2d2d30;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #404040;
    }
    .audio-player {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0.5rem 0;
    }
    .play-button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        border: none;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.3s;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .play-button:hover {
        background: linear-gradient(45deg, #45a049, #4CAF50);
        transform: translateY(-1px);
    }
    .voice-selector {
        background-color: #3d3d40;
        border: 1px solid #555;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
    }
    .audio-status {
        font-size: 12px;
        color: #888;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced sidebar for audio settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Audio Settings Section
    st.subheader("🎵 Audio Settings")
    enable_audio = st.checkbox("Enable Audio Responses", value=True)
    
    if enable_audio:
        # Voice selection
        voice_options = {
            'Joanna': 'Female US English (Joanna)',
            'Matthew': 'Male US English (Matthew)',
            'Salli': 'Female US English (Salli)',
            'Joey': 'Male US English (Joey)',
            'Amy': 'Female British English (Amy)',
            'Brian': 'Male British English (Brian)',
            'Emma': 'Female British English (Emma)',
            'Aditi': 'Female Hindi/English (Aditi)',
            'Raveena': 'Female Indian English (Raveena)',
            'Celine': 'Female French (Celine)',
            'Mathieu': 'Male French (Mathieu)',
            'Marlene': 'Female German (Marlene)',
            'Hans': 'Male German (Hans)',
            'Conchita': 'Female Spanish (Conchita)',
            'Enrique': 'Male Spanish (Enrique)'
        }
        
        selected_voice = st.selectbox(
            "Select Voice",
            options=list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=0
        )
        
        # Audio quality settings
        audio_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
        max_chars_tts = st.slider("Max characters for TTS", 100, 2000, 800, 
                                 help="Limit text length for audio generation")
        
        # Auto-play option
        auto_play_audio = st.checkbox("Auto-play responses", value=False)
        
        # Audio format
        audio_format = "wav"  # Force audio format to WAV only
    
    st.divider()
    st.subheader("🔧 Other Settings")

video_url = st.text_input("📺 Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}

# Helper: Validate video accessibility
def validate_video(video_id):
    """Check if video is accessible and has transcripts"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available = list(transcript_list)
        
        transcript_info = []
        for t in available:
            transcript_info.append({
                'language': t.language,
                'language_code': t.language_code,
                'is_generated': t.is_generated,
                'is_translatable': t.is_translatable
            })
        
        return {
            'accessible': True,
            'transcript_count': len(available),
            'transcripts': transcript_info
        }
    except TranscriptsDisabled:
        return {'accessible': False, 'error': 'Transcripts disabled'}
    except NoTranscriptFound:
        return {'accessible': False, 'error': 'No transcripts found'}
    except Exception as e:
        return {'accessible': False, 'error': str(e)}

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    if not url:
        return None
    
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query).get('v', [None])[0]
        elif query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        elif query.path.startswith('/v/'):
            return query.path.split('/')[2]
    return None

# Helper: Get transcript with comprehensive error handling and debugging
def get_transcript(video_id):
    """Fetch transcript with comprehensive language support and detailed debugging"""
    try:
        # First, list all available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_transcripts = list(transcript_list)
        
        if not available_transcripts:
            logger.error("No transcripts available for this video")
            return None, None
        
        # Log available transcripts for debugging
        logger.info(f"Available transcripts: {[(t.language_code, t.language) for t in available_transcripts]}")
        
        # Try to get transcript in preferred order
        languages_to_try = ["en", "en-US", "en-GB", "en-IN", "hi", "fr", "de", "es", "it", "pt", "ja", "ko", "zh"]
        
        for lang in languages_to_try:
            try:
                transcript = transcript_list.find_transcript([lang])
                transcript_data = transcript.fetch()
                text = " ".join(entry.text for entry in transcript_data)
                logger.info(f"Successfully fetched transcript in {lang}")
                return text, lang
            except:
                continue
        
        # Try auto-generated transcripts
        for transcript in available_transcripts:
            try:
                if transcript.is_generated:
                    transcript_data = transcript.fetch()
                    text = " ".join(entry.text for entry in transcript_data)
                    lang = transcript.language_code[:2] if transcript.language_code else 'en'
                    logger.info(f"Using auto-generated transcript in {transcript.language_code}")
                    return text, lang
            except Exception as e:
                logger.error(f"Failed to fetch auto-generated transcript: {e}")
                continue
        
        # Try any available transcript as last resort
        for transcript in available_transcripts:
            try:
                transcript_data = transcript.fetch()
                text = " ".join(entry.text for entry in transcript_data)
                lang = transcript.language_code[:2] if transcript.language_code else 'en'
                logger.info(f"Using transcript in {transcript.language_code}")
                return text, lang
            except Exception as e:
                logger.error(f"Failed to fetch transcript in {transcript.language_code}: {e}")
                continue
                
        return None, None
        
    except TranscriptsDisabled:
        logger.error("Transcripts are disabled for this video")
        return None, None
    except NoTranscriptFound:
        logger.error("No transcript found for this video")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error getting transcript: {e}")
        return None, None

# Enhanced text-to-speech with better error handling and caching
def generate_audio(text, voice_id='Joanna', speed=1.0, format='mp3'):
    """Robust AWS Polly TTS with fallback conversion to WAV"""
    if not enable_audio or not text.strip():
        return None

    try:
        # Truncate text
        if len(text) > max_chars_tts:
            text = text[:max_chars_tts] + "..."
        cache_key = hashlib.md5(f"{text[:100]}{voice_id}{speed}{format}".encode()).hexdigest()

        # Return cached audio if exists
        if cache_key in st.session_state.audio_cache:
            return st.session_state.audio_cache[cache_key]

        # Polly client
        polly = boto3.client(
            "polly",
            region_name="us-east-1",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        response = polly.synthesize_speech(
            Text=text,
            TextType='text',
            OutputFormat="mp3",  # Polly only supports mp3, ogg_vorbis, pcm, json
            VoiceId=voice_id
        )

        if "AudioStream" in response:
            mp3_bytes = response["AudioStream"].read()

            if audio_format == "wav":
                # Convert MP3 to WAV using ffmpeg
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
                    mp3_file.write(mp3_bytes)
                    mp3_path = mp3_file.name
                
                wav_path = mp3_path.replace(".mp3", ".wav")

                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", mp3_path, wav_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    with open(wav_path, "rb") as wav_file:
                        wav_bytes = wav_file.read()
                    st.session_state.audio_cache[cache_key] = wav_bytes
                    return wav_bytes
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg conversion failed: {e}")
                    st.error("Audio conversion failed. Check ffmpeg installation.")
                    return None

            else:
                # Default return MP3
                st.session_state.audio_cache[cache_key] = mp3_bytes
                return mp3_bytes

        st.error("No audio stream returned from Polly.")
        return None

    except Exception as e:
        logger.error(f"TTS error: {e}")
        st.error(f"TTS generation failed: {e}")
        return None

def create_audio_player(text, response_id, voice_id='Joanna'):
    """Create a custom audio player with controls and safe stream handling"""
    if not enable_audio:
        return

    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button(f"🔊 Play", key=f"play_{response_id}"):
            with st.spinner("🎵 Generating audio..."):
                audio_bytes = generate_audio(text, voice_id, audio_speed, audio_format)
                if audio_bytes:
                    if len(audio_bytes) < 500:
                        st.session_state[f"audio_{response_id}"] = None
                        st.warning("⚠️ Audio is too short to play.")
                    else:
                        st.session_state[f"audio_{response_id}"] = audio_bytes
                    st.rerun()

    with col2:
        # Show audio status
        if st.session_state.get(f"audio_{response_id}"):
            st.success("🎵 Audio ready!")
        else:
            st.info("Click play to generate audio")

    with col3:
        if st.session_state.get(f"audio_{response_id}"):
            if st.button(f"⏹️ Clear", key=f"clear_{response_id}"):
                del st.session_state[f"audio_{response_id}"]
                st.rerun()

    # Display player if ready
    audio_bytes = st.session_state.get(f"audio_{response_id}")
    if audio_bytes:
        st.audio(BytesIO(audio_bytes), format=f"audio/{audio_format}", autoplay=auto_play_audio)
        st.download_button(
            label=f"📥 Download Audio ({audio_format.upper()})",
            data=audio_bytes,
            file_name=f"response_{response_id}.{audio_format}",
            mime=f"audio/{audio_format}",
            key=f"download_{response_id}"
        )


# Helper: Process video with progress tracking
def process_video(video_id):
    """Process video with detailed progress feedback"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Get transcript
        status_text.text("🎬 Fetching video transcript...")
        progress_bar.progress(20)
        
        transcript, lang = get_transcript(video_id)
        if not transcript:
            # Enhanced error message with suggestions
            st.error("""
            ❌ **Could not fetch transcript for this video.**
            
            **Possible reasons:**
            - Video doesn't have captions/subtitles
            - Video is private or restricted
            - Captions are disabled by the creator
            - Video is too new (captions not yet generated)
            
            **Try:**
            - A different video with captions
            - Enabling auto-generated captions if you're the creator
            - Waiting if the video is very recent
            """)
            
            # Suggest alternative videos for testing
            with st.expander("🔍 Try these test videos"):
                st.markdown("""
                **Videos with reliable transcripts:**
                - `https://www.youtube.com/watch?v=dQw4w9WgXcQ` (Rick Astley - Never Gonna Give You Up)
                - `https://www.youtube.com/watch?v=9bZkp7q19f0` (PSY - Gangnam Style)
                - Any TED Talk video
                - Most educational channels (Khan Academy, Crash Course, etc.)
                """)
            return False
        
        # Step 2: Process transcript
        status_text.text("📝 Processing transcript...")
        progress_bar.progress(40)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.create_documents([transcript])
        
        # Step 3: Create embeddings
        status_text.text("🧠 Creating embeddings...")
        progress_bar.progress(60)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        # Step 4: Setup QA chain
        status_text.text("⚡ Setting up AI assistant...")
        progress_bar.progress(80)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        
        # Step 5: Store in session
        progress_bar.progress(100)
        status_text.text("✅ Ready to chat!")
        
        st.session_state.qa_chain = qa_chain
        st.session_state.vector_store = vector_store
        st.session_state.transcript_lang = lang or 'en'
        st.session_state.ready = True
        st.session_state.video_id = video_id
        
        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        logger.error(f"Video processing error: {e}")
        st.error(f"❌ Processing failed: {str(e)}")
        return False

# Main processing
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🔍 Analyze Video", disabled=st.session_state.get("processing", False)):
        if not video_url:
            st.error("Please enter a YouTube URL")
        else:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL format")
            else:
                st.session_state.processing = True
                success = process_video(video_id)
                st.session_state.processing = False
                
                if success:
                    st.success("🎉 Video processed successfully! You can now ask questions.")
                    # Auto-generate first summary
                    if "chat_history" not in st.session_state or not st.session_state.chat_history:
                        with st.spinner("Generating initial summary..."):
                            try:
                                summary_question = "Please provide a comprehensive summary of this video's main points."
                                answer_data = st.session_state.qa_chain.invoke({"question": summary_question})
                                answer = answer_data["answer"]
                                response_id = str(uuid.uuid4())
                                st.session_state.chat_history.append((summary_question, answer, response_id))
                            except Exception as e:
                                logger.error(f"Auto-summary error: {e}")

with col2:
    if st.session_state.get("ready"):
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            # Clear audio cache
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith('audio_')]
            for key in keys_to_remove:
                del st.session_state[key]
            st.rerun()
    
    # Add video validation button
    if video_url and not st.session_state.get("ready"):
        if st.button("🔍 Check Video"):
            video_id = extract_video_id(video_url)
            if video_id:
                with st.spinner("Checking video..."):
                    validation = validate_video(video_id)
                    
                if validation['accessible']:
                    st.success(f"✅ Video accessible! Found {validation['transcript_count']} transcript(s)")
                    
                    with st.expander("📋 Available transcripts"):
                        for t in validation['transcripts']:
                            status = "🤖 Auto-generated" if t['is_generated'] else "👤 Manual"
                            translate = "🌍 Translatable" if t['is_translatable'] else ""
                            st.write(f"• **{t['language']}** ({t['language_code']}) {status} {translate}")
                else:
                    st.error(f"❌ Cannot access video: {validation['error']}")
            else:
                st.error("Invalid video URL")

# Voice input section
if st.session_state.get("ready"):
    st.subheader("🎤 Voice Input")
    uploaded_audio = st.file_uploader(
        "Upload audio file to ask a question", 
        type=["mp3", "wav", "m4a"],
        help="Record a question and upload it here"
    )
    
    if uploaded_audio:
        with st.spinner("🎧 Transcribing your question..."):
            try:
                # Create OpenAI client
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Transcribe audio
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=uploaded_audio
                )
                question = transcript_response.text
                
                st.success(f"🗣️ **You asked:** {question}")
                
                # Get answer
                with st.spinner("🤔 Thinking..."):
                    answer_data = st.session_state.qa_chain.invoke({"question": question})
                    answer = answer_data["answer"]
                    response_id = str(uuid.uuid4())
                    st.session_state.chat_history.append((question, answer, response_id))
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Voice processing failed: {str(e)}")

# Text input section
if st.session_state.get("ready"):
    st.subheader("❓ Ask Questions")
    
    with st.form("question_form"):
        user_question = st.text_area(
            "What would you like to know about the video?",
            placeholder="e.g., What are the main topics discussed? Can you explain the key points?",
            height=100
        )
        ask_button = st.form_submit_button("Ask Question", use_container_width=True)
        
        if ask_button and user_question.strip():
            with st.spinner("🤔 Analyzing your question..."):
                try:
                    answer_data = st.session_state.qa_chain.invoke({"question": user_question})
                    answer = answer_data["answer"]
                    response_id = str(uuid.uuid4())
                    st.session_state.chat_history.append((user_question, answer, response_id))
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to get answer: {str(e)}")

# Enhanced chat history display with custom audio players
if st.session_state.get("chat_history"):
    st.subheader("💬 Conversation History")
    
    for i, chat_item in enumerate(reversed(st.session_state.chat_history)):
        # Handle both old format (question, answer) and new format (question, answer, response_id)
        if len(chat_item) == 3:
            question, answer, response_id = chat_item
        else:
            question, answer = chat_item
            response_id = f"legacy_{i}"
        
        # Question
        st.markdown(f"""
        <div class="question-box">
            <strong>🙋 You asked:</strong><br>
            {question}
        </div>
        """, unsafe_allow_html=True)
        
        # Answer
        st.markdown(f"""
        <div class="answer-box">
            <strong>🤖 Answer:</strong><br>
            {answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Custom Audio Player
        if enable_audio:
            st.markdown("""
            <div class="audio-controls">
                <strong>🎵 Audio Response</strong>
            </div>
            """, unsafe_allow_html=True)
            
            create_audio_player(answer, response_id, selected_voice)
                
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

# Footer
if not st.session_state.get("ready"):
    st.info("👆 Enter a YouTube URL above to get started!")
    
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Paste a YouTube URL** in the input field above
        2. **Click 'Analyze Video'** to process the video transcript
        3. **Ask questions** about the video content using text or voice
        4. **Listen to responses** with custom audio controls and voice selection
        5. **Download audio** responses for offline listening
        
        **Enhanced Audio Features:**
        - 🎵 Multiple voice options (15+ voices)
        - ⚡ Adjustable speech speed
        - 🎛️ Custom audio controls
        - 📥 Download audio responses
        - 🔄 Audio caching for better performance
        - 🎯 Auto-play option
        
        **Supported features:**
        - 🎬 Automatic transcript extraction
        - 🧠 AI-powered question answering
        - 🎤 Voice input support
        - 🔊 Enhanced text-to-speech responses
        - 🌍 Multiple language support
        """)

# Debug info (only in development)
if st.checkbox("🔧 Show debug info"):
    with st.expander("Debug Information"):
        st.write("Session state keys:", list(st.session_state.keys()))
        if st.session_state.get("ready"):
            st.write("Video ID:", st.session_state.get("video_id"))
            st.write("Transcript language:", st.session_state.get("transcript_lang"))
            st.write("Chat history length:", len(st.session_state.get("chat_history", [])))
            st.write("Audio cache size:", len(st.session_state.get("audio_cache", {})))
            st.write("Selected voice:", selected_voice if enable_audio else "Audio disabled")