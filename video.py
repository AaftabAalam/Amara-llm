import os
import shutil
import whisper
from groq import Groq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from moviepy.video.io.VideoFileClip import VideoFileClip

PERSIST_DIRECTORY = "chroma_db"

# Groq API key
GROQ_API_KEY = "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"

# Whisper model for transcription
whisper_model = whisper.load_model("base")


class VideoProcessor:
    def __init__(self, groq_api_key):
        self.groq_client = Groq(api_key=groq_api_key)

    def convert_video_to_audio(self, video_path):
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
            video = VideoFileClip(video_path)
            audio_path = os.path.splitext(video_path)[0] + "_audio.wav"
            video.audio.write_audiofile(audio_path)
            video.close()
            return audio_path
        except Exception as e:
            raise RuntimeError(f"Audio extraction error: {e}")

    def transcribe_audio(self, audio_path):
        try:
            result = whisper_model.transcribe(audio_path)
            return result['text']
        except Exception as e:
            raise RuntimeError(f"Transcription error: {e}")

    def summarize_with_groq(self, transcription):
        try:
            prompt = f"""
            Please summarize the following transcription into a concise and meaningful paragraph:
            {transcription}
            """
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes content."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192"
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Summarization error: {e}")

    def generate_qa_with_groq(self, summary, num_questions):
        try:
            prompt = f"""
            Based on the following summary, generate {num_questions} unique and insightful 
            questions along with their corresponding answers:
            {summary}
            """
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates questions and answers."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192"
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Q&A generation error: {e}")
