import os
import json
import shutil
import whisper
from groq import Groq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image
import json
from ultralytics import YOLO
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime

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
        
    def analyze_sentiment_with_groq(self, text: str) -> str:
        """
        Analyzes the sentiment of the given text and returns 'Positive', 'Neutral', or 'Negative'.
        """
        try:
            prompt = f"""
            Analyze the sentiment of the following responses and return only one word: 'Positive', 'Neutral', or 'Negative'.
            Responses:
            {text}
            """
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI that strictly returns sentiment analysis results."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192"
            )
            sentiment = chat_completion.choices[0].message.content.strip()
            return sentiment if sentiment in ["Positive", "Neutral", "Negative"] else "Neutral"
        except Exception as e:
            return "Neutral"

def extract_tags(data):
    prompt = f"""
    Your task is to analyze the given text data and extract the most relevant tags that accurately represent the key topics, themes, or concepts mentioned in the text.
    Instructions:
    1. Identify the core topics, entities, or themes present in the text.
    2. Extract concise, meaningful tags that summarize the main points.
    3. Ensure the tags are short (1-8 or 1-15 words each less or more), relevant, and specific to the content.
    4. Exclude generic words (e.g., "the," "is," "very") and focus on substantive keywords.
    5. Return only a JSON-formatted array of tags, nothing else.
    
    Example output: ["Tag1", "Tag2", "Tag3", "Tag4"]

    This is the incoming data on which you have to perform the operations:
    {data}
"""
    client = Groq(api_key="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")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    response = chat_completion.choices[0].message.content
    try:
        return json.loads(response)
    except json.JSONDecoderError:
        return []

class AttentionTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.phone_detector = YOLO("yolov8n.pt")

        self.attention_threshold = 0.81
        self.attention_window = 5
        self.attention_history = []
        self.last_alert_time = 0
        self.alert_cooldown = 10

        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.NOSE = 1

    def calculate_eye_aspect_ratio(self, eye_landmarks):
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (v1 + v2) / (2.0 * h)

    def detect_phone(self, frame):
        results = self.phone_detector(frame)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 67:
                    return True, box.xyxy[0].tolist()
        return False, None

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        phone_detected, phone_box = self.detect_phone(frame)
        current_time = time.time()

        attention_data = {
            "timestamp": datetime.now().isoformat(),
            "attention_score": None,
            "phone_detected": phone_detected,
            "phone_box": phone_box,
            "eye_aspect_ratio": None,
            "gaze_score": None,
            "alert": False,
            "alert_message": None
        }

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1], 
                                face_landmarks.landmark[i].y * frame.shape[0]] 
                                for i in self.LEFT_EYE])
            right_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1], 
                                 face_landmarks.landmark[i].y * frame.shape[0]] 
                                 for i in self.RIGHT_EYE])

            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            ear = np.clip(ear / 0.35, 0, 1)

            nose = face_landmarks.landmark[self.NOSE]
            gaze_score = max(0, 1 - abs(nose.x - 0.5) * 2)

            attention_score = (0.6 * ear) + (0.4 * gaze_score)
            if phone_detected:
                attention_score *= 0.5

            self.attention_history.append((current_time, attention_score))
            self.attention_history = [(t, s) for t, s in self.attention_history 
                                    if current_time - t <= self.attention_window]
            avg_attention = np.mean([s for _, s in self.attention_history])

            alert = False
            alert_message = None
            if avg_attention < self.attention_threshold and current_time - self.last_alert_time > self.alert_cooldown:
                alert = True
                alert_message = "Distraction detected"
                self.last_alert_time = current_time

            attention_data.update({
                "attention_score": float(avg_attention),
                "eye_aspect_ratio": float(ear),
                "gaze_score": float(gaze_score),
                "alert": alert,
                "alert_message": alert_message
            })

        return attention_data