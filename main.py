from fastapi import FastAPI, HTTPException, Request,UploadFile, File, Form, Body, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional,Union
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from typing import List, Optional, Dict
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import shutil
from datetime import datetime
import uuid
from video import VideoProcessor, extract_tags, AttentionTracker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import re
from groq import Groq
import requests
import json
import statistics
from tenacity import retry, stop_after_attempt, wait_fixed
import base64
import cv2
from PIL import Image
import io
import numpy as np

app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reume and linkedin comparison code and api
def extract_key_info(data):
    try:
        prompt = f"""
        Please extract the following information from the resume text below:

        1. Full Name
        2. Industry
        3. Job Title
        4. Current Location
        5. Job Location
        6. Education Details
        7. Job Start Date
        8. Job End Date (if applicable)
        9. Skills
        10. Interests
        11. Experience
        12. Current Company Name

        Resume Text:
        {data}

        Provide ONLY the extracted information in the following JSON format without any gap or space and with no additional commentary or explanations:

        {{
            "full_name": "<extracted_full_name>",
            "industry": "<extracted_industry>",
            "job_title": "<extracted_job_title>",
            "current_location": "<extracted_current_location>",
            "job_location": "<extracted_job_location>",
            "education": "<extracted_education_details>",
            "job_start_date": "<extracted_job_start_date>",
            "job_end_date": "<extracted_job_end_date>",
            "skills": ["<skill_1>", "<skill_2>", ...],
            "interests": ["<interest_1>", "<interest_2>", ...],
            "experience": ["<experience_1>", "<experience_2>", ...],
            "current_company": "<extracted_current_company>"
        }}
        """

        client = Groq(api_key="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")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile"
        )
        message = chat_completion.choices[0].message.content
        return message
    except Exception as e:
        return{
            "error": f"An error occurred: {str(e)}"
        }

def compare_data(data1, data2):
    try:
        prompt = f"""
        You are an advanced LLM tasked with verifying the correctness of information in data2 by comparing it with data1. For each field, check the accuracy of the information provided in data2 and calculate a percentage match.
        Please do not generate your comments and explanations.
        Provide an individual score for each field and an overall percentage score based on the following criteria:

        1. **Full Name**: Check if the `full_name` fields in data1 and data2 match exactly (case-insensitive).
        2. **Occupation**: Compare the `occupation` field in data1 with the `job_title` field in data2 using semantic similarity.
        3. **Headline**: Compare the `headline` in data1 with any relevant title or description in data2 for semantic similarity.
        4. **Country**: Verify if the `country_full_name` field in data1 matches the `current_location` or `job_location` field in data2.
        5. **City**: Check if the `city` field in data1 is included in the `current_location` field in data2.
        6. **State**: Verify if the `state` field in data1 matches any part of the `current_location` or `job_location` field in data2.
        7. **Experiences**: Compare the `experiences` list in data1 with the `experience` list in data2 for similarity. Check company names, job titles, and descriptions.
        8. **Education**: Compare the `education` fields in data1 and data2, including degree name, field of study, school, and date ranges.
        9. **Skills**: Calculate the percentage overlap between the `skills` lists in data1 and data2.

        Use the following weights for the overall trueness score:
        - Full Name: 10%
        - Occupation: 10%
        - Headline: 10%
        - Country: 10%
        - City: 5%
        - State: 5%
        - Experiences: 30%
        - Education: 10%
        - Skills: 10%

        Provide the response in the following JSON format:
        {{
            "full_name_match_percentage": <value>,
            "occupation_match_percentage": <value>,
            "headline_match_percentage": <value>,
            "country_match_percentage": <value>,
            "city_match_percentage": <value>,
            "state_match_percentage": <value>,
            "experiences_match_percentage": <value>,
            "education_match_percentage": <value>,
            "skills_match_percentage": <value>,
            "overall_trueness_percentage": <value>
        }}

        Use the data below:
        data1:
        {data1}

        data2:
        {data2}
        """

        client = Groq(api_key="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")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile"
        )

        message = chat_completion.choices[0].message.content
        return message
    except Exception as e:
        return {
            "error": f"An error occurred: {str(e)}"
        }

def extract_data(data, keys):
    def recursive_extract(data, key, path=""):
        if isinstance(data, dict):
            for k, v in data.items():
                if k == key:
                    yield (path + k, v)
                if isinstance(v, (dict, list)):
                    yield from recursive_extract(v, key, path + k + ".")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                yield from recursive_extract(item, key, path + f"[{i}].")

    extracted = {}
    for key in keys:
        matches = list(recursive_extract(data, key))
        extracted[key] = matches if len(matches) > 1 else (matches[0][1] if matches else None)

    return extracted

def extract_linkedin_url(text):
    linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/in/[\w-]+/?'
    match = re.search(linkedin_pattern, text)
    return match.group() if match else None

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only resume pdf's are supported.")
        
        with pdfplumber.open(file.file) as pdf:
            resume_extraxted_text = ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
            
        linkedin_url = extract_linkedin_url(resume_extraxted_text)

        if not linkedin_url:
            raise HTTPException(status_code=400, detail="Linkedin url is not present in resume.")
        
        if linkedin_url:
            api_key = 'xrhZ_CIZWGPnTIdQwwSFVQ'
            headers = {'Authorization': 'Bearer ' + api_key}
            api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
            params = {
                'linkedin_profile_url': linkedin_url,
                'extra': 'include',
                'github_profile_id': 'include',
                'personal_contact_number': 'include',
                'personal_email': 'include',
                'inferred_salary': 'include',
                'skills': 'include',
                'use_cache': 'if-present',
                'fallback_to_cache': 'on-error',
            }
            response = requests.get(api_endpoint,
                                    params=params,
                                    headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Unable to fetch linkedin person profile for this resume.")
            
            linkedin_data = response.json()
            with open("linkedin_data.json", "w") as file:
                json.dump(linkedin_data, file, indent=4)

            with open("linkedin_data.json", "r") as file:
                linkedin_data = json.load(file)

            keys = ["full_name", "occupation", "headline", "country_full_name", "city", "state", "experiences", "education", "skills"]
            formatted_linkedin_data = extract_data(linkedin_data, keys)
            formatted_resume_data = extract_key_info(resume_extraxted_text)

            with open("formatted_linkedin.json", "w") as file:
                json.dump(formatted_linkedin_data, file, indent=4)

            with open("resume_data.json", 'w') as file:
                json.dump(formatted_resume_data, file, indent=4)

            with open("resume_data.json", "r") as file:
                formatted_resume_data = json.load(file)
            
            with open("formatted_linkedin.json", "r") as file:
                formatted_linkedin_data = json.load(file)

            result = compare_data(formatted_linkedin_data, formatted_resume_data)

            return{
                "Resume and Linkedin profile comparison": result
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error" : str(e)})

### end

# Initialize both model pipelines
# try:
#     persona_pipe = pipeline("text-generation", 
#                           model="rohangbs/fine-tuned-model-persona", 
#                           device_map="auto")
#     brenin_pipe = pipeline("text-generation", 
#                           model="rohangbs/fine-tuned-model-brenin", 
#                           device_map="auto")
# except Exception as e:
#     print(f"Error loading models: {e}")
#     raise

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
GROQ_API_KEY = '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'
MODEL_NAME = "llama-3.3-70b-versatile"
PERSIST_DIRECTORY = 'db'
video_processor = VideoProcessor(GROQ_API_KEY)
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create persistent directory
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Define models
class Trigger(BaseModel):
    name: str
    desc: str

class ChatMessage(BaseModel):
    question: str

class DocumentInfo(BaseModel):
    document_id: str
    name: str
    upload_time: str
    analysis: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str

def generate_response(pipe, messages):
    """Helper function to generate responses using a given pipeline"""
    try:
        outputs = pipe(
            messages[-1]['content'],
            max_new_tokens=50,
            return_full_text=False
        )
        return outputs[0]['generated_text'].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        raise

# State management (in memory for demo, consider using a database for production)
documents = {}
resume_analyses = {}
trigger_faqs = {}
chat_histories = {}

# Predefined triggers
TRIGGER_OPTIONS = [
    "Time Gap Between Jobs",
    "Domain Switch",
    "Same Role for Long Duration"
]

COMMON_QUESTIONS = {
    "Time Gap Between Jobs": [
        "Can you explain the reason for the gap between jobs?",
        "What were you doing during this time?",
        "Did you pursue any personal or professional development during the gap?"
    ],
    "Domain Switch": [
        "What motivated you to switch domains?",
        "What skills from your previous domain are transferable to your new domain?",
        "Did you face any challenges while transitioning between domains?"
    ],
    "Same Role for Long Duration": [
        "What kept you in the same role for such a long period?",
        "Did you take on additional responsibilities during this time?",
        "What achievements or growth did you experience in this role?"
    ]
}

def extract_resume_details(text: str, job_description: str) -> str:
    """Extract key details from resume text based on job description using Groq"""
    chat_model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )

    prompt = f"""
    Given the following resume text and job description, extract and organize the following key details:
    1. Contact Information
    2. Years of Experience
    3. Key Skills (especially those matching the job description)
    4. Previous Companies and Roles
    5. Education
    6. Certifications
    7. Job Description Match Score (analyze how well the resume matches the requirements)

    Additionally, analyze the following trigger responses:
    - Detect gaps between jobs (e.g., more than 6 months between leaving one company and starting another)
    - Identify any major domain switches between roles
    - Highlight if the candidate worked in the same role or company for more than 5 years

    Job Description:
    {job_description}

    Resume Text:
    {text}
    """

    response = chat_model.invoke(prompt)
    return response.content

def analyze_custom_triggers(text: str, custom_triggers: List[Trigger]) -> Dict[str, List[str]]:
    """Analyze custom triggers and generate FAQs"""
    chat_model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )

    trigger_faqs = {}

    for trigger in custom_triggers:
        detection_prompt = f"""
        Analyze if the following condition exists in the resume text:
        
        Condition: {trigger.desc}
        
        Resume Text:
        {text}
        
        Respond with: "FOUND" or "NOT_FOUND"
        """

        detection_response = chat_model.invoke(detection_prompt).content.strip().upper()

        if "FOUND" in detection_response:
            question_prompt = f"""
            Generate 3-5 interview questions about this condition:
            {trigger.desc}
            
            Based on this resume text:
            {text}
            
            Format: Start each question with "Q: "
            """
            
            question_response = chat_model.invoke(question_prompt).content.strip()
            questions = [q.strip().replace('Q: ', '') for q in question_response.split('\n') 
                        if q.strip().startswith('Q:')]
            
            trigger_faqs[trigger.name] = questions
        else:
            trigger_faqs[trigger.name] = []

    return trigger_faqs

def process_document(file_path: str, job_description: str) -> tuple[bool, str]:
    """Process uploaded document and return analysis"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Generate unique ID for document
        doc_id = str(uuid.uuid4())
        
        # Analyze resume
        resume_analysis = extract_resume_details(text, job_description)
        
        # Split text for vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)

        # Create vector store
        vector_store = Chroma.from_texts(
            chunks,
            embeddings,
            persist_directory=f"{PERSIST_DIRECTORY}/{doc_id}"
        )

        # Store document info
        documents[doc_id] = {
            'name': os.path.basename(file_path),
            'path': f"{PERSIST_DIRECTORY}/{doc_id}",
            'upload_time': datetime.now().isoformat()
        }
        resume_analyses[doc_id] = resume_analysis
        chat_histories[doc_id] = []

        return True, doc_id

    except Exception as e:
        return False, str(e)

@app.post("/upload/")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """Upload and process a resume"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")

    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        success, result = process_document(temp_path, job_description)
        
        if success:
            return {"document_id": result, "message": "Resume processed successfully"}
        else:
            raise HTTPException(500, f"Error processing document: {result}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/documents/")
async def list_documents():
    """List all processed documents"""
    return [
        DocumentInfo(
            document_id=doc_id,
            name=info['name'],
            upload_time=info['upload_time'],
            analysis=resume_analyses.get(doc_id)
        )
        for doc_id, info in documents.items()
    ]

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get analysis for a specific document"""
    if document_id not in documents:
        raise HTTPException(404, "Document not found")
    
    return {
        "document_info": documents[document_id],
        "analysis": resume_analyses.get(document_id),
        "trigger_faqs": trigger_faqs.get(document_id, {})
    }

@app.post("/documents/{document_id}/analyze_triggers")
async def analyze_document_triggers(
    document_id: str,
    triggers: List[Trigger]
):
    """Analyze document with custom triggers"""
    if document_id not in documents:
        raise HTTPException(404, "Document not found")

    try:
        # Get document text from vector store
        vector_store = Chroma(
            persist_directory=f"{PERSIST_DIRECTORY}/{document_id}",
            embedding_function=embeddings
        )
        
        retrieved_documents = vector_store.get()
        if not retrieved_documents or 'documents' not in retrieved_documents:
            raise HTTPException(500, "Could not retrieve document text")
            
        text = " ".join(retrieved_documents['documents'])
        
        # Analyze triggers
        analysis_results = analyze_custom_triggers(text, triggers)
        
        # Store results
        if document_id not in trigger_faqs:
            trigger_faqs[document_id] = {}
        trigger_faqs[document_id].update(analysis_results)
        
        return analysis_results
        
    except Exception as e:
        raise HTTPException(500, f"Error analyzing triggers: {str(e)}")

@app.post("/documents/{document_id}/chat")
async def chat_with_document(
    document_id: str,
    message: ChatMessage
):
    """Chat with a document"""
    if document_id not in documents:
        raise HTTPException(404, "Document not found")

    try:
        # Load vector store
        vector_store = Chroma(
            persist_directory=f"{PERSIST_DIRECTORY}/{document_id}",
            embedding_function=embeddings
        )

        # Initialize chat model
        chat_model = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME
        )

        # Create conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

        # Get chat history
        chat_history = chat_histories.get(document_id, [])

        # Get response
        response = qa_chain({
            "question": message.question,
            "chat_history": chat_history
        })

        # Update chat history
        chat_history.append((message.question, response['answer']))
        chat_histories[document_id] = chat_history

        return {"answer": response['answer']}

    except Exception as e:
        raise HTTPException(500, f"Error processing chat: {str(e)}")

@app.get("/triggers/predefined")
async def get_predefined_triggers():
    """Get list of predefined triggers and their questions"""
    return {
        "triggers": TRIGGER_OPTIONS,
        "questions": COMMON_QUESTIONS
    }
from fastapi import Request

@app.post("/triggers_webhook/predefined")
async def analyze_predefined_triggers(request: Request):
    """Analyze predefined triggers and return their associated questions without requiring a body"""
    try:
        # Predefined triggers (use the default ones)
        predefined_trigger_names = [
            "Time Gap Between Jobs",
            "Domain Switch",
            "Same Role for Long Duration"
        ]
        
        # Filter predefined triggers and questions
        predefined_results = {}
        for trigger_name in predefined_trigger_names:
            if trigger_name in COMMON_QUESTIONS:
                predefined_results[trigger_name] = COMMON_QUESTIONS[trigger_name]
            else:
                predefined_results[trigger_name] = ["No questions available for this trigger."]
        
        # Format response message
        response_message = "Predefined Trigger Analysis Results:\n"
        for trigger_name, questions in predefined_results.items():
            response_message += f"\n{trigger_name}:\n"
            if questions:
                for i, question in enumerate(questions, 1):
                    response_message += f"{i}. {question}\n"
            else:
                response_message += "No relevant questions generated.\n"
        
        # Return response with sessionInfo
        return {
            "sessionInfo": {
                "parameters": {
                    "response_message": response_message,
                    "predefined_analysis_results": predefined_results
                }
            }
        }
    
    except Exception as e:
        return {
            "sessionInfo": {
                "parameters": {
                    "response_message": f"An error occurred while analyzing predefined triggers: {str(e)}"
                }
            }
        }


@app.post("/webhook/analyze_triggers")
async def dialogflow_webhook_analyze_triggers(request: Request):
    """Dialogflow webhook endpoint for analyzing resume triggers"""
    try:
        # Get request data from Dialogflow
        req_data = await request.json()
        
        # Extract parameters from session info
        session_info = req_data.get('sessionInfo', {})
        parameters = session_info.get('parameters', {})
        
        # Get document ID and trigger details
        document_id = parameters.get('document_id')
        trigger_name = parameters.get('trigger_name', '')
        trigger_desc = parameters.get('trigger_desc', '')
        
        # Validate document ID
        if not document_id or document_id not in documents:
            return {
                "sessionInfo": {
                    "parameters": {
                        "response_message": "Document not found. Please provide a valid document ID."
                    }
                }
            }
        
        # Validate trigger details
        if not trigger_name or not trigger_desc:
            return {
                "sessionInfo": {
                    "parameters": {
                        "response_message": "Trigger name or description is missing. Please provide valid details."
                    }
                }
            }
        
        # Create a trigger object
        formatted_trigger = Trigger(name=trigger_name, desc=trigger_desc)
        
        # Get document text from vector store
        vector_store = Chroma(
            persist_directory=f"{PERSIST_DIRECTORY}/{document_id}",
            embedding_function=embeddings
        )
        
        retrieved_documents = vector_store.get()
        if not retrieved_documents or 'documents' not in retrieved_documents:
            return {
                "sessionInfo": {
                    "parameters": {
                        "response_message": "Could not retrieve document text"
                    }
                }
            }
        
        text = " ".join(retrieved_documents['documents'])
        
        # Analyze the trigger
        analysis_results = analyze_custom_triggers(text, [formatted_trigger])
        
        # Store results
        if document_id not in trigger_faqs:
            trigger_faqs[document_id] = {}
        trigger_faqs[document_id].update(analysis_results)
        
        # Format response for Dialogflow
        response_message = "Trigger Analysis Results:\n"
        for trigger_name, questions in analysis_results.items():
            response_message += f"\n{trigger_name}:\n"
            if questions:
                for i, question in enumerate(questions, 1):
                    response_message += f"{i}. {question}\n"
            else:
                response_message += "No relevant questions generated.\n"
        
        return {
            "sessionInfo": {
                "parameters": {
                    "response_message": response_message,
                    "analysis_results": analysis_results  
                }
            }
        }
        
    except Exception as e:
        error_message = f"An error occurred while analyzing triggers: {str(e)}"
        return {
            "sessionInfo": {
                "parameters": {
                    "response_message": error_message
                }
            }
        }

def extract_important_insights(text):
    prompt = f"""
    You are an advanced AI specialized in text analysis. Analyze the following text and extract three key elements:
    
    1. **Important Points**: Identify the 5 most crucial facts, insights, or takeaways that are central to understanding the content.
    2. **Main Topics**: Identify the 3 overarching themes or subjects discussed in the text.
    3. **Crucial Statements**: Extract 3 sentences that serve as the foundation on which the rest of the content depends.

    Text:
    {text}
    """
    client = Groq(api_key="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")
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a helpful assistant that analyzes incoming data."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content.strip()

@app.post("/store-content/")
async def store_content(
    video: UploadFile = File(None),
    pdf: UploadFile = File(None)
):
    try:
        if not video and not pdf:
            raise HTTPException(status_code=400, detail="At least one of video or PDF must be uploaded.")
        
        summaries = []  

        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        if video:
            video_path = os.path.join(temp_dir, video.filename)
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            
            audio_path = video_processor.convert_video_to_audio(video_path)
            transcription = video_processor.transcribe_audio(audio_path)
            video_tags = extract_tags(transcription)
            video_summary = video_processor.summarize_with_groq(transcription)
            summaries.append(video_summary)
            
            #os.remove(video_path)
            #os.remove(audio_path)

        if pdf:
            pdf_path = os.path.join(temp_dir, pdf.filename)
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(pdf.file, buffer)
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(pdf_path)
                pdf_text = "\n".join([page.extract_text() for page in reader.pages])
                pdf_tags = extract_tags(pdf_text)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
                chunks = text_splitter.split_text(pdf_text)

                # Summarize each chunk
                chunk_summaries = []
                for chunk in chunks:
                    prompt = f"""
                    Please summarize the following text into a concise and meaningful paragraph:
                    {chunk}
                    """
                    chat_completion = video_processor.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes content."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama3-8b-8192"
                    )
                    chunk_summary = chat_completion.choices[0].message.content.strip()
                    chunk_summaries.append(chunk_summary)

                pdf_summary = "\n\n".join(chunk_summaries)
                summaries.append(pdf_summary)

            except Exception as e:
                raise RuntimeError(f"PDF processing error: {e}")

            os.remove(pdf_path)
            os.remove(video_path)

        combined_summary = "\n\n".join(summaries)
        insights = extract_important_insights(combined_summary)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(combined_summary)
        doc_id = str(uuid.uuid4())
        vector_store = Chroma.from_texts(chunks, embeddings, persist_directory=PERSIST_DIRECTORY)
        vector_store.persist()

        return JSONResponse(
            content={
                "document_id": doc_id,
                "summary": combined_summary,
                "Insights": insights,
                "sources": {
                    "video": video.filename if video else None,
                    "pdf": pdf.filename if pdf else None
                },
                "tags":{
                    "pdf_tags": pdf_tags if pdf else [],
                    "video_tags": video_tags if video else [],
                }
            },
            status_code=200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/generate_qa_webhook/")
# async def generate_qa(video_id: str = Body(...), num_questions: int = Body(5)):
#     try:
#         vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

#         docs = vector_store.similarity_search(query="summary", k=1)
#         if not docs:
#             raise HTTPException(status_code=404, detail="Summary not found for the given video_id.")
#         summary = docs[0].page_content

#         qa_text = video_processor.generate_qa_with_groq(summary, num_questions)

#         predefined_analysis_results = {}
#         sections = qa_text.strip().split("\n\n")
#         for section in sections:
#             if ": " in section:
#                 heading, questions_text = section.split(": ", 1)
#                 questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
#                 predefined_analysis_results[heading.strip()] = questions
 
#         response = {
#             "sessionInfo": {
#                 "parameters": {
#                     "response_message": f"Here are some Questions and Answers:\n\n{qa_text.strip()}",
#                     "Questions_Answers": predefined_analysis_results
#                 }
#             }
#         }

#         return JSONResponse(content=response, status_code=200)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
async def attempt_generate_qa(video_id: str, num_questions: int):
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    docs = vector_store.similarity_search(query=video_id, k=1)
    if not docs:
        raise HTTPException(status_code=404, detail=f"Summary not found for video id: {video_id}")
    
    summary = docs[0].page_content
    qa_text = video_processor.generate_qa_with_groq(summary, num_questions)
    qa_pairs = []

    sections = qa_text.strip().split("\n\n")
    for i in range(len(sections)):
        section = sections[i].strip()

        if section.startswith("**Q"):
            question = section.replace("**", "", 1).strip()
            question = re.sub(r"^\w+\s*\d*\*\*:\s*", "", question)
            if i + 1 < len(sections) and sections[i + 1].startswith("**A"):
                answer = sections[i + 1].replace("**", "", 1).strip()
                answer = re.sub(r"^\w+\*\*:\s*", "", answer)
                qa_pairs.append({
                    "question": question.strip(),
                    "answer": answer.strip()
                })
    if not qa_pairs:
        raise ValueError("No questions and answers generated.")

    return qa_pairs

@app.post("/generate-qa_webhook/")
async def generate_qa(video_id: str = Body(...), num_questions: int = Body(5)):
    try:
        try:
            qa_pairs = await attempt_generate_qa(video_id, num_questions)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Maximum number of retry completed, Please hit api again: {str(e)}")

        response = {
            "sessionInfo": {
                "parameters": {
                    "video_id": video_id,
                    "Questions_Answers": qa_pairs
                }
            }
        }
        return JSONResponse(content=response, status_code=200)

    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/generate-qa_webhook/")
# async def generate_qa(
#     video_id: str = Body(...), 
#     num_questions: int = Body(5)
# ):
#     try:
#         vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

#         docs = vector_store.similarity_search(query=video_id, k=1)
#         if not docs:
#             raise HTTPException(status_code=404, detail=f"Summary not found for video_id: {video_id}")
        
#         summary = docs[0].page_content

#         qa_text = video_processor.generate_qa_with_groq(summary, num_questions)
#         qa_pairs = []

#         sections = qa_text.strip().split("\n\n")

#         for i in range(len(sections)):
#             section = sections[i].strip()

#             if section.startswith("**Q"):
#                 question = section.replace("**Q", "", 1).strip()
#                 question = re.sub(r"^\w+\s*\d*\*\*:\s*", "", question)
#                 if i + 1 < len(sections) and sections[i + 1].startswith("**A"):
#                     answer = sections[i + 1].replace("**A", "", 1).strip()
#                     answer = re.sub(r"^\w+\*\*:\s*", "", answer)
#                     qa_pairs.append({
#                         "question": question.strip(),
#                         "answer": answer.strip()
#                     })
#         response = {
#             "sessionInfo": {
#                 "parameters": {
#                     "video_id": video_id,
#                     "Questions_Answers": qa_pairs
#                 }
#             }
#         }
#         return JSONResponse(content=response, status_code=200)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

        # predefined_analysis_results = {}
        # sections = qa_text.strip().split("\n\n")
        # for section in sections:
        #     if ": " in section:
        #         heading, questions_text = section.split(": ", 1)
        #         questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        #         predefined_analysis_results[heading.strip()] = questions

        # response = {
        #     "sessionInfo": {
        #         "parameters": {
        #             "video_id": video_id,
        #             "response_message": f"Here are some Questions and Answers:\n\n{qa_text.strip()}",
        #             "Questions_Answers": predefined_analysis_results
        #         }
        #     }
        # }

import requests

@app.post("/single-qa/")
async def generate_qa(video_id: str = Body(...), num_questions: int = Body(1)):
    """
    Generate Q&A from stored video summary and format the response as a webhook output.
    
    Args:
        video_id (str): ID of the video (used for retrieval from ChromaDB)
        num_questions (int): Number of questions to generate
    
    Returns:
        JSONResponse: Webhook-compatible output with nested fields
    """
    try:
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

        docs = vector_store.similarity_search(query="summary", k=1)
        if not docs:
            raise HTTPException(status_code=404, detail="Summary not found for the given video_id.")
        summary = docs[0].page_content

        qa_text = video_processor.generate_qa_with_groq(summary, num_questions)

        predefined_analysis_results = {}
        sections = qa_text.strip().split("\n\n")
        for section in sections:
            if ": " in section:
                heading, questions_text = section.split(": ", 1)
                questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
                predefined_analysis_results[heading.strip()] = questions

        response = {
            "sessionInfo": {
                "parameters": {
                    "response_message": f"Here are some Questions and Answers:\n\n{qa_text.strip()}",
                    "Questions_Answers": predefined_analysis_results
                }
            }
        }

        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assess-answer/")
async def assess_answer(
    reference_answer: str = Body(...), 
    user_answer: str = Body(...)
):
    """
    Assess user's answer similarity to the reference answer using Sentence-Transformers
    and format the response as a webhook-compatible output.
    
    Args:
        reference_answer (str): Original generated answer
        user_answer (str): User's submitted answer
    
    Returns:
        JSONResponse: Webhook-compatible output with similarity score and pass/fail status
    """
    try:
        similarity_threshold = 0.8

        reference_embedding = sentence_model.encode(reference_answer, convert_to_tensor=True)
        user_embedding = sentence_model.encode(user_answer, convert_to_tensor=True)
        
        cosine_sim = cosine_similarity(
            [reference_embedding.cpu().numpy()], 
            [user_embedding.cpu().numpy()]
        )[0][0]
        
        is_correct = cosine_sim >= similarity_threshold
        
        response = {
            "sessionInfo": {
                "parameters": {
                    "similarity_score": float(cosine_sim),
                    "is_correct": bool(is_correct),
                    "threshold": float(similarity_threshold)
                }
            }
        }
        
        return JSONResponse(content=response, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess-summary/")
async def assess_summary(
    video_id: str = Body(...),
    chat_history: List[Dict[str, Union[str, float]]] = Body(...)
):
    """
    Generate a summary of the user's chat assessment based on similarity scores and sentiment using Groq LLM.
    
    Args:
        video_id (str): ID of the video.
        chat_history (List[Dict[str, Union[str, float]]]): List of user responses with similarity scores.
    
    Returns:
        JSONResponse: Summary of performance, emotions, and recommendations.
    """
    try:
        if not chat_history:
            raise HTTPException(status_code=400, detail="Chat history is empty.")

        similarity_scores = []
        correct_answers = 0
        incorrect_answers = 0
        misconceptions = []

        conversation_text = ""
        for entry in chat_history:
            question = entry.get("question", "")
            user_response = entry.get("user_response", "")
            similarity = entry.get("similarity_score", 0)

            # Collect similarity scores
            similarity_scores.append(similarity)

            # Count correct/incorrect answers
            if similarity >= 0.8:
                correct_answers += 1
            else:
                incorrect_answers += 1
                misconceptions.append(user_response)

            # Append to conversation text for sentiment analysis
            conversation_text += f"{user_response}\n"

        # Compute confidence level
        avg_similarity = round(statistics.mean(similarity_scores), 2)
        confidence_level = "High" if avg_similarity > 0.8 else "Medium" if avg_similarity > 0.5 else "Low"

        # Detect improvement over time
        improvement_detected = False
        if len(similarity_scores) > 1 and similarity_scores[-1] > similarity_scores[0]:
            improvement_detected = True

        # Use Groq for sentiment analysis with a strict format
        sentiment_prompt = f"""
        Analyze the sentiment of the following responses. Only return one word: 'Positive', 'Neutral', or 'Negative'.
        Responses:
        {conversation_text}
        """
        sentiment_summary = video_processor.analyze_sentiment_with_groq(sentiment_prompt).strip()
        if sentiment_summary not in ["Positive", "Negative", "Neutral"]:
            sentiment_summary = "Neutral"

        # Format the final summary
        response = {
            "sessionInfo": {
                "parameters": {
                    "video_id": video_id,
                    "performance_summary": {
                        "average_similarity": avg_similarity,
                        "confidence_level": confidence_level,
                        "correct_answers": correct_answers,
                        "incorrect_answers": incorrect_answers,
                        "improvement_detected": improvement_detected
                    },
                    "emotional_analysis": {
                        "overall_sentiment": sentiment_summary
                    },
                    "common_misconceptions": misconceptions if misconceptions else "None detected",
                    "recommendations": "Review misunderstood concepts" if incorrect_answers > 0 else "Great progress! Keep practicing."
                }
            }
        }
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ImagePayload(BaseModel):
    image: str  

tracker = AttentionTracker()

@app.post("/analyze/")
async def analyze_attention(payload: ImagePayload):
    try:
        image_bytes = base64.b64decode(payload.image)
        image = Image.open(io.BytesIO(image_bytes))
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = tracker.process_frame(opencv_image)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws/track")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            image_bytes = base64.b64decode(data.split(',')[1] if ',' in data else data)
            image = Image.open(io.BytesIO(image_bytes))
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = tracker.process_frame(opencv_image)
            await websocket.send_json(results)
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

@app.post("/webhook/persona")
async def dialogflow_webhook_persona(request: Request):
    """Dialogflow webhook endpoint for persona model"""
    try:
        req_data = await request.json()
        print("Received request:", req_data) 
        
        query_text = req_data.get('queryInput', {}).get('text', {}).get('text', '')
        
        if not query_text:
            session_info = req_data.get('sessionInfo', {})
            parameters = session_info.get('parameters', {})
            query_text = parameters.get('query_text_persona', '')

        if not query_text:
            return {
                "fulfillmentResponse": {
                    "messages": [
                        {"text": {"text": ["No query text provided"]}}
                    ]
                }
            }

        response_text = generate_response(persona_pipe, [{"role": "user", "content": query_text}])
        
        return {
            "fulfillmentResponse": {
                "messages": [
                    {"text": {"text": [response_text]}}
                ]
            },
            "sessionInfo": {
                "parameters": {
                    "query_text_persona": query_text,
                    "response_message": response_text
                }
            }
        }
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print("Error:", error_message) 
        return {
            "fulfillmentResponse": {
                "messages": [
                    {"text": {"text": [error_message]}}
                ]
            }
        }

@app.post("/webhook/brenin")
async def dialogflow_webhook_brenin(request: Request):
    """Dialogflow webhook endpoint for brenin model"""
    try:
        req_data = await request.json()
        print("Received request:", req_data)  
        
        query_text = req_data.get('queryInput', {}).get('text', {}).get('text', '')
        
        if not query_text:
            session_info = req_data.get('sessionInfo', {})
            parameters = session_info.get('parameters', {})
            query_text = parameters.get('query_text_brenin', '')

        if not query_text:
            return {
                "fulfillmentResponse": {
                    "messages": [
                        {"text": {"text": ["No query text provided"]}}
                    ]
                }
            }

        response_text = generate_response(brenin_pipe, [{"role": "user", "content": query_text}])
        
        return {
            "fulfillmentResponse": {
                "messages": [
                    {"text": {"text": [response_text]}}
                ]
            },
            "sessionInfo": {
                "parameters": {
                    "query_text_brenin": query_text,
                    "response_message": response_text
                }
            }
        }
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print("Error:", error_message)  
        return {
            "fulfillmentResponse": {
                "messages": [
                    {"text": {"text": [error_message]}}
                ]
            }
        }
@app.get("/health")
async def health_check():
    """Health check endpoint that also verifies models are loaded"""
    return {
        "status": "healthy",
        "models": {
            "persona": "loaded" if 'persona_pipe' in globals() else "not loaded",
            "brenin": "loaded" if 'brenin_pipe' in globals() else "not loaded"
        }
    }
def run_uvicorn():
    """Run uvicorn server with SSL using subprocess"""
    command = [
        "uvicorn",
        "main_copy:app",
        "--host", "0.0.0.0",
        "--port", "8004",
        "--reload",
        "--ssl-certfile", "/etc/letsencrypt/live/llm.brenin.co/fullchain.pem",
        "--ssl-keyfile", "/etc/letsencrypt/live/llm.brenin.co/privkey.pem"
    ]
    subprocess.run(command)

if __name__ == "__main__":
    run_uvicorn()

