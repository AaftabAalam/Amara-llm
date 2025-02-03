from fastapi import FastAPI, HTTPException, Request,UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
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
from video import VideoProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random

app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize both model pipelines
try:
    persona_pipe = pipeline("text-generation", 
                          model="rohangbs/fine-tuned-model-persona", 
                          device_map="auto")
    brenin_pipe = pipeline("text-generation", 
                          model="rohangbs/fine-tuned-model-brenin", 
                          device_map="auto")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

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
                    "analysis_results": analysis_results  # Including raw results for potential future use
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


@app.post("/store-video/")
async def store_video(video: UploadFile = File(...)):
    try:
        # Save the video
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, video.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Process video
        audio_path = video_processor.convert_video_to_audio(video_path)
        transcription = video_processor.transcribe_audio(audio_path)
        summary = video_processor.summarize_with_groq(transcription)

        # Split and store in ChromaDB
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(summary)
        doc_id = str(uuid.uuid4())
        vector_store = Chroma.from_texts(chunks, embeddings, persist_directory=PERSIST_DIRECTORY)
        vector_store.persist()

        # Clean up
        os.remove(video_path)
        os.remove(audio_path)

        return JSONResponse(content={"video_id": doc_id}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-qa_webhook/")
async def generate_qa(video_id: str = Body(...), num_questions: int = Body(5)):
    """
    Generate Q&A from stored video summary and format the response as a webhook output.
    
    Args:
        video_id (str): ID of the video (used for retrieval from ChromaDB)
        num_questions (int): Number of questions to generate
    
    Returns:
        JSONResponse: Webhook-compatible output with nested fields
    """
    try:
        # Load the vector store
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

        # Retrieve stored summary
        docs = vector_store.similarity_search(query="summary", k=1)
        if not docs:
            raise HTTPException(status_code=404, detail="Summary not found for the given video_id.")
        summary = docs[0].page_content

        # Generate Q&A
        qa_text = video_processor.generate_qa_with_groq(summary, num_questions)

        # Parse the generated Q&A into a structured format
        predefined_analysis_results = {}
        sections = qa_text.strip().split("\n\n")
        for section in sections:
            if ": " in section:
                heading, questions_text = section.split(": ", 1)
                questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
                predefined_analysis_results[heading.strip()] = questions

        # Format the output as a webhook-compatible response
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
        # Fixed similarity threshold
        similarity_threshold = 0.8
        
        # Compute embeddings
        reference_embedding = sentence_model.encode(reference_answer, convert_to_tensor=True)
        user_embedding = sentence_model.encode(user_answer, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity(
            [reference_embedding.cpu().numpy()], 
            [user_embedding.cpu().numpy()]
        )[0][0]
        
        # Determine if the answer is correct
        is_correct = cosine_sim >= similarity_threshold
        
        # Format the webhook-compatible response
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

@app.post("/fetch-random-qa/")
async def fetch_random_qa(video_id: str = Body(...), num_questions: int = Body(5)):
    """
    Fetch a random question and its corresponding answer from the dynamically 
    generated QA pairs.

    Args:
        video_id (str): Unique ID of the video or document for which QA is generated.
        num_questions (int): Number of questions to generate (passed to the QA generation API).

    Returns:
        JSONResponse: A single random question and its corresponding answer.
    """
    try:
        # Call the existing Generate QA API to get QA pairs dynamically
        qa_generation_response = await generate_qa(video_id=video_id, num_questions=num_questions)

        # Parse the response to extract questions and answers
        response_data = qa_generation_response.body.decode("utf-8")
        response_dict = json.loads(response_data)
        questions_answers_text = response_dict["sessionInfo"]["parameters"]["response_message"]

        # Extract individual questions and answers from the response
        questions_answers = []
        qa_sections = questions_answers_text.strip().split("\n\n")
        for section in qa_sections:
            if section.startswith("**Q") and "**A" in section:
                question, answer = section.split("**A:", 1)
                question = question.replace("**Q:", "").strip()
                answer = answer.strip()
                questions_answers.append({"question": question, "answer": answer})

        # Randomly select one question and answer
        if questions_answers:
            random_qa = random.choice(questions_answers)
            return JSONResponse(content=random_qa, status_code=200)
        else:
            raise HTTPException(status_code=404, detail="No questions and answers available.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/persona")
async def dialogflow_webhook_persona(request: Request):
    """Dialogflow webhook endpoint for persona model"""
    try:
        req_data = await request.json()
        print("Received request:", req_data)  # Debug log
        
        # Get the query text from the request data
        query_text = req_data.get('queryInput', {}).get('text', {}).get('text', '')
        
        # If not found in queryInput, try getting it from tags or session parameters
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

        # Generate response using persona model
        response_text = generate_response(persona_pipe, [{"role": "user", "content": query_text}])
        
        # Return response in Dialogflow format
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
        print("Error:", error_message)  # Debug log
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
        print("Received request:", req_data)  # Debug log
        
        # Get the query text from the request data
        query_text = req_data.get('queryInput', {}).get('text', {}).get('text', '')
        
        # If not found in queryInput, try getting it from tags or session parameters
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

        # Generate response using brenin model
        response_text = generate_response(brenin_pipe, [{"role": "user", "content": query_text}])
        
        # Return response in Dialogflow format
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
        print("Error:", error_message)  # Debug log
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

