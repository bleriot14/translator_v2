import sys
sys.path.append("..")
import os
import time
import logging
import traceback
import psutil
import httpx
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session, create_engine, select
from transformers import SeamlessM4Tv2ForTextToText, SeamlessM4TTokenizer, SeamlessM4TFeatureExtractor
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Tuple
from sentence_splitter import SentenceSplitter, split_text_into_sentences

# Configuration
MODEL_PATH = "..\\app\\aiend\\models\\seamless-m4t-v2-large"
AIEND_URL = "http://translator_aiend.havai-network:8001/translate"

# Enhanced Logging setup
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates setup
templates = Jinja2Templates(directory="templates")

# Model setup
def load_model():
    logger.info(f"Loading model from {MODEL_PATH}")
    try:
        model = SeamlessM4Tv2ForTextToText.from_pretrained(MODEL_PATH)
        tokenizer = SeamlessM4TTokenizer.from_pretrained(MODEL_PATH)
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(MODEL_PATH)
        logger.info("Model, Tokenizer, and Feature Extractor loaded successfully")
        return model, tokenizer, feature_extractor
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        logger.error(traceback.format_exc())
        raise

model, tokenizer, feature_extractor = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model = model.to(device)

# Enhanced error handling middleware
@app.middleware("http")
async def log_request_and_errors(request: Request, call_next):
    start_time = time.time()
    method = request.method
    path = request.url.path
    logger.info(f"Received {method} request for {path}")
    
    try:
        response = await call_next(request)
        process = psutil.Process(os.getpid())
        duration = time.time() - start_time
        memory = process.memory_info().rss / 1024 / 1024
        cpu = psutil.cpu_percent()
        
        logger.info(f"Request completed: path={path}, method={method}, status={response.status_code}, "
                    f"duration={duration:.2f}s, memory={memory:.2f}MB, CPU={cpu}%")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Request details: method={method}, path={path}, headers={request.headers}")
        try:
            body = await request.json()
            logger.error(f"Request body: {body}")
        except:
            logger.error("Could not parse request body")
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "error": str(e), "traceback": traceback.format_exc()}
        )

# Updated Text splitting function
def split_text(text: str, language: str = 'en') -> List[Tuple[str, int, int]]:
    sentences = split_text_into_sentences(text=text, language=language)
    chunks = []
    current_position = 0

    for sentence in sentences:
        start_char = text.index(sentence, current_position)
        end_char = start_char + len(sentence)
        if start_char != end_char:
            chunks.append((sentence, start_char, end_char))
        current_position = end_char

    return chunks

# Updated TranslationRequest model
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    language: str = "tr"  # Default language for sentence splitting

class Translation(BaseModel):
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    total_time: float
    input_prep_time: float
    translation_time: float
    decoding_time: float
    model_load_time: float
    input_token_count: int
    output_token_count: int

# Updated Translation function
async def perform_translation(request: TranslationRequest):
    logger.debug(f"Starting translation: {request}")
    start_time = time.time()
    try:
        # Split input text into sentences
        text_chunks = split_text(request.text, language=request.language)
        logger.info(f"Split input text into {len(text_chunks)} sentences")

        translated_chunks = []
        total_input_token_count = 0
        total_output_token_count = 0
        total_input_prep_time = 0
        total_translation_time = 0
        total_decoding_time = 0

        for chunk, start, end in text_chunks:
            # Prepare input
            input_prep_start = time.time()
            inputs = tokenizer(chunk, return_tensors="pt", src_lang=request.source_lang)
            input_token_count = inputs.input_ids.shape[1]
            total_input_token_count += input_token_count
            logger.info(f"Input sentence prepared. Token count: {input_token_count}")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            total_input_prep_time += time.time() - input_prep_start

            # Perform translation
            translation_start = time.time()
            with torch.no_grad():
                output_tokens = model.generate(**inputs, tgt_lang=request.target_lang)
            total_translation_time += time.time() - translation_start

            # Decode output
            decoding_start = time.time()
            translated_chunk = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=True)
            output_token_count = output_tokens.shape[1]
            total_output_token_count += output_token_count
            total_decoding_time += time.time() - decoding_start

            translated_chunks.append((translated_chunk, start, end))
            logger.info(f"Sentence translated:")
            logger.info(f"  Original: '{chunk}' (Start: {start}, End: {end})")
            logger.info(f"  Translated: '{translated_chunk}' (Output token count: {output_token_count})")

        # Combine translated chunks
        translated_text = " ".join(chunk for chunk, _, _ in translated_chunks)
        logger.info(f"Combined translated text: {translated_text}")

        total_time = time.time() - start_time
        logger.info(f"Translation completed. Total time: {total_time:.2f}s")

        return Translation(
            original_text=request.text,
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            total_time=total_time,
            input_prep_time=total_input_prep_time,
            translation_time=total_translation_time,
            decoding_time=total_decoding_time,
            model_load_time=0.0,
            input_token_count=total_input_token_count,
            output_token_count=total_output_token_count
        )
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Translation request: {request}")
        raise

# API endpoints
@app.post("/translate", response_model=Translation)
async def translate(request: TranslationRequest):
    logger.info(f"Received translation request: {request}")
    try:
        translation = await perform_translation(request)
        return translation
    except Exception as e:
        logger.error(f"An error occurred during translation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Translation request: {request}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        # Check model
        if model is None:
            raise ValueError("Model is not loaded")
        
        return {
            "status": "healthy",
            "database": "connected",
            "model_loaded": True,
            "device": str(device),
            "memory_usage": f"{psutil.virtual_memory().percent}%",
            "cpu_usage": f"{psutil.cpu_percent()}%"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Frontend route
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with frontend")
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="debug")