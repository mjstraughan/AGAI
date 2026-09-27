from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from bigram_model import bigram_model
import spacy

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantes who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence.",
    "we are generating text based on bigram probabilities.",
    "bigram models are simple but effective."
]

bigram_model_inst = bigram_model(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model_inst.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

nlp = spacy.load("en_core_web_lg")

def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector.tolist()

def calculate_similarity(word1, word2):
    return nlp(word1).similarity(nlp(word2))

class EmbeddingRequest(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding_vector = calculate_embedding(request.word)
    return {"word": request.word, "embedding": embedding_vector}

@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = calculate_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity_score": score}