import io
from typing import List

import spacy
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from fastapi.responses import StreamingResponse
import torchvision.transforms as T

# Assuming helper_lib is located in your src/ folder. Adjust if necessary.
from src.helper_lib.model import get_model 
from app.bigram_model import bigram_model

app = FastAPI()

# --- Initialize Models ---

# 1. Bigram Model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantes who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence.",
    "we are generating text based on bigram probabilities.",
    "bigram models are simple but effective."
]
bigram_model_inst = bigram_model(corpus)

# 2. Spacy NLP Model
nlp = spacy.load("en_core_web_lg")

# 3. CNN Image Classifier Model (Updated for CIFAR10)
cnn_model = get_model("CustomCNN")
# Load the weights saved from train.py
cnn_model.load_state_dict(torch.load("cifar10_model.pth", weights_only=True))
cnn_model.eval()

# 4. GAN MODEL
gan_model = get_model("GAN")
gan_model.load_state_dict(torch.load("gan_mnist.pth", weights_only=True))
gan_model.eval()

# --- Pydantic Request Models ---

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

# --- Helper Functions ---

def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector.tolist()

def calculate_similarity(word1, word2):
    return nlp(word1).similarity(nlp(word2))

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model_inst.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding_vector = calculate_embedding(request.word)
    return {"word": request.word, "embedding": embedding_vector}

@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = calculate_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity_score": score}

@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Define transformations to convert the image to a tensor
    transform = transforms.Compose([
     transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])
    
    input_tensor = transform(image)
    
    # Add the batch dimension to create a 4D tensor: [batch, channel, row, column]
    input_batch = input_tensor.unsqueeze(0)
    
    # Perform forward propagation to get the class prediction
    with torch.no_grad():
        output = cnn_model(input_batch)
        
    # Extract the predicted class index from the output tensor
    _, predicted_class = torch.max(output, 1)
    
    return {"predicted_class": predicted_class.item()}

@app.get("/generate-digit/")
def generate_digit():
    """Generates a random MNIST digit using the trained GAN."""
    with torch.no_grad():
        # Generate random noise vector
        z = torch.randn(1, 100)
        # Pass through generator
        fake_image_tensor = gan_model.generator(z)
        
        # Denormalize image from [-1, 1] Tanh output to [0, 1]
        fake_image_tensor = (fake_image_tensor.squeeze(0).squeeze(0) + 1.0) / 2.0
        fake_image_tensor = fake_image_tensor.clamp(0, 1)
        
        # Convert to PIL Image
        img_pil = T.ToPILImage()(fake_image_tensor)
        
        # Save to buffer and serve as streaming response
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")