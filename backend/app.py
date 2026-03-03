import re
import time
import torch
import uvicorn
import torch.nn as nn
from typing import List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configuration
MODEL_PATH = "models/best_next_prediction.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PORT = 8000

# Request/Response Models
class PredictionRequest(BaseModel):
    text: str
    top_k: int = 5
    temperature: float = 0.8

class PredictionResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, float]]
    input_text: str
    message: str = ""

class GenerationRequest(BaseModel):
    prompt: str
    length: int = 25
    temperature: float = 0.8
    top_k: int = 10

class GenerationResponse(BaseModel):
    success: bool
    generation_text: str
    prompt: str
    message: str = ""

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    vocab_size: int

# LSTM Model Definition
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm =  nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)

        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        return self.fc(last_hidden)

# Model Manager
class ModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.hyperparams = None
        self.seq_len = None
        self.pad_idx = 0
        self.unk_idx = 1

    def load_model(self):
        try:
            print(f"Loading model from {self.model_path}")

            checkpoint = torch.load(self.model_path, map_location=DEVICE)

            # Load vocabulary
            self.vocab = checkpoint['vocab']
            self.word_to_idx = checkpoint['word_to_idx']
            self.idx_to_word = checkpoint['idx_to_word']
            self.hyperparams = checkpoint['hyperparameters']
            self.seq_len = self.hyperparams['seq_len']

            # Initialize model
            self.model = NextWordLSTM(
                vocab_size=len(self.vocab),
                embed_dim=self.hyperparams['embed_dim'],
                hidden_dim=self.hyperparams['hidden_dim'],
                num_layers=self.hyperparams['num_layers'],
                dropout=self.hyperparams['dropout'],
                pad_idx=self.pad_idx
            )

            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(DEVICE)
            self.model.eval()

            print(f"✔ Model loaded successfully!")
            print("Device:", DEVICE)

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    # Text Clean
    def clean_text(self, text: str) -> str:
        # Step 1: Fix known line break issues before general cleaning
        text = text.replace("ten-m\nillions", "ten-millions")
        text = text.replace("tende\nr", "tender")
        text = text.replace("d\neadline", "deadline")
        text = text.replace("note\ns", "notes")
        text = text.replace("th\ne", "the")

        # Step 2: Remove line breaks & normalize whitespace
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)

        # Step 3: Fix fraction symbol
        text = text.replace('⁄', ' ')

        # Step 4: Remove currency pattern
        text = text.replace('/-', '')

        # Step 5: Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\'\.\,]', ' ', text.lower())

        # Step 6: Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    
    # Predict
    def predict_next_word(self, text: str, top_k: int = 5, temperature: float = 0.8):
        
        if self.model is None:
            raise ValueError("Model not loaded")

        temperature = max(temperature, 1e-5)

        # Clean and tokenize
        words = self.clean_text(text).split()

        if len(words) == 0:
            return []
        
        # Get last seq_len words
        seq = words[-self.seq_len:]

        # Pad if necessary
        if len(seq) < self.seq_len:
            seq = ["<pad>"] * (self.seq_len - len(seq)) + seq

        # Convert to tensor
        input_ids = torch.tensor(
            [[self.word_to_idx.get(w, self.unk_idx) for w in seq]],
            dtype=torch.long
        ).to(DEVICE)

        with torch.no_grad():

            logits = self.model(input_ids)
            logits = logits / temperature

            # Get top-k predictions
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idx = torch.topk(probs, min(top_k, len(self.vocab)), dim=-1)

            top_probs = top_probs.squeeze(0).cpu().numpy()
            top_idx = top_idx.squeeze(0).cpu().numpy()

            # Create predictions list
            predictions = [
                {
                    "word": self.idx_to_word.get(int(idx), "<unk>"),
                    "probability": float(prob)
                }
                for idx, prob in zip(top_idx, top_probs)
            ]

        return predictions

    def generate_text(self, prompt: str, length: int = 25, temperature: float = 0.8, top_k: int = 10):
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        temperature = min(max(temperature, 0.3), 1.5)

        generated = self.clean_text(prompt).split()

        for _ in range(length):

            seq = generated[-self.seq_len:]

            if len(seq) < self.seq_len:
                seq = ["<pad>"] * (self.seq_len - len(seq)) + seq

            input_ids = torch.tensor(
                [[self.word_to_idx.get(w, self.unk_idx) for w in seq]],
                dtype=torch.long
            ).to(DEVICE)

            with torch.no_grad():
                logits = self.model(input_ids)
                logits = logits / temperature

                # Top-k sampling
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_idx = torch.topk(probs, min(top_k, len(self.vocab)), dim=-1)

                top_probs = top_probs.squeeze(0)
                top_idx = top_idx.squeeze(0)

                sampled = torch.multinomial(top_probs, 1)
                next_idx = top_idx[sampled].item()

            next_word = self.idx_to_word.get(next_idx, "<unk>")

            if next_word in {"<pad>", "<unk>"}:
                break

            generated.append(next_word)

            if next_word in {'.', '!', '?'}:
                break

        return " ".join(generated)

# FASTAPI APPLICATION

app = FastAPI(
    title="Next Word Prediction API",
    description="API for predicting next words using LSTM model",
    version="1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager(MODEL_PATH)

# STARTUP EVENT
@app.on_event("startup")
async def startup_event():
    model_manager.load_model()
    print("Backend Ready")

# Root
@app.get("/")
async def root():
    return {
        "message": "Next Word Prediction API Running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "generate": "/generate"
    }

# Health
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_manager.model else "unhealthy",
        "model_loaded": model_manager.model is not None,
        "device": str(DEVICE),
        "vocab_size": len(model_manager.vocab) if model_manager.vocab else 0
    }

# Predict
@app.post("/predict")
async def predict(request: PredictionRequest):

    if not request.text.strip():
        raise HTTPException(400, "Input text empty")

    start_time = time.time()    

    predictions = model_manager.predict_next_word(
        request.text,
        request.top_k,
        request.temperature
    )

    end_time = time.time()

    return {
        "success": True,
        "predictions": predictions,
        "input_text": request.text,
        "message": "Prediction success",
        "model_inference_time_ms": round((end_time - start_time) * 1000, 2)
    }

# Generate
@app.post("/generate")
async def generate(request: GenerationRequest):

    if not request.prompt.strip():
        raise HTTPException(400, "Prompt empty")
    
    start_time = time.time()

    generated = model_manager.generate_text(
        request.prompt,
        request.length,
        request.temperature,
        request.top_k
    )

    end_time = time.time()

    return {
        "success": True,
        "generated_text": generated,
        "prompt": request.prompt,
        "message": "Generation success",
        "model_inference_time_ms": round((end_time - start_time) * 1000, 2)
    }

# Main
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )    