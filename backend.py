from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import uvicorn
import os

# Curated code patterns database (embed directly in backend.py)
CODE_PATTERNS = {
    "hello world function": "def hello_world():\n    return 'Hello, World!'",
    "hello world": "print('Hello, World!')",
    "add numbers function": "def add_numbers(a, b):\n    return a + b",
    "sum function": "def calculate_sum(a, b):\n    return a + b",
    "create list": "my_list = [1, 2, 3, 4, 5]",
    "read csv file": "import pandas as pd\ndf = pd.read_csv('filename.csv')",
    "write file": "with open('filename.txt', 'w') as f:\n    f.write('Hello')",
    "read file": "with open('filename.txt', 'r') as f:\n    content = f.read()",
    "for loop": "for i in range(5):\n    print(i)",
    "while loop": "count = 0\nwhile count < 5:\n    print(count)\n    count += 1",
    "dictionary": "my_dict = {'key': 'value', 'name': 'John'}",
    "class person": "class Person:\n    def __init__(self, name):\n        self.name = name",
    "try catch": "try:\n    result = risky_operation()\nexcept Exception as e:\n    print(f'Error: {e}')",
    "import pandas": "import pandas as pd\ndf = pd.DataFrame(data)",
    "api request": "import requests\nresponse = requests.get('https://api.example.com')",
}

app = FastAPI(title="Python Code Generator API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class CodeRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


class CodeResponse(BaseModel):
    generated_code: str
    prompt: str
    success: bool
    message: str = ""


# Model architecture classes (same as training script)
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PythonCodeGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=4, n_layer=4, block_size=128, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Global variables for model and tokenizers
model = None
stoi = None
itos = None
encode = None
decode = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path='python1_stackoverflow_gpt_model.pkl'):
    """Load the trained model and tokenizers"""
    global model, stoi, itos, encode, decode

    try:
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Extract model config
        config = model_data.get('model_config', {})
        vocab_size = model_data['vocab_size']
        stoi = model_data['stoi']
        itos = model_data['itos']

        # Create encode/decode functions
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join([itos[i] for i in l if i in itos])

        # Initialize model
        model = PythonCodeGPT(
            vocab_size=vocab_size,
            n_embd=config.get('n_embd', 256),
            n_head=config.get('n_head', 4),
            n_layer=config.get('n_layer', 4),
            block_size=config.get('block_size', 128),
            dropout=config.get('dropout', 0.2)
        )

        # Load trained weights
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        model.to(device)

        print(f"Model loaded successfully! Vocab size: {vocab_size}")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def clean_generated_text(text, prompt):
    """Clean and format generated text"""
    # Remove the original prompt from generated text
    if prompt in text:
        text = text.replace(prompt, "", 1)

    # Look for Q:/A: patterns and extract relevant parts
    if "A:" in text:
        parts = text.split("A:")
        if len(parts) > 1:
            text = "A:" + parts[1]

    # Clean up common artifacts
    text = text.strip()

    # Limit length and clean up
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines[:10]:  # Limit to first 10 lines
        line = line.strip()
        if len(line) > 0 and len(line) < 200:  # Skip overly long lines
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def calculate_similarity(user_tokens, pattern_tokens):
    """Calculate simple token overlap similarity"""
    if not user_tokens or not pattern_tokens:
        return 0.0

    user_set = set(user_tokens)
    pattern_set = set(pattern_tokens)

    intersection = len(user_set & pattern_set)
    union = len(user_set | pattern_set)

    return intersection / union if union > 0 else 0.0


def find_best_match(user_prompt):
    """Find best matching code pattern"""
    user_tokens = encode(user_prompt.lower())
    best_match = None
    best_score = 0
    best_pattern = ""

    for pattern, code in CODE_PATTERNS.items():
        pattern_tokens = encode(pattern.lower())
        similarity = calculate_similarity(user_tokens, pattern_tokens)

        if similarity > best_score:
            best_score = similarity
            best_match = code
            best_pattern = pattern

    # Return match if similarity is above threshold
    return (best_match, best_pattern) if best_score > 0.3 else (None, None)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("Warning: Could not load model. API will return errors.")



@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    return FileResponse('index.html')

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


# Modify your generate_code function
@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """Generate Python code using semantic search + model fallback"""

    try:
        # First, try semantic search
        matched_code, matched_pattern = find_best_match(request.prompt)

        if matched_code:
            return CodeResponse(
                generated_code=matched_code,
                prompt=request.prompt,
                success=True,
                message=f"Generated using pattern matching (matched: {matched_pattern})"
            )

        # If no good match, try your trained model
        if model is not None:
            formatted_prompt = f"Q: {request.prompt}\nA:"
            context = torch.tensor(encode(formatted_prompt), dtype=torch.long, device=device).unsqueeze(0)

            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=min(request.max_tokens, 100))
                generated_text = decode(generated[0].tolist())

            # Clean model output
            cleaned_text = clean_generated_text(generated_text, formatted_prompt)

            if len(cleaned_text.strip()) > 10:  # If model output is reasonable
                return CodeResponse(
                    generated_code=cleaned_text,
                    prompt=request.prompt,
                    success=True,
                    message="Generated using trained model"
                )

        # Final fallback
        return CodeResponse(
            generated_code=f"# Code for: {request.prompt}\n# Please provide more specific details",
            prompt=request.prompt,
            success=True,
            message="Generated fallback response"
        )

    except Exception as e:
        return CodeResponse(
            generated_code=f"# Error generating code\n# Please try again",
            prompt=request.prompt,
            success=False,
            message="Generation error"
        )
@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"error": "No model loaded"}

    return {
        "model_loaded": True,
        "device": device,
        "vocab_size": len(itos) if itos else 0,
        "model_parameters": sum(p.numel() for p in model.parameters()) if model else 0
    }

if __name__ == "__main__":
    print(f"Starting server on http://localhost:8000")
    print(f"API documentation: http://localhost:8000/docs")
    # Use PORT environment variable for deployment, fallback to 8000 for local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")