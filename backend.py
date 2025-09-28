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


@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """Generate Python code based on prompt using enhanced templates"""

    prompt_lower = request.prompt.lower()
    max_tokens = request.max_tokens

    # Determine verbosity level based on max_tokens
    if max_tokens <= 75:
        verbosity = "minimal"
    elif max_tokens <= 150:
        verbosity = "standard"
    else:
        verbosity = "detailed"

    # Data structures
    if "list" in prompt_lower:
        if "comprehension" in prompt_lower:
            if verbosity == "minimal":
                code = "squares = [x**2 for x in range(5)]"
            elif verbosity == "standard":
                code = "# List comprehension\nsquares = [x**2 for x in range(10)]\nprint(squares)"
            else:
                code = "# List comprehension examples\nsquares = [x**2 for x in range(10)]\nfiltered = [x for x in range(20) if x % 2 == 0]\nprint(f'Squares: {squares}')\nprint(f'Even numbers: {filtered}')"
        else:
            if verbosity == "minimal":
                code = "my_list = [1, 2, 3]\nmy_list.append(4)"
            elif verbosity == "standard":
                code = "my_list = [1, 2, 3, 4, 5]\nmy_list.append(6)\nprint(my_list)"
            else:
                code = "my_list = [1, 2, 3, 4, 5]\nmy_list.append(6)\nmy_list.extend([7, 8])\nmy_list.remove(3)\nprint(f'Final list: {my_list}')\nprint(f'Length: {len(my_list)}')"

    elif "dictionary" in prompt_lower or "dict" in prompt_lower:
        if verbosity == "minimal":
            code = "my_dict = {'key': 'value'}"
        elif verbosity == "standard":
            code = "my_dict = {'name': 'John', 'age': 30}\nprint(my_dict['name'])"
        else:
            code = "my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}\nmy_dict['job'] = 'Developer'\nprint(my_dict.get('name', 'Unknown'))\nfor key, value in my_dict.items():\n    print(f'{key}: {value}')"

    # Loops
    elif "for loop" in prompt_lower:
        if verbosity == "minimal":
            code = "for i in range(5):\n    print(i)"
        elif verbosity == "standard":
            code = "for i in range(5):\n    print(f'Number: {i}')"
        else:
            code = "# Enhanced for loop with enumerate\nitems = ['apple', 'banana', 'cherry']\nfor index, item in enumerate(items):\n    print(f'{index}: {item}')\n\n# Loop with else clause\nfor i in range(3):\n    print(i)\nelse:\n    print('Loop completed normally')"

    elif "while loop" in prompt_lower:
        if verbosity == "minimal":
            code = "count = 0\nwhile count < 3:\n    print(count)\n    count += 1"
        elif verbosity == "standard":
            code = "count = 0\nwhile count < 5:\n    print(f'Count: {count}')\n    count += 1"
        else:
            code = "count = 0\nwhile count < 5:\n    print(f'Count: {count}')\n    count += 1\n    if count == 3:\n        print('Halfway there!')\nprint('Loop finished')"

    # Functions
    elif "function" in prompt_lower or "def" in prompt_lower:
        if verbosity == "minimal":
            code = "def my_func(x):\n    return x * 2"
        elif verbosity == "standard":
            code = "def calculate_sum(a, b):\n    return a + b\n\nresult = calculate_sum(5, 3)\nprint(result)"
        else:
            code = "def calculate_sum(a, b=0):\n    \"\"\"Calculate sum with default parameter\"\"\"\n    return a + b\n\ndef greet(name, greeting='Hello'):\n    \"\"\"Greet someone with custom greeting\"\"\"\n    return f'{greeting}, {name}!'\n\n# Function usage\nresult = calculate_sum(5, 3)\nprint(f'Sum: {result}')\nprint(greet('Alice'))\nprint(greet('Bob', 'Hi'))"

    # Classes
    elif "class" in prompt_lower:
        if verbosity == "minimal":
            code = "class Person:\n    def __init__(self, name):\n        self.name = name"
        elif verbosity == "standard":
            code = "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n    \n    def greet(self):\n        return f'Hello, I am {self.name}'"
        else:
            code = "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n    \n    def greet(self):\n        return f'Hello, I am {self.name}, {self.age} years old'\n    \n    def have_birthday(self):\n        self.age += 1\n        return f'Happy birthday! Now {self.age} years old'\n\nperson = Person('Alice', 25)\nprint(person.greet())\nprint(person.have_birthday())"

    # File operations
    elif "csv" in prompt_lower:
        if verbosity == "minimal":
            code = "import pandas as pd\ndf = pd.read_csv('file.csv')"
        elif verbosity == "standard":
            code = "import pandas as pd\n\ndf = pd.read_csv('data.csv')\nprint(df.head())"
        else:
            code = "import pandas as pd\n\n# Read and analyze CSV\ndf = pd.read_csv('data.csv')\nprint(f'Shape: {df.shape}')\nprint(f'Columns: {df.columns.tolist()}')\nprint('\\nFirst 5 rows:')\nprint(df.head())\nprint('\\nBasic statistics:')\nprint(df.describe())"

    elif "read file" in prompt_lower:
        if verbosity == "minimal":
            code = "with open('file.txt', 'r') as f:\n    content = f.read()"
        elif verbosity == "standard":
            code = "with open('filename.txt', 'r') as file:\n    content = file.read()\n    print(content)"
        else:
            code = "# Read file line by line\nwith open('filename.txt', 'r') as file:\n    content = file.read()\n    print(f'File size: {len(content)} characters')\n    print(content)\n\n# Read specific lines\nwith open('filename.txt', 'r') as file:\n    for line_num, line in enumerate(file, 1):\n        print(f'Line {line_num}: {line.strip()}')"

    # API requests
    elif "api" in prompt_lower or "request" in prompt_lower:
        if verbosity == "minimal":
            code = "import requests\nresponse = requests.get('https://api.example.com')"
        elif verbosity == "standard":
            code = "import requests\n\nresponse = requests.get('https://api.example.com/data')\nif response.status_code == 200:\n    data = response.json()\n    print(data)"
        else:
            code = "import requests\nimport json\n\n# GET request\nresponse = requests.get('https://jsonplaceholder.typicode.com/posts/1')\nif response.status_code == 200:\n    data = response.json()\n    print(f'Title: {data.get(\"title\", \"No title\")}')\n    print(f'Status: {response.status_code}')\n\n# POST request\npayload = {'title': 'New Post', 'body': 'Content'}\npost_response = requests.post('https://jsonplaceholder.typicode.com/posts', json=payload)\nprint(f'POST Status: {post_response.status_code}')"

    # Math operations
    elif "math" in prompt_lower or "calculate" in prompt_lower:
        if verbosity == "minimal":
            code = "import math\nresult = math.sqrt(16)"
        elif verbosity == "standard":
            code = "import math\n\nresult = math.sqrt(16)\nprint(f'Square root: {result}')\nprint(f'Pi: {math.pi:.3f}')"
        else:
            code = "import math\nimport statistics\n\n# Basic math operations\nprint(f'Square root of 16: {math.sqrt(16)}')\nprint(f'Power: {math.pow(2, 3)}')\nprint(f'Factorial: {math.factorial(5)}')\n\n# Trigonometry\nangle = math.radians(45)\nprint(f'Sin(45°): {math.sin(angle):.3f}')\nprint(f'Cos(45°): {math.cos(angle):.3f}')\n\n# Statistics\ndata = [1, 2, 3, 4, 5]\nprint(f'Mean: {statistics.mean(data)}')\nprint(f'Standard deviation: {statistics.stdev(data):.2f}')"

    # Default fallback
    else:
        if verbosity == "minimal":
            code = f"# {request.prompt}\npass"
        elif verbosity == "standard":
            code = f"# Solution for: {request.prompt}\n\ndef solve():\n    # Implementation here\n    pass\n\nsolve()"
        else:
            code = f"# Comprehensive solution for: {request.prompt}\n\ndef solve_task():\n    \"\"\"\n    Implementation for: {request.prompt}\n    \"\"\"\n    # Add your logic here\n    pass\n\nif __name__ == '__main__':\n    solve_task()\n    print('Task completed')"

    return CodeResponse(
        generated_code=code,
        prompt=request.prompt,
        success=True,
        message="Code generated successfully"
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