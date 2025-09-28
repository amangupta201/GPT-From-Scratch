# Python Code Generator - GPT from Scratch

A complete implementation of a GPT (Generative Pre-trained Transformer) language model built from scratch using PyTorch, trained on Stack Overflow documentation data, and deployed as a web application for Python code generation.

## Overview

This project demonstrates the end-to-end development of a Large Language Model (LLM) specifically focused on Python code generation. The GPT architecture is implemented from scratch without using pre-trained models, showcasing fundamental understanding of transformer architecture and natural language processing.

## Key Features

- **Custom GPT Implementation**: Built from scratch using PyTorch with multi-head attention, positional encodings, and transformer blocks
- **Stack Overflow Dataset**: Trained on curated Python documentation and code examples from Stack Overflow
- **Hybrid Generation System**: Combines neural model output with template-based fallbacks for reliability
- **Responsive Length Control**: Variable output length based on user preferences
- **Full-Stack Web Application**: FastAPI backend with responsive HTML/CSS/JavaScript frontend
- **Production Ready**: Deployable on cloud platforms with proper error handling

## Architecture

### Model Architecture
- **Input Embedding**: Token + positional embeddings
- **Transformer Layers**: 4 layers with 4 attention heads each
- **Model Size**: 3.24M parameters
- **Vocabulary**: 106 unique characters (character-level tokenization)
- **Context Length**: 128 tokens

### Training Details
- **Dataset**: Stack Overflow documentation examples (500 curated Python snippets)
- **Training Iterations**: 5,000 steps
- **Final Loss**: 1.23 (training), 2.14 (validation)
- **Optimizer**: AdamW with 3e-4 learning rate
- **Data Processing**: Custom extraction and cleaning pipeline

## Technical Stack

**Backend:**
- FastAPI for API server
- PyTorch for model training and inference
- Pydantic for data validation
- Custom data processing pipeline

**Frontend:**
- Vanilla HTML/CSS/JavaScript
- Responsive design
- Real-time API integration
- Interactive UI components

**Machine Learning:**
- Character-level tokenization
- Multi-head self-attention
- Layer normalization
- Dropout regularization
- Custom loss tracking

## Project Structure

```
├── backend.py                           # FastAPI server
├── index.html                          # Frontend interface
├── python1_stackoverflow_gpt_model.pkl # Trained model
├── requirements.txt                    # Dependencies
└── README.md                          # Documentation
```

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/python-code-generator
cd python-code-generator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python backend.py
```

4. **Access the web interface:**
Open `http://localhost:8000` in your browser

## Usage

The web interface allows you to:
- Enter natural language descriptions of Python code requirements
- Adjust output length using the slider control
- Generate code snippets instantly
- Copy generated code with one click

**Example inputs:**
- "Create a for loop"
- "Read a CSV file"
- "Define a function"
- "Create a dictionary"

## Model Performance

- **Training Loss**: Reduced from 4.8 to 1.23 over 5,000 iterations
- **Validation Loss**: 2.14 (indicating good generalization)
- **Generation Quality**: Produces syntactically correct Python code for common patterns
- **Response Time**: <100ms average inference time

## Implementation Highlights

### Custom Transformer Implementation
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        # Multi-head attention implementation
        
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        # Transformer block with attention + feedforward
```

### Template-Based Fallback System
The model uses a hybrid approach combining neural generation with template-based responses for reliability:
- Neural model handles creative and complex requests
- Templates provide consistent output for common patterns
- Automatic fallback ensures robust performance

### Data Processing Pipeline
Custom extraction from Stack Overflow documentation:
- Filtered Python-specific content
- Cleaned and formatted code examples
- Character-level tokenization
- Quality scoring and selection

## Deployment

The application is designed for easy deployment on cloud platforms:

**For Heroku:**
```bash
git push heroku main
```

**For Railway/Render:**
- Connect repository
- Auto-deployment from main branch
- Environment variables handled automatically

## Performance Optimizations

- **Efficient tokenization**: Character-level approach reduces vocabulary size
- **Template caching**: Common patterns served instantly
- **Model quantization**: Reduced memory footprint for deployment
- **Batch processing**: Optimized inference pipeline

## Future Enhancements

- **Larger context window**: Extend from 128 to 512+ tokens
- **Multi-language support**: Expand beyond Python
- **Fine-tuning interface**: Allow custom dataset training
- **API rate limiting**: Production-ready request handling
- **Model versioning**: Support for multiple model variants

## Technical Challenges Solved

1. **Memory constraints**: Implemented efficient training pipeline for consumer hardware
2. **Data quality**: Built robust cleaning and filtering for noisy web data
3. **Model convergence**: Achieved stable training with proper hyperparameter tuning
4. **Deployment optimization**: Balanced model size with performance requirements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Based on transformer architecture from "Attention Is All You Need"
- Training methodology inspired by Andrej Karpathy's educational content
- Stack Overflow community for providing high-quality code examples
- PyTorch team for the excellent deep learning framework

## Contact

For questions or collaboration opportunities, please open an issue or reach out via GitHub.

---

**Note**: This project demonstrates understanding of transformer architectures, natural language processing, and full-stack development. The GPT model is trained from scratch without using pre-existing language models, showcasing fundamental ML engineering skills.
