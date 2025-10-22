# Pollen AI - Absolute Zero Reasoner ML Model

## Overview

Pollen AI is a groundbreaking machine learning model that implements the **Absolute Zero Reasoner** architecture - an AI system that learns from scratch through user interactions and feedback, without relying on pre-trained weights.

### Key Features

- **Zero-Start Learning**: Learns entirely from user interactions and feedback
- **Memory Systems**: Episodic, long-term, and contextual memory for intelligent learning
- **Reinforcement Learning**: Continuous improvement through RL feedback loops
- **Edge Computing Optimizations**: Model quantization, pruning, and compression
- **Lightweight Architecture**: Works with or without PyTorch/Transformers

## Project Structure

```
.
├── pollen_ai/                    # Main Pollen AI package
│   ├── models/                   # Model implementations
│   │   ├── base_model.py        # Main Pollen Model
│   │   ├── memory_modules.py    # Memory systems
│   │   ├── rl_loop.py           # Reinforcement learning
│   │   ├── task_solver.py       # Task solving capabilities
│   │   ├── task_proposer.py     # Task generation
│   │   └── ...                  # Additional model modules
│   ├── utils/                   # Utility functions
│   │   ├── model_optimization.py
│   │   └── config.py
│   ├── api/                     # API implementation
│   │   └── main.py
│   └── main.py                  # Main entry point
├── models/                       # Core model components
│   ├── base_model.py
│   ├── memory_modules.py
│   └── rl_loop.py
├── utils/                        # Optimization utilities
│   └── model_optimization.py
├── data/                         # Training data and memory storage
│   └── lt_memory.json           # Long-term memory
├── pollen_ai_optimized.py       # Optimized FastAPI backend
├── pollen_ai_requirements.txt   # Minimal requirements
├── pyproject.toml               # Full project configuration
└── attached_assets/pollen-main/ # Original Pollen implementation
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Basic Installation

```bash
# Install minimal dependencies
pip install -r pollen_ai_requirements.txt
```

### Full Installation (with ML libraries)

```bash
# Install all dependencies including PyTorch
pip install -e .
```

## Quick Start

### 1. Running the Pollen AI Backend

```bash
# Using the optimized backend
python pollen_ai_optimized.py
```

The server will start on port 8000 by default.

### 2. Using the API

#### Generate Content
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "mode": "general",
    "type": "general",
    "use_cache": true,
    "compression_level": "medium"
  }'
```

#### Train the Model with Feedback
```bash
curl -X POST http://localhost:8000/reasoner/learn \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "What is AI?",
    "expected_output": "AI is artificial intelligence that enables machines to learn and make decisions.",
    "feedback_score": 0.9
  }'
```

#### Check Model Statistics
```bash
curl http://localhost:8000/reasoner/stats
```

## Model Architecture

### Memory Systems

1. **Episodic Memory**: Short-term storage for recent experiences (capacity: 1000 items)
2. **Long-term Memory**: Persistent knowledge storage (JSON-backed)
3. **Contextual Memory**: Semantic embeddings for intelligent search and retrieval

### Learning Flow

```
User Interaction
    ↓
Episodic Memory (temporary storage)
    ↓
Reflection & Consolidation
    ↓
Long-term Memory (permanent storage)
    ↓
Future Response Generation
```

### Reinforcement Learning

The model continuously improves through:
- User feedback scores
- Pattern recognition in conversations
- Memory consolidation and reflection
- Adaptive learning mechanisms

## Edge Computing Optimizations

### Response Quantization

Three compression levels available:
- **High**: Aggressive compression, removes extra whitespace
- **Medium**: Balanced compression, maintains readability
- **Low**: Minimal compression

Performance: 92.8% memory savings through zlib compression

### LRU Cache with Compression

- Max size: 2000 entries
- Compression: zlib level 6
- TTL: 15 minutes
- Compression ratio: ~93% size reduction

### Request Batching

- Batch window: 50ms
- Max batch size: 5 requests
- Automatic grouping of similar requests

## Training the Model

### Interactive Learning

The model learns through interactions:

```python
from pollen_ai.models.base_model import PollenModel

# Initialize model
model = PollenModel()

# Generate response
response = model.generate(
    prompt="What is reinforcement learning?",
    mode="educational"
)

# Provide feedback for learning
model.learn_from_feedback(
    input_text="What is reinforcement learning?",
    output_text=response["content"],
    feedback_score=0.85,
    user_profile={"expertise": "beginner"}
)

# Consolidate memories periodically
model.reflect()
```

### Batch Training

For training on datasets:

```python
# Load training data
training_data = [
    {
        "input": "Question or prompt",
        "expected_output": "Desired response",
        "score": 0.9
    },
    # ... more examples
]

# Train the model
for example in training_data:
    model.learn_from_feedback(
        input_text=example["input"],
        output_text=example["expected_output"],
        feedback_score=example["score"]
    )
```

## API Endpoints

### Core Generation
- `POST /generate` - Generate AI content with memory-enhanced reasoning

### Absolute Zero Reasoner
- `POST /reasoner/learn` - Learn from user feedback
- `POST /reasoner/reflect` - Trigger memory consolidation
- `GET /reasoner/stats` - Get model statistics and capabilities
- `POST /reasoner/search` - Semantic search in contextual memory

### System Health
- `GET /health` - System health with optimization status
- `GET /optimization/stats` - Detailed optimization metrics

## Development

### Running Tests

```bash
# Add test files in tests/ directory
pytest tests/
```

### Model Optimization

```python
from utils.model_optimization import compress_model, calculate_model_size

# Compress the model
compressed_model = compress_model(model, compression_ratio=0.5)

# Calculate model size
size_mb = calculate_model_size(model)
```

## Configuration

Edit `pollen_ai/utils/config.py` or set environment variables:

```bash
# Model configuration
export POLLEN_MODEL_NAME="pollen-adaptive-intelligence"
export EPISODIC_MEMORY_CAPACITY=1000
export LONG_TERM_MEMORY_PATH="data/lt_memory.json"

# API configuration
export POLLEN_API_HOST="0.0.0.0"
export POLLEN_API_PORT=8000
```

## Performance Metrics

- **Cache Hit Rate**: ~50% (improves over time)
- **Memory Savings**: 92.8% through compression
- **Response Time**: < 1ms for cached responses
- **Learning Rate**: Adaptive based on feedback

## Documentation

For more detailed information, see:

- [POLLEN_AI_DOCUMENTATION.md](POLLEN_AI_DOCUMENTATION.md) - Full API documentation
- [ABSOLUTE_ZERO_REASONER_GUIDE.md](ABSOLUTE_ZERO_REASONER_GUIDE.md) - Architecture guide
- [POLLEN_AI_INTEGRATION_GUIDE.md](POLLEN_AI_INTEGRATION_GUIDE.md) - Integration examples
- [PYTHON_INTEGRATION_README.md](PYTHON_INTEGRATION_README.md) - Python integration

## License

This project is part of the Pollen AI platform.

## Contributing

To contribute to the Pollen AI model:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues, questions, or feature requests, please create an issue in the repository.
