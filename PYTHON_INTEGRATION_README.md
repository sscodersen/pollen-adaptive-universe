
# Python Script Integration Guide

This document explains how to integrate your Python automation scripts with the Lovable frontend application.

## Overview

The application is designed to work with Python scripts running on `http://localhost:8000` that handle AI automation for:
- 🎵 Music Generation (Pollen AI + ACE-Step)
- 🎮 Games Content Generation
- 🎬 Entertainment Content Generation  
- 🛍️ Smart Shop Product Generation
- 📢 Ad Creation and Optimization

## Required Python Script Endpoints

Create a Python Flask/FastAPI server with these endpoints:

### 1. Music Generation (`/generate-music`)
```python
@app.route('/generate-music', methods=['POST'])
def generate_music():
    data = request.json
    # Use Pollen AI to filter/enhance the prompt
    # Call ACE-Step Hugging Face model
    # Return generated music metadata
    return {
        "success": True,
        "data": {
            "title": "Generated Track Name",
            "artist": "AI Composer",
            "duration": "3:24",
            "audioUrl": "path/to/generated/audio.mp3",
            "genre": data.get("genre"),
            "mood": data.get("mood")
        }
    }
```

### 2. Games Content (`/generate-games`)
```python
@app.route('/generate-games', methods=['POST'])
def generate_games():
    data = request.json
    # Generate game content based on type and parameters
    return {
        "success": True,
        "data": [
            {
                "id": "game-1",
                "title": "Generated Game",
                "genre": "Action",
                "rating": 4.5,
                "players": "2.5M",
                "description": "AI-generated game description",
                # ... more game data
            }
        ]
    }
```

### 3. Entertainment Content (`/generate-entertainment`)
```python
@app.route('/generate-entertainment', methods=['POST'])
def generate_entertainment():
    data = request.json
    # Generate movies, series, documentaries, etc.
    return {
        "success": True,
        "data": [
            {
                "id": "content-1",
                "title": "Generated Content",
                "type": "Movie",
                "genre": "Sci-Fi",
                "description": "AI-generated content description",
                # ... more content data
            }
        ]
    }
```

### 4. Shop Products (`/generate-shop`)
```python
@app.route('/generate-shop', methods=['POST'])
def generate_shop():
    data = request.json
    # Generate product listings with Pollen AI filtering
    return {
        "success": True,
        "data": [
            {
                "id": "product-1",
                "name": "AI-Generated Product",
                "description": "Smart product description",
                "price": "$299",
                "category": "Technology",
                # ... more product data
            }
        ]
    }
```

### 5. Ad Generation (`/generate-ad`)
```python
@app.route('/generate-ad', methods=['POST'])
def generate_ad():
    data = request.json
    # Generate advertising content with targeting
    return {
        "success": True,
        "data": {
            "title": "Generated Ad Title",
            "description": "Compelling ad copy",
            "estimatedCTR": 4.2,
            "estimatedReach": "45K-78K",
            "costPerClick": "$0.65"
        }
    }
```

### 6. Health Check Endpoints
```python
@app.route('/health/<script_type>', methods=['GET'])
def health_check(script_type):
    return {"status": "healthy", "script": script_type}
```

## Integration with Pollen AI

Each endpoint should:
1. Receive the enhanced prompt from Pollen AI
2. Use it to generate better, more targeted content
3. Apply Pollen AI's filtering and ranking algorithms
4. Return optimized results

## Example Python Server Structure

```
python_automation/
├── app.py                 # Main Flask/FastAPI app
├── services/
│   ├── music_generator.py # ACE-Step + Pollen AI integration
│   ├── games_generator.py # Games content generation
│   ├── entertainment_generator.py
│   ├── shop_generator.py
│   └── ads_generator.py
├── models/
│   └── pollen_integration.py # Pollen AI connection
└── requirements.txt
```

## Running Your Python Scripts

1. Install dependencies: `pip install -r requirements.txt`
2. Start your server: `python app.py` or `uvicorn app:app --host 0.0.0.0 --port 8000`
3. The Lovable frontend will automatically detect and connect to your scripts
4. Python Script Status indicators will show connection status in the UI

## CORS Configuration

Make sure your Python server allows CORS for the Lovable frontend:

```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://your-lovable-domain.app"])
```

## Environment Variables

Set these in your Python environment:
- `HUGGINGFACE_API_KEY` - For ACE-Step model access
- `POLLEN_AI_ENDPOINT` - Your Pollen AI backend URL
- `OPENAI_API_KEY` - If using OpenAI models
- Any other API keys for your automation services

The Lovable frontend will call your Python scripts and display the results in the UI with enhanced automation capabilities.
