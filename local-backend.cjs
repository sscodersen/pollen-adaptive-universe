const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const port = 3001;

// Enable CORS
app.use(cors());
app.use(express.json());

// In-memory storage to replace Vercel KV
const storage = {
  feed: [],
  general_content: [],
  music_content: [],
  product_content: []
};

// Real Pollen AI Service
class PollenAI {
  constructor() {
    this.baseURL = process.env.POLLEN_AI_ENDPOINT || 'http://localhost:8000';
    console.log(`✨ Connecting to real Pollen AI at ${this.baseURL}`);
  }

  async generate(type, prompt, mode = 'chat') {
    try {
      // Connect to real Pollen AI service
      const response = await axios.post(`${this.baseURL}/generate`, {
        prompt,
        mode,
        type
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 10000
      });
      
      if (response.data && response.data.content) {
        console.log(`✅ Real AI response generated for ${type} mode: ${mode}`);
        return {
          content: response.data.content,
          confidence: response.data.confidence || 0.8,
          reasoning: response.data.reasoning || 'AI-generated content',
          type: type,
          mode: mode,
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error('Invalid response from Pollen AI');
      }
    } catch (error) {
      console.error(`❌ Failed to connect to Pollen AI for ${type}: ${error.message}`);
      throw new Error(`Pollen AI service unavailable: ${error.message}`);
    }
  }

  // Mock responses removed - now using real Pollen AI only
}

const pollenAI = new PollenAI();

// API Routes
app.post('/api/ai/generate', async (req, res) => {
  try {
    const { prompt, mode = 'chat', type = 'general' } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Missing "prompt" in request body' });
    }

    console.log(`Generating ${type} content for prompt: "${prompt}"`);
    
    // Generate content using Pollen AI
    const generatedData = await pollenAI.generate(type, prompt, mode);

    // Add metadata and save to storage
    const contentToStore = {
      ...generatedData,
      id: `content_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date().toISOString(),
      views: 0,
      type,
      prompt
    };

    // Save to appropriate list based on type
    const listKey = type === 'feed_post' ? 'feed' : `${type}_content`;
    storage[listKey].unshift(contentToStore);
    
    // Trim the list to keep it manageable
    if (storage[listKey].length > 100) {
      storage[listKey] = storage[listKey].slice(0, 100);
    }

    res.json(generatedData);

  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ 
      error: 'Failed to generate content',
      detail: error.message 
    });
  }
});

app.get('/api/content/feed', (req, res) => {
  try {
    const { type = 'feed', limit = 20 } = req.query;
    
    const listKey = type === 'feed' ? 'feed' : `${type}_content`;
    const data = storage[listKey].slice(0, parseInt(limit));

    res.json({ success: true, data });

  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ error: 'Failed to fetch content' });
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    storage_stats: Object.keys(storage).reduce((acc, key) => {
      acc[key] = storage[key].length;
      return acc;
    }, {})
  });
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Local Pollen AI backend running on http://0.0.0.0:${port}`);
  console.log('API endpoints:');
  console.log('  POST /api/ai/generate - Generate AI content');
  console.log('  GET /api/content/feed - Fetch content feed');
  console.log('  GET /api/health - Health check');
});