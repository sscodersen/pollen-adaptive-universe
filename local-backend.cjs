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

// Pollen AI Mock Service
class PollenAI {
  constructor() {
    this.baseURL = process.env.POLLEN_AI_ENDPOINT || 'http://localhost:8000';
    this.apiKey = process.env.POLLEN_AI_API_KEY;
  }

  async generate(type, prompt, mode = 'chat') {
    try {
      // Try to connect to actual Pollen AI service if available
      const response = await axios.post(`${this.baseURL}/generate`, {
        type,
        prompt,
        mode
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: 5000
      });
      return response.data;
    } catch (error) {
      console.log(`Using mock response for type: ${type} - ${error.message}`);
      return this.getMockResponse(type, prompt);
    }
  }

  getMockResponse(type, prompt) {
    const responses = {
      feed_post: {
        content: `AI Breakthrough: ${prompt}. Revolutionary advances in machine learning are transforming how we approach complex problems.`,
        confidence: 0.85 + Math.random() * 0.1,
        reasoning: "Generated based on current AI research trends",
        title: `AI Innovation: ${prompt.substring(0, 50)}...`,
        industry: "technology",
        impact_level: "High",
      },
      general: {
        content: `Here's a thoughtful response to: "${prompt}". This demonstrates advanced AI reasoning and contextual understanding.`,
        confidence: 0.82 + Math.random() * 0.15,
        reasoning: "Contextual response based on query analysis",
      },
      music: {
        content: `🎵 Generated music track: "${prompt}" - An innovative composition blending electronic and acoustic elements.`,
        confidence: 0.88 + Math.random() * 0.1,
        title: `Musical Creation: ${prompt}`,
        genre: ["Electronic", "Ambient", "Jazz", "Classical"][Math.floor(Math.random() * 4)],
        mood: ["Energetic", "Relaxing", "Uplifting", "Contemplative"][Math.floor(Math.random() * 4)]
      },
      product: {
        content: `Innovative product concept: ${prompt}. A cutting-edge solution designed for modern needs.`,
        confidence: 0.90 + Math.random() * 0.08,
        name: `Smart ${prompt.split(' ')[0]} Device`,
        price: Math.floor(Math.random() * 500) + 99.99,
        category: "Smart Technology"
      }
    };
    return responses[type] || responses.general;
  }
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