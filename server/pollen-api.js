const express = require('express');
const cors = require('cors');

const app = express();

// Enable CORS for all requests
app.use(cors({
  origin: ['http://localhost:5000', 'http://localhost:8080', 'https://*.replit.dev'],
  credentials: true
}));

app.use(express.json());

// Import the Vercel functions (adapted for Express)
const generateHandler = require('../api/ai/generate.js');
const feedHandler = require('../api/content/feed.js');

// Wrapper to convert Vercel functions to Express middleware
function adaptVercelFunction(vercelHandler) {
  return async (req, res) => {
    // Create Vercel-style req/res objects
    const vercelReq = {
      method: req.method,
      body: req.body,
      query: req.query,
      headers: req.headers
    };
    
    const vercelRes = {
      status: (code) => ({
        json: (data) => res.status(code).json(data),
        end: () => res.status(code).end()
      }),
      setHeader: (name, value) => res.setHeader(name, value),
      json: (data) => res.json(data)
    };
    
    try {
      await vercelHandler.default(vercelReq, vercelRes);
    } catch (error) {
      console.error('API Error:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  };
}

// Routes
app.post('/api/ai/generate', adaptVercelFunction({ default: generateHandler.default || generateHandler }));
app.get('/api/content/feed', adaptVercelFunction({ default: feedHandler.default || feedHandler }));

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Pollen AI server running on port ${PORT}`);
});

module.exports = app;