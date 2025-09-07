import { pollenAI } from '../_lib/pollenAI.js';
import { kv } from '../_lib/kv.js';

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { prompt, mode = 'chat', type = 'general' } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'Missing "prompt" in request body' });
    }

    // 1. Generate content using your AI model
    const generatedData = await pollenAI.generate(type, prompt, mode);

    // 2. Add metadata and save it to the Vercel KV database
    const contentToStore = {
      ...generatedData,
      id: `content_${Date.now()}`,
      createdAt: new Date().toISOString(),
      views: 0,
      type,
      prompt
    };

    // Save to appropriate list based on type
    const listKey = type === 'feed_post' ? 'feed' : `${type}_content`;
    await kv.lpush(listKey, JSON.stringify(contentToStore));
    
    // Trim the list to keep it from growing indefinitely
    await kv.ltrim(listKey, 0, 99);

    res.status(200).json(generatedData);

  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ 
      error: 'Failed to generate content',
      detail: error.message 
    });
  }
}
