import { kv } from '../_lib/kv.js';

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { type = 'feed', limit = 20 } = req.query;
    
    // Fetch items from the appropriate list
    const listKey = type === 'feed' ? 'feed' : `${type}_content`;
    const feedItems = await kv.lrange(listKey, 0, parseInt(limit) - 1);
    
    // Parse the JSON strings back into objects
    const data = feedItems.map(item => {
      try {
        return JSON.parse(item);
      } catch (e) {
        console.error('Failed to parse item:', item);
        return null;
      }
    }).filter(Boolean);

    res.status(200).json({ success: true, data });

  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ error: 'Failed to fetch content' });
  }
}