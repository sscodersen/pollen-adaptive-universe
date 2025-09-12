import { pollenAI } from '../_lib/pollenAI';
import { db } from '../../lib/server/db';
import { content, feedItems } from '../../lib/shared/schema';

export default async function handler(req, res) {
  // Production CORS - restrict to your domain in production
  const allowedOrigins = process.env.NODE_ENV === 'production' 
    ? [process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 'https://your-domain.vercel.app']
    : ['http://localhost:3000', 'http://localhost:5000', 'http://localhost:8080'];
  
  const origin = req.headers.origin;
  if (allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }
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

    // 2. Add metadata and save it to PostgreSQL database
    const contentId = `content_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Normalize type to prevent invalid storage
    const normalizedType = ['general', 'music', 'product', 'feed_post'].includes(type) ? type : 'general';
    
    // Insert content into database
    const [insertedContent] = await db.insert(content).values({
      contentId,
      content: generatedData.content,
      confidence: generatedData.confidence || 0.8,
      reasoning: generatedData.reasoning || 'AI-generated content',
      type: normalizedType,
      mode,
      prompt,
      views: 0,
      metadata: {
        timestamp: generatedData.timestamp,
        originalType: type
      }
    }).returning();

    // Also add to main feed for visibility (unless it's already a feed post)
    if (normalizedType !== 'feed_post') {
      await db.insert(feedItems).values({
        contentId: insertedContent.contentId,
        feedId: `feed_${insertedContent.contentId}`,
        featured: 0
      });
    }

    res.status(200).json(generatedData);

  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ 
      error: 'Failed to generate content',
      detail: error.message 
    });
  }
}
