import { db } from '../../server/db.ts';
import { content, feedItems } from '../../shared/schema.ts';
import { eq, desc, or } from 'drizzle-orm';

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

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { type = 'feed', limit = 20 } = req.query;
    const limitNum = Math.min(parseInt(limit), 100); // Cap at 100 items max
    
    let data;
    
    if (type === 'feed') {
      // Fetch all content that's in the main feed or is a feed_post
      data = await db
        .select({
          id: content.id,
          contentId: content.contentId,
          content: content.content,
          confidence: content.confidence,
          reasoning: content.reasoning,
          type: content.type,
          mode: content.mode,
          prompt: content.prompt,
          views: content.views,
          metadata: content.metadata,
          createdAt: content.createdAt
        })
        .from(content)
        .leftJoin(feedItems, eq(content.contentId, feedItems.contentId))
        .where(or(
          eq(content.type, 'feed_post'),
          eq(feedItems.contentId, content.contentId)
        ))
        .orderBy(desc(content.createdAt))
        .limit(limitNum);
    } else {
      // Fetch content by specific type
      const normalizedType = ['general', 'music', 'product', 'feed_post'].includes(type) ? type : 'general';
      data = await db
        .select()
        .from(content)
        .where(eq(content.type, normalizedType))
        .orderBy(desc(content.createdAt))
        .limit(limitNum);
    }

    // Format response to match existing API structure
    const formattedData = data.map(item => ({
      id: `content_${item.id}`, // Legacy compatibility
      contentId: item.contentId,
      content: item.content,
      confidence: item.confidence,
      reasoning: item.reasoning,
      type: item.type,
      mode: item.mode,
      prompt: item.prompt,
      views: item.views,
      timestamp: item.createdAt?.toISOString(),
      createdAt: item.createdAt?.toISOString(),
      metadata: item.metadata
    }));

    res.status(200).json({ success: true, data: formattedData });

  } catch (error) {
    console.error('API Error:', error);
    res.status(500).json({ error: 'Failed to fetch content' });
  }
}