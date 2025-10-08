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

// Enhanced Pollen AI Service with Fallback Content Generation
class PollenAI {
  constructor() {
    this.baseURL = process.env.POLLEN_AI_ENDPOINT || 'http://localhost:8000';
    this.fallbackEnabled = true;
    this.isPollenAIAvailable = false;
    console.log(`✨ Connecting to Pollen AI at ${this.baseURL}`);
    this.checkPollenAIHealth();
  }

  async checkPollenAIHealth() {
    try {
      const response = await axios.get(`${this.baseURL}/health`, { timeout: 3000 });
      if (response.status === 200) {
        this.isPollenAIAvailable = true;
        console.log(`✅ Pollen AI is available and healthy`);
      }
    } catch (error) {
      this.isPollenAIAvailable = false;
      console.log(`⚠️ Pollen AI not available, using fallback mode`);
    }
  }

  async generate(type, prompt, mode = 'chat') {
    // Always check health first if we haven't recently
    if (!this.isPollenAIAvailable) {
      await this.checkPollenAIHealth();
    }

    if (this.isPollenAIAvailable) {
      try {
        const response = await axios.post(`${this.baseURL}/generate`, {
          input_text: prompt,
          mode,
          type
        }, {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 10000  // Increased timeout for AI processing
        });
        
        if (response.data && response.data.content) {
          console.log(`✅ Pollen AI response generated for ${type} mode: ${mode}`);
          return {
            content: response.data.content,
            confidence: response.data.confidence || 0.8,
            reasoning: response.data.reasoning || 'AI-generated content',
            type: type,
            mode: mode,
            timestamp: response.data.timestamp || new Date().toISOString()
          };
        } else {
          throw new Error('Invalid response from Pollen AI');
        }
      } catch (error) {
        console.log(`⚠️ Pollen AI request failed: ${error.message}`);
        this.isPollenAIAvailable = false;
        // Fall through to fallback generation
      }
    }
    
    // Use fallback content generation
    console.log(`🔄 Using enhanced fallback generation for ${type}`);
    return this.generateFallbackContent(type, prompt, mode);
  }

  generateFallbackContent(type, prompt, mode) {
    const contentTemplates = {
      general: this.generateGeneralContent(prompt),
      social: this.generateSocialContent(prompt),
      music: this.generateMusicContent(prompt),
      product: this.generateProductContent(prompt),
      shop: this.generateShopContent(prompt),
      entertainment: this.generateEntertainmentContent(prompt),
      learning: this.generateLearningContent(prompt),
      trend_analysis: this.generateTrendContent(prompt)
    };

    const content = contentTemplates[type] || contentTemplates.general;
    
    return {
      content: content,
      confidence: 0.7,
      reasoning: 'Generated using enhanced fallback algorithms',
      type: type,
      mode: mode,
      timestamp: new Date().toISOString(),
      source: 'fallback'
    };
  }

  generateGeneralContent(prompt) {
    const topics = prompt.toLowerCase();
    
    if (topics.includes('ai') || topics.includes('artificial intelligence')) {
      return {
        title: "The Future of Artificial Intelligence",
        content: "Artificial Intelligence continues to evolve rapidly, transforming industries and creating new opportunities. From machine learning breakthroughs to autonomous systems, AI is reshaping how we work, learn, and interact with technology. Key developments include advanced natural language processing, computer vision improvements, and ethical AI frameworks.",
        summary: "AI technology is advancing rapidly across multiple domains, creating transformative opportunities while requiring thoughtful ethical considerations.",
        significance: 8.5
      };
    }
    
    if (topics.includes('technology') || topics.includes('tech')) {
      return {
        title: "Technology Trends Shaping Tomorrow",
        content: "Emerging technologies are creating unprecedented opportunities for innovation. Cloud computing, quantum research, and sustainable tech solutions are driving the next wave of digital transformation. These advances promise to solve complex challenges while creating new possibilities for human advancement.",
        summary: "Technology continues to advance rapidly, offering solutions to global challenges and new opportunities for innovation.",
        significance: 8.0
      };
    }

    return {
      title: "Innovative Insights and Discoveries",
      content: `Exploring ${prompt} reveals fascinating insights into current trends and future possibilities. This topic encompasses various aspects of innovation, creativity, and human progress, offering valuable perspectives for understanding our rapidly evolving world.`,
      summary: `Quality content about ${prompt} with practical insights and forward-thinking perspectives.`,
      significance: 7.5
    };
  }

  generateSocialContent(prompt) {
    return {
      title: "Trending Discussion",
      content: `💡 ${prompt}\n\nThis topic is generating meaningful conversations across communities. Share your thoughts and insights!\n\n#Innovation #Community #TrendingNow`,
      engagement: Math.floor(Math.random() * 500) + 50,
      shares: Math.floor(Math.random() * 100) + 10,
      significance: 7.8
    };
  }

  generateMusicContent(prompt) {
    return {
      title: `Music Generation: ${prompt}`,
      content: "🎵 AI-powered music composition based on your prompt. This would generate melodic patterns, harmonic progressions, and rhythmic elements tailored to your creative vision.",
      genre: "Electronic/Ambient",
      duration: "3:24",
      bpm: Math.floor(Math.random() * 60) + 80,
      significance: 8.2
    };
  }

  generateProductContent(prompt) {
    const products = [
      { name: "Smart Wireless Earbuds Pro", price: "$129.99", category: "Electronics", rating: 4.8 },
      { name: "Eco-Friendly Water Bottle", price: "$24.99", category: "Lifestyle", rating: 4.6 },
      { name: "Portable Power Bank 20000mAh", price: "$39.99", category: "Tech Accessories", rating: 4.7 },
      { name: "Premium Yoga Mat", price: "$49.99", category: "Fitness", rating: 4.9 },
      { name: "LED Desk Lamp with Wireless Charging", price: "$79.99", category: "Home Office", rating: 4.5 }
    ];
    
    const product = products[Math.floor(Math.random() * products.length)];
    
    return {
      ...product,
      description: `High-quality ${product.name.toLowerCase()} designed for modern users. Features premium materials, excellent performance, and outstanding value.`,
      inStock: true,
      trending: true,
      significance: 8.0
    };
  }

  generateShopContent(prompt) {
    return this.generateProductContent(prompt);
  }

  generateEntertainmentContent(prompt) {
    return {
      title: "Entertainment Spotlight",
      content: `🎬 Discover ${prompt} - an engaging entertainment experience that captivates audiences with innovative storytelling and exceptional quality. Perfect for those seeking meaningful entertainment with depth and creativity.`,
      category: "Featured Content",
      rating: (Math.random() * 1.5 + 3.5).toFixed(1),
      significance: 7.9
    };
  }

  generateLearningContent(prompt) {
    return {
      title: `Learning: ${prompt}`,
      content: `📚 Comprehensive learning resource covering ${prompt}. This educational content provides structured insights, practical examples, and actionable knowledge to help learners advance their understanding and skills.`,
      difficulty: ["Beginner", "Intermediate", "Advanced"][Math.floor(Math.random() * 3)],
      duration: `${Math.floor(Math.random() * 45) + 15} min`,
      category: "Professional Development",
      significance: 8.1
    };
  }

  generateTrendContent(prompt) {
    const trendingTopics = [
      "Artificial Intelligence Breakthroughs",
      "Sustainable Technology Solutions", 
      "Remote Work Innovations",
      "Digital Health Advances",
      "Clean Energy Developments",
      "Space Technology Progress",
      "Cybersecurity Enhancements",
      "Blockchain Applications"
    ];

    return {
      title: "Current Trending Topics",
      content: trendingTopics.slice(0, 4).map(topic => `• ${topic}: Significant developments and growing interest`).join('\n'),
      trends: trendingTopics.slice(0, 4),
      significance: 8.3
    };
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

    // Normalize type to prevent invalid storage keys
    const normalizedType = ['general', 'music', 'product', 'feed_post'].includes(type) ? type : 'general';
    const listKey = normalizedType === 'feed_post' ? 'feed' : `${normalizedType}_content`;
    
    // Ensure the array exists before using unshift
    if (!storage[listKey]) {
      storage[listKey] = [];
    }
    storage[listKey].unshift(contentToStore);
    
    // Also add to main feed for visibility
    if (normalizedType !== 'feed_post') {
      storage.feed.unshift({...contentToStore, id: `feed_${contentToStore.id}`});
    }
    
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
    
    // Ensure the array exists before accessing it
    if (!storage[listKey]) {
      storage[listKey] = [];
    }
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

// AI Ethics and Community Routes
storage.ethicalReports = [];
storage.biasLogs = [];
storage.communities = [];
storage.communityMembers = [];
storage.communityPosts = [];

// Ethics - Report ethical concerns
app.post('/api/ethics/reports', (req, res) => {
  const { userId, contentId, concernType, description, severity } = req.body;
  const reportId = `report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const report = {
    reportId, userId, contentId, concernType, description, severity,
    status: 'pending', createdAt: new Date().toISOString()
  };
  storage.ethicalReports.push(report);
  res.json({ success: true, report });
});

// Ethics - Get ethical reports
app.get('/api/ethics/reports', (req, res) => {
  const { userId, status } = req.query;
  let reports = storage.ethicalReports;
  if (userId) reports = reports.filter(r => r.userId === userId);
  if (status) reports = reports.filter(r => r.status === status);
  res.json({ success: true, reports: reports.slice(0, 100) });
});

// Ethics - Get transparency logs
app.get('/api/ethics/transparency', (req, res) => {
  res.json({ success: true, logs: [] });
});

// Ethics - Get bias statistics
app.get('/api/ethics/bias-stats', (req, res) => {
  res.json({
    success: true,
    stats: {
      totalDetections: storage.biasLogs.length,
      biasTypeCounts: {},
      mitigationRate: '0%',
      averageScore: 0
    }
  });
});

// Community - Get communities
app.get('/api/community/communities', (req, res) => {
  res.json({ success: true, communities: storage.communities });
});

// Community - Create community
app.post('/api/community/communities', (req, res) => {
  const { name, description, type, category, isPrivate, creatorId } = req.body;
  const communityId = `community_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const community = {
    communityId, name, description, type, category, isPrivate, creatorId,
    memberCount: 1, createdAt: new Date().toISOString()
  };
  storage.communities.push(community);
  storage.communityMembers.push({
    communityId, userId: creatorId, role: 'admin', status: 'active'
  });
  res.json({ success: true, community });
});

// Community - Join community
app.post('/api/community/:communityId/join', (req, res) => {
  const { communityId } = req.params;
  const { userId } = req.body;
  const existing = storage.communityMembers.find(
    m => m.communityId === communityId && m.userId === userId
  );
  if (existing) {
    return res.status(400).json({ error: 'Already a member' });
  }
  storage.communityMembers.push({
    communityId, userId, role: 'member', status: 'active'
  });
  const community = storage.communities.find(c => c.communityId === communityId);
  if (community) community.memberCount++;
  res.json({ success: true });
});

// Community - Get posts
app.get('/api/community/:communityId/posts', (req, res) => {
  const { communityId } = req.params;
  const posts = storage.communityPosts.filter(p => p.communityId === communityId);
  res.json({ success: true, posts });
});

// Community - Create post
app.post('/api/community/:communityId/posts', (req, res) => {
  const { communityId } = req.params;
  const { userId, content, postType } = req.body;
  const postId = `post_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const post = {
    postId, communityId, userId, content, postType: postType || 'discussion',
    likes: 0, replies: 0, createdAt: new Date().toISOString()
  };
  storage.communityPosts.push(post);
  res.json({ success: true, post });
});

// Community - User suggestions
app.get('/api/community/users/:userId/suggestions', (req, res) => {
  res.json({ success: true, suggestions: storage.communities.slice(0, 3).map(c => ({
    community: c, relevanceScore: 0.8, matchingInterests: []
  }))});
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Local Pollen AI backend running on http://0.0.0.0:${port}`);
  console.log('API endpoints:');
  console.log('  POST /api/ai/generate - Generate AI content');
  console.log('  GET /api/content/feed - Fetch content feed');
  console.log('  GET /api/health - Health check');
  console.log('  POST /api/ethics/reports - Report ethical concerns');
  console.log('  GET /api/ethics/transparency - Get AI transparency logs');
  console.log('  GET /api/ethics/bias-stats - Get bias statistics');
  console.log('  POST /api/community/communities - Create community');
  console.log('  GET /api/community/communities - Get communities');
  console.log('  POST /api/community/:id/join - Join community');
  console.log('  GET /api/community/:id/posts - Get community posts');
  console.log('  POST /api/community/:id/posts - Create community post');
});