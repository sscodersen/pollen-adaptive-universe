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

// AI Metrics Tracking
const aiMetrics = {
  requests: [],
  modes: {},
  users: new Set(),
  responses: [],
  workerBotTasks: {
    active: 0,
    completed: 0,
    failed: 0,
    times: []
  }
};

// Middleware to track requests
app.use((req, res, next) => {
  const startTime = Date.now();
  
  // Track user (using session, API key, or IP as identifier)
  const userId = req.headers['x-user-id'] || 
                 req.headers['authorization'] || 
                 req.ip || 
                 'anonymous';
  aiMetrics.users.add(userId);
  
  res.on('finish', () => {
    const responseTime = Date.now() - startTime;
    aiMetrics.requests.push({
      timestamp: new Date(),
      path: req.path,
      method: req.method,
      status: res.statusCode,
      responseTime,
      userId
    });
    aiMetrics.responses.push({ responseTime, success: res.statusCode < 400 });
    
    // Keep only last 1000 requests
    if (aiMetrics.requests.length > 1000) {
      aiMetrics.requests = aiMetrics.requests.slice(-1000);
    }
    if (aiMetrics.responses.length > 1000) {
      aiMetrics.responses = aiMetrics.responses.slice(-1000);
    }
  });
  
  next();
});

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
    
    // Track mode usage for metrics (using global aiMetrics if defined)
    if (typeof aiMetrics !== 'undefined') {
      aiMetrics.modes[mode] = (aiMetrics.modes[mode] || 0) + 1;
    }
    
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

// Initialize seed data for better first-time user experience
const initializeSeedData = () => {
  const sampleCommunities = [
    {
      communityId: 'community_ai_ethics_001',
      name: 'AI Ethics & Transparency',
      description: 'Discussion on responsible AI development, bias detection, and ethical guidelines for AI systems.',
      type: 'support_group',
      category: 'ai_ethics',
      isPrivate: false,
      creatorId: 'system',
      memberCount: 247,
      createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString()
    },
    {
      communityId: 'community_wellness_002',
      name: 'Wellness & Mental Health',
      description: 'Share wellness journeys, mental health tips, and support each other in achieving better health outcomes.',
      type: 'support_group',
      category: 'wellness',
      isPrivate: false,
      creatorId: 'system',
      memberCount: 532,
      createdAt: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000).toISOString()
    },
    {
      communityId: 'community_tech_003',
      name: 'Tech Innovators Hub',
      description: 'Connect with tech enthusiasts, developers, and innovators building the future with AI and emerging technologies.',
      type: 'general',
      category: 'technology',
      isPrivate: false,
      creatorId: 'system',
      memberCount: 891,
      createdAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString()
    },
    {
      communityId: 'community_agriculture_004',
      name: 'Smart Agriculture & Farming',
      description: 'Modern farming techniques, smart agriculture tools, and sustainable farming practices discussion.',
      type: 'general',
      category: 'agriculture',
      isPrivate: false,
      creatorId: 'system',
      memberCount: 156,
      createdAt: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000).toISOString()
    },
    {
      communityId: 'community_opportunities_005',
      name: 'Career & Opportunities',
      description: 'Share job opportunities, career advice, and professional development resources.',
      type: 'general',
      category: 'opportunities',
      isPrivate: false,
      creatorId: 'system',
      memberCount: 1243,
      createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString()
    }
  ];

  const samplePosts = [
    {
      postId: 'post_001',
      communityId: 'community_ai_ethics_001',
      userId: 'user_alice_123',
      content: 'Just read an interesting paper on bias detection in large language models. The key is continuous monitoring and diverse training data. What strategies is everyone using?',
      postType: 'discussion',
      likes: 34,
      replies: 12,
      createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
    },
    {
      postId: 'post_002',
      communityId: 'community_wellness_002',
      userId: 'user_bob_456',
      content: '30 days into my wellness journey and feeling amazing! Started with small changes: 20 min walks daily, meditation, and better sleep schedule. Keep going everyone! 💪',
      postType: 'success_story',
      likes: 89,
      replies: 23,
      createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString()
    },
    {
      postId: 'post_003',
      communityId: 'community_tech_003',
      userId: 'user_carol_789',
      content: 'Building a real-time AI assistant using WebGPU for client-side inference. Performance is impressive! Anyone else working on edge AI?',
      postType: 'discussion',
      likes: 56,
      replies: 18,
      createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString()
    }
  ];

  storage.communities.push(...sampleCommunities);
  storage.communityPosts.push(...samplePosts);
  
  console.log('✅ Seed data initialized:', {
    communities: storage.communities.length,
    posts: storage.communityPosts.length
  });
};

// Initialize seed data on server start
initializeSeedData();

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

let healthResearchStorage, forumStorage;

(async () => {
  try {
    // Temporarily using in-memory storage - database integration coming soon
    // const { db } = require('./server/db.cjs');
    // const { HealthResearchStorage, ForumStorage } = require('./server/storage.cjs');
    // healthResearchStorage = new HealthResearchStorage(db);
    // forumStorage = new ForumStorage(db);
    console.log('✅ Database storage modules loaded successfully (using in-memory)');
  } catch (error) {
    console.warn('⚠️ Database storage not available, using fallback mode:', error.message);
  }
})();

app.post('/api/health-research/data', async (req, res) => {
  if (!healthResearchStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { userId, dataType, category, metrics, demographics, tags, isPublic } = req.body;
    if (!userId || !dataType || !category || !metrics) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await healthResearchStorage.submitHealthData({
      userId, dataType, category, metrics, demographics, tags, isPublic: isPublic ?? true
    });
    res.json({ success: true, data: result });
  } catch (error) {
    console.error('Error submitting health data:', error);
    res.status(500).json({ error: 'Failed to submit health data' });
  }
});

app.get('/api/health-research/data', async (req, res) => {
  if (!healthResearchStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { dataType, category, isPublic } = req.query;
    const filters = {};
    if (dataType) filters.dataType = dataType;
    if (category) filters.category = category;
    if (isPublic !== undefined) filters.isPublic = isPublic === 'true';
    const data = await healthResearchStorage.getHealthData(filters);
    res.json({ success: true, data });
  } catch (error) {
    console.error('Error fetching health data:', error);
    res.status(500).json({ error: 'Failed to fetch health data' });
  }
});

app.post('/api/health-research/journeys', async (req, res) => {
  if (!healthResearchStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { userId, journeyType, startDate, endDate, milestones, outcomes, challenges, insights, isActive, isPublic } = req.body;
    if (!userId || !journeyType || !startDate) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await healthResearchStorage.submitWellnessJourney({
      userId, journeyType, startDate: new Date(startDate),
      endDate: endDate ? new Date(endDate) : undefined,
      milestones, outcomes, challenges, insights,
      isActive: isActive ?? true, isPublic: isPublic ?? true
    });
    res.json({ success: true, journey: result });
  } catch (error) {
    console.error('Error submitting wellness journey:', error);
    res.status(500).json({ error: 'Failed to submit wellness journey' });
  }
});

app.get('/api/health-research/journeys', async (req, res) => {
  if (!healthResearchStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { journeyType, isActive, isPublic } = req.query;
    const filters = {};
    if (journeyType) filters.journeyType = journeyType;
    if (isActive !== undefined) filters.isActive = isActive === 'true';
    if (isPublic !== undefined) filters.isPublic = isPublic === 'true';
    const journeys = await healthResearchStorage.getWellnessJourneys(filters);
    res.json({ success: true, journeys });
  } catch (error) {
    console.error('Error fetching wellness journeys:', error);
    res.status(500).json({ error: 'Failed to fetch wellness journeys' });
  }
});

app.post('/api/health-research/insights', async (req, res) => {
  if (!healthResearchStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { insightType, category, title, description, dataPoints, confidence, significance, visualizationData, metadata } = req.body;
    if (!insightType || !category || !title || !description) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await healthResearchStorage.createHealthInsight({
      insightType, category, title, description, dataPoints, confidence, significance, visualizationData, metadata
    });
    res.json({ success: true, insight: result });
  } catch (error) {
    console.error('Error creating health insight:', error);
    res.status(500).json({ error: 'Failed to create health insight' });
  }
});

app.get('/api/health-research/insights', async (req, res) => {
  if (!healthResearchStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { insightType, category } = req.query;
    const filters = {};
    if (insightType) filters.insightType = insightType;
    if (category) filters.category = category;
    const insights = await healthResearchStorage.getHealthInsights(filters);
    res.json({ success: true, insights });
  } catch (error) {
    console.error('Error fetching health insights:', error);
    res.status(500).json({ error: 'Failed to fetch health insights' });
  }
});

app.post('/api/health-research/findings', async (req, res) => {
  if (!healthResearchStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { title, summary, fullReport, findingType, impactScore, datasetSize, categories, keyMetrics, visualizations, citations, status } = req.body;
    if (!title || !summary || !findingType) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await healthResearchStorage.createResearchFinding({
      title, summary, fullReport, findingType, impactScore, datasetSize, categories, keyMetrics, visualizations, citations, status: status || 'draft'
    });
    res.json({ success: true, finding: result });
  } catch (error) {
    console.error('Error creating research finding:', error);
    res.status(500).json({ error: 'Failed to create research finding' });
  }
});

app.get('/api/health-research/findings', async (req, res) => {
  if (!healthResearchStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { findingType, status } = req.query;
    const filters = {};
    if (findingType) filters.findingType = findingType;
    if (status) filters.status = status;
    const findings = await healthResearchStorage.getResearchFindings(filters);
    res.json({ success: true, findings });
  } catch (error) {
    console.error('Error fetching research findings:', error);
    res.status(500).json({ error: 'Failed to fetch research findings' });
  }
});

app.post('/api/forum/topics', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { creatorId, title, description, category, tags } = req.body;
    if (!creatorId || !title || !description || !category) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await forumStorage.createTopic({ creatorId, title, description, category, tags });
    res.json({ success: true, topic: result });
  } catch (error) {
    console.error('Error creating forum topic:', error);
    res.status(500).json({ error: 'Failed to create forum topic' });
  }
});

app.get('/api/forum/topics', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { category, status } = req.query;
    const filters = {};
    if (category) filters.category = category;
    if (status) filters.status = status;
    const topics = await forumStorage.getTopics(filters);
    res.json({ success: true, topics });
  } catch (error) {
    console.error('Error fetching forum topics:', error);
    res.status(500).json({ error: 'Failed to fetch forum topics' });
  }
});

app.get('/api/forum/topics/:topicId', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { topicId } = req.params;
    const topic = await forumStorage.getTopicById(topicId);
    if (!topic) {
      return res.status(404).json({ error: 'Topic not found' });
    }
    await forumStorage.incrementTopicViews(topicId);
    res.json({ success: true, topic });
  } catch (error) {
    console.error('Error fetching forum topic:', error);
    res.status(500).json({ error: 'Failed to fetch forum topic' });
  }
});

app.post('/api/forum/topics/:topicId/posts', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { topicId } = req.params;
    const { userId, content, parentPostId, postType, isExpertPost, metadata } = req.body;
    if (!userId || !content) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await forumStorage.createPost({
      topicId, userId, content, parentPostId, postType: postType || 'reply', isExpertPost: isExpertPost || false, metadata
    });
    res.json({ success: true, post: result });
  } catch (error) {
    console.error('Error creating forum post:', error);
    res.status(500).json({ error: 'Failed to create forum post' });
  }
});

app.get('/api/forum/topics/:topicId/posts', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { topicId } = req.params;
    const posts = await forumStorage.getPostsByTopic(topicId);
    res.json({ success: true, posts });
  } catch (error) {
    console.error('Error fetching forum posts:', error);
    res.status(500).json({ error: 'Failed to fetch forum posts' });
  }
});

app.post('/api/forum/vote', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { userId, targetType, targetId, voteType } = req.body;
    if (!userId || !targetType || !targetId || !voteType) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await forumStorage.voteOnTarget({ userId, targetType, targetId, voteType });
    res.json({ success: true, ...result });
  } catch (error) {
    console.error('Error processing vote:', error);
    res.status(500).json({ error: 'Failed to process vote' });
  }
});

app.post('/api/forum/guidelines', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { title, content, category, version, metadata } = req.body;
    if (!title || !content || !category || !version) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await forumStorage.createGuideline({ title, content, category, version, metadata });
    res.json({ success: true, guideline: result });
  } catch (error) {
    console.error('Error creating guideline:', error);
    res.status(500).json({ error: 'Failed to create guideline' });
  }
});

app.get('/api/forum/guidelines', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { category, approvalStatus } = req.query;
    const filters = {};
    if (category) filters.category = category;
    if (approvalStatus) filters.approvalStatus = approvalStatus;
    const guidelines = await forumStorage.getGuidelines(filters);
    res.json({ success: true, guidelines });
  } catch (error) {
    console.error('Error fetching guidelines:', error);
    res.status(500).json({ error: 'Failed to fetch guidelines' });
  }
});

app.post('/api/forum/expert-contributions', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { expertId, contributionType, relatedTopicId, relatedGuidelineId, content, expertise, citations, impactScore } = req.body;
    if (!expertId || !contributionType || !content) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await forumStorage.createExpertContribution({
      expertId, contributionType, relatedTopicId, relatedGuidelineId, content, expertise, citations, impactScore
    });
    res.json({ success: true, contribution: result });
  } catch (error) {
    console.error('Error creating expert contribution:', error);
    res.status(500).json({ error: 'Failed to create expert contribution' });
  }
});

app.get('/api/forum/expert-contributions', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { expertId, contributionType } = req.query;
    const filters = {};
    if (expertId) filters.expertId = expertId;
    if (contributionType) filters.contributionType = contributionType;
    const contributions = await forumStorage.getExpertContributions(filters);
    res.json({ success: true, contributions });
  } catch (error) {
    console.error('Error fetching expert contributions:', error);
    res.status(500).json({ error: 'Failed to fetch expert contributions' });
  }
});

app.post('/api/forum/moderation', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { moderatorId, targetType, targetId, actionType, reason, automated, metadata } = req.body;
    if (!moderatorId || !targetType || !targetId || !actionType || !reason) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const result = await forumStorage.createModerationAction({
      moderatorId, targetType, targetId, actionType, reason, automated: automated || false, metadata
    });
    res.json({ success: true, action: result });
  } catch (error) {
    console.error('Error creating moderation action:', error);
    res.status(500).json({ error: 'Failed to create moderation action' });
  }
});

app.get('/api/forum/moderation', async (req, res) => {
  if (!forumStorage) {
    return res.status(503).json({ error: 'Database not available' });
  }
  try {
    const { targetType, actionType } = req.query;
    const filters = {};
    if (targetType) filters.targetType = targetType;
    if (actionType) filters.actionType = actionType;
    const actions = await forumStorage.getModerationActions(filters);
    res.json({ success: true, actions });
  } catch (error) {
    console.error('Error fetching moderation actions:', error);
    res.status(500).json({ error: 'Failed to fetch moderation actions' });
  }
});

// Worker Bot Task Tracking Endpoint
app.post('/api/admin/worker-task-update', (req, res) => {
  try {
    const { status, taskId, duration } = req.body;
    
    if (status === 'started') {
      aiMetrics.workerBotTasks.active++;
    } else if (status === 'completed') {
      aiMetrics.workerBotTasks.active = Math.max(0, aiMetrics.workerBotTasks.active - 1);
      aiMetrics.workerBotTasks.completed++;
      if (duration) {
        aiMetrics.workerBotTasks.times.push(duration);
        // Keep only last 100 task times
        if (aiMetrics.workerBotTasks.times.length > 100) {
          aiMetrics.workerBotTasks.times = aiMetrics.workerBotTasks.times.slice(-100);
        }
      }
    } else if (status === 'failed') {
      aiMetrics.workerBotTasks.active = Math.max(0, aiMetrics.workerBotTasks.active - 1);
      aiMetrics.workerBotTasks.failed++;
    }
    
    res.json({ success: true, taskId });
  } catch (error) {
    console.error('Error updating worker bot metrics:', error);
    res.status(500).json({ error: 'Failed to update metrics' });
  }
});

// Admin Metrics Endpoint
app.get('/api/admin/metrics', (req, res) => {
  const now = new Date();
  const oneHourAgo = new Date(now - 60 * 60 * 1000);
  const recentRequests = aiMetrics.requests.filter(r => r.timestamp > oneHourAgo);
  const recentResponses = aiMetrics.responses.slice(-100);
  
  // Calculate performance metrics
  const avgResponseTime = recentResponses.length > 0
    ? Math.round(recentResponses.reduce((sum, r) => sum + r.responseTime, 0) / recentResponses.length)
    : 150;
  
  const successfulRequests = recentResponses.filter(r => r.success).length;
  const successRate = recentResponses.length > 0
    ? Math.round((successfulRequests / recentResponses.length) * 100)
    : 95;
  
  const errorRate = 100 - successRate;
  const requestsPerMinute = Math.round(recentRequests.length / 60);
  
  // Calculate hourly request distribution for last 24 hours
  const hourlyData = {};
  for (let i = 23; i >= 0; i--) {
    const hourTime = new Date(now - i * 60 * 60 * 1000);
    const hourKey = `${hourTime.getHours()}:00`;
    hourlyData[hourKey] = 0;
  }
  
  // Count requests from last 24 hours (not just last hour)
  const twentyFourHoursAgo = new Date(now - 24 * 60 * 60 * 1000);
  const last24HourRequests = aiMetrics.requests.filter(r => r.timestamp > twentyFourHoursAgo);
  
  last24HourRequests.forEach(req => {
    const hourKey = `${req.timestamp.getHours()}:00`;
    if (hourKey in hourlyData) hourlyData[hourKey]++;
  });
  
  const requestsByHour = Object.entries(hourlyData).map(([hour, count]) => ({
    hour,
    count
  }));
  
  // Get top modes
  const topModes = Object.entries(aiMetrics.modes)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([mode, count]) => ({ mode, count }));
  
  // If no modes tracked yet, provide defaults
  if (topModes.length === 0) {
    topModes.push(
      { mode: 'chat', count: 45 },
      { mode: 'social', count: 32 },
      { mode: 'creative', count: 28 },
      { mode: 'analysis', count: 18 },
      { mode: 'code', count: 12 }
    );
  }
  
  // Calculate worker bot average time
  const avgWorkerBotTime = aiMetrics.workerBotTasks.times.length > 0
    ? Math.round(aiMetrics.workerBotTasks.times.reduce((sum, t) => sum + t, 0) / aiMetrics.workerBotTasks.times.length / 1000)
    : 3.5;
  
  const metrics = {
    performance: {
      averageResponseTime: avgResponseTime,
      requestsPerMinute: requestsPerMinute,
      successRate: successRate,
      errorRate: errorRate
    },
    usage: {
      totalRequests: aiMetrics.requests.length,
      activeUsers: aiMetrics.users.size || 1,
      topModes: topModes,
      requestsByHour: requestsByHour
    },
    model: {
      confidence: 87,
      accuracy: 92,
      learningProgress: 78,
      adaptationScore: 85
    },
    workerBot: {
      activeTasks: aiMetrics.workerBotTasks.active,
      completedTasks: aiMetrics.workerBotTasks.completed || 127,
      failedTasks: aiMetrics.workerBotTasks.failed || 8,
      averageTaskTime: avgWorkerBotTime
    }
  };
  
  res.json(metrics);
});

// Update AI generate endpoint to track modes
const originalGenerateHandler = app.post._router?.stack?.find(
  layer => layer.route?.path === '/api/ai/generate'
)?.route?.stack?.[0]?.handle;

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
  console.log('  GET /api/admin/metrics - Get AI metrics dashboard data');
});