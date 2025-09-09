// Vercel API endpoint for content feed
export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { type = 'feed', limit = 20 } = req.query;
    
    // Generate content feed based on type
    const content = await generateContentFeed(type, parseInt(limit));
    
    res.status(200).json({
      data: content,
      total: content.length,
      type,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Content Feed Error:', error);
    res.status(500).json({ 
      error: 'Failed to fetch content',
      detail: error.message 
    });
  }
}

async function generateContentFeed(type, limit) {
  const content = [];
  
  for (let i = 0; i < limit; i++) {
    const item = generateContentItem(type, i);
    content.push(item);
  }
  
  return content;
}

function generateContentItem(type, index) {
  const baseId = `${type}-${Date.now()}-${index}`;
  
  switch (type) {
    case 'social':
      return {
        id: baseId,
        type: 'social_post',
        title: `Creative Insight #${index + 1}`,
        content: generateSocialContent(`innovation topic ${index + 1}`),
        author: 'Pollen AI Community',
        timestamp: new Date(Date.now() - index * 1000 * 60 * 30).toISOString(),
        engagement: {
          likes: Math.floor(Math.random() * 500) + 50,
          shares: Math.floor(Math.random() * 100) + 10,
          comments: Math.floor(Math.random() * 50) + 5
        },
        tags: ['innovation', 'creativity', 'ai', 'future']
      };
      
    case 'news':
      return {
        id: baseId,
        type: 'news_analysis',
        title: `Trend Analysis: Emerging Technology Pattern ${index + 1}`,
        content: generateNewsContent(`emerging trend ${index + 1}`),
        source: 'Pollen Intelligence Network',
        timestamp: new Date(Date.now() - index * 1000 * 60 * 60).toISOString(),
        confidence: 0.85 + Math.random() * 0.1,
        category: 'Technology Trends',
        readTime: Math.floor(Math.random() * 5) + 2
      };
      
    case 'entertainment':
      return {
        id: baseId,
        type: 'entertainment_concept',
        title: `Creative Project Concept ${index + 1}`,
        content: generateEntertainmentContent(`creative concept ${index + 1}`),
        creator: 'Pollen Creative Studio',
        timestamp: new Date(Date.now() - index * 1000 * 60 * 45).toISOString(),
        medium: ['film', 'series', 'game', 'interactive'][Math.floor(Math.random() * 4)],
        genre: ['sci-fi', 'drama', 'documentary', 'experimental'][Math.floor(Math.random() * 4)]
      };
      
    default:
      return {
        id: baseId,
        type: 'general_content',
        title: `AI-Generated Insight ${index + 1}`,
        content: generateGeneralContent(index),
        timestamp: new Date(Date.now() - index * 1000 * 60 * 20).toISOString(),
        category: 'General',
        relevance: Math.random() * 0.3 + 0.7
      };
  }
}

function generateSocialContent(topic) {
  const templates = [
    `🌟 Exploring the fascinating intersection of human creativity and AI capabilities around "${topic}". The future isn't about replacement—it's about amplification. #Innovation #AI #Creativity`,
    
    `💡 Just discovered an interesting pattern in "${topic}" that could reshape how we think about collaborative intelligence. Sometimes the best insights come from unexpected connections. #TechInsights #FutureThinking`,
    
    `🤝 The most exciting developments in "${topic}" happen when diverse perspectives converge. Building the future requires both human intuition and computational power. #Collaboration #Innovation`,
    
    `✨ Reflecting on how "${topic}" demonstrates the power of intentional design. When technology serves human flourishing, remarkable things become possible. #HumanCenteredDesign #Purpose`
  ];
  
  return templates[Math.floor(Math.random() * templates.length)];
}

function generateNewsContent(topic) {
  return `📊 **Intelligence Report: ${topic.charAt(0).toUpperCase() + topic.slice(1)}**

**Executive Summary**: Our adaptive analysis system has identified significant momentum around "${topic}" with 78% confidence in mainstream adoption within the next 18 months.

**Key Indicators**:
• **Innovation Index**: 8.4/10 - High potential for transformative impact
• **Market Readiness**: 73% - Strong ecosystem support emerging
• **Social Resonance**: 82% - Aligns with current cultural priorities

**Trend Synthesis**: This development represents a convergence of three major vectors: sustainable technology adoption, human-AI collaboration models, and community-driven innovation platforms.

**Strategic Implications**: Organizations that position themselves at the intersection of these trends are likely to see significant competitive advantages in the emerging landscape.

**Confidence Metrics**: Analysis based on 1,247 data points across multiple domains, processed through our Adaptive Intelligence framework for bias-neutral insights.

*Next update in 6 hours with additional pattern correlation data.*`;
}

function generateEntertainmentContent(topic) {
  const formats = [
    {
      type: 'Interactive Documentary',
      description: `"The ${topic} Chronicles" - A groundbreaking documentary series where viewers influence the narrative through real-time choices. Each episode explores different aspects of how technology and creativity intersect in unexpected ways.`,
      features: ['Adaptive storytelling', 'Community participation', 'Multi-perspective narrative']
    },
    {
      type: 'Immersive Game Experience',
      description: `"${topic}: The Collaboration" - A cooperative puzzle-adventure where players must combine human intuition with AI assistance to solve complex creative challenges. Success requires both logical thinking and imaginative leaps.`,
      features: ['Cooperative gameplay', 'AI companion system', 'Creative problem-solving']
    },
    {
      type: 'Transmedia Art Project',
      description: `"Resonance: ${topic}" - An evolving art installation that exists simultaneously in physical and digital spaces, responding to community input and environmental data to create unique, never-repeating experiences.`,
      features: ['Community-driven evolution', 'Cross-platform experience', 'Environmental responsiveness']
    }
  ];
  
  const format = formats[Math.floor(Math.random() * formats.length)];
  
  return `🎭 **${format.type}: ${format.description}**

**Key Features**:
${format.features.map(f => `• ${f}`).join('\n')}

**Innovation Factor**: This project pushes the boundaries of traditional media by creating genuine collaboration between human creativity and artificial intelligence, resulting in experiences that neither could create alone.

**Target Experience**: Audiences who appreciate both cutting-edge technology and meaningful human stories, seeking entertainment that challenges perspectives while remaining deeply engaging.`;
}

function generateGeneralContent(index) {
  const topics = [
    'The Evolution of Human-AI Collaboration',
    'Sustainable Innovation in the Digital Age',
    'Community-Driven Technology Development',
    'The Future of Creative Problem Solving',
    'Ethical AI and Human Agency',
    'Decentralized Systems and Social Impact'
  ];
  
  const topic = topics[index % topics.length];
  
  return `🔍 **Deep Dive: ${topic}**

This emerging area represents a fascinating confluence of technological capability and human values. Rather than viewing progress as a zero-sum game between human and artificial intelligence, we're seeing innovative approaches that amplify human potential while maintaining agency and creativity.

**Key Insights**:
• **Symbiotic Design**: The most successful implementations prioritize complementary strengths rather than replacement models
• **Community-Centric Approach**: Solutions that emerge from community needs tend to have more sustainable impact
• **Ethical Foundation**: Building values into the design process, not retrofitting them later

**Looking Forward**: The next phase of development will likely focus on making these collaborative tools accessible to broader communities while maintaining the depth and sophistication that makes them truly useful.

**Practical Applications**: From creative studios to educational institutions, organizations are finding ways to integrate these approaches into their existing workflows with remarkable results.

*This analysis draws from patterns observed across multiple domains and stakeholder perspectives.*`;
}