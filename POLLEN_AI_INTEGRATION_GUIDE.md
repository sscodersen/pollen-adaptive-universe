# Pollen AI Platform Integration Guide

## Overview
This guide explains how to use Pollen AI across all features in the Pollen Universe platform. **All AI operations use Pollen AI exclusively** - no OpenAI, Claude, or other external AI services.

## Architecture

### AI Service Layers

```
┌─────────────────────────────────────────┐
│         User Interface (React)          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│    Unified Pollen AI Service Layer      │
│  (pollenAIUnified.ts - All AI calls)    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│        Worker Bot Task Queue            │
│    (Priority-based task management)     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   Pollen AI v3.0.0-Edge-Optimized       │
│  (Edge caching, batching, quantization) │
└─────────────────────────────────────────┘
```

### Complementary Services

**Client-Side Processing (Hugging Face Transformers)**
- Runs in user's browser
- Used for instant feedback (sentiment, classification)
- Lightweight models (distilbert, distilgpt2)
- **Does NOT replace Pollen AI** - used for quick local tasks only

**Pure Algorithms**
- InsightFlow: Content significance scoring
- TrendAggregator: RSS feed aggregation
- No AI service calls

## Using Pollen AI Across Features

### 1. Import the Unified Service

```typescript
import { pollenAI, generateWithPollenAI, enhanceWithPollenAI } from '@/services/pollenAIUnified';
```

### 2. Content Generation

#### Social Media Posts
```typescript
const post = await pollenAI.generateSocial(
  'Latest innovations in sustainable technology'
);
```

#### News Articles
```typescript
const article = await pollenAI.generateNews(
  'Breakthrough in renewable energy'
);
```

#### Entertainment Content
```typescript
const content = await pollenAI.generateEntertainment(
  'Inspiring documentary about human innovation'
);
```

#### Product Descriptions
```typescript
const product = await pollenAI.generateProduct(
  'Eco-friendly smart home device'
);
```

#### Wellness Tips
```typescript
const tip = await pollenAI.generateWellness(
  'Mental health strategies for remote workers'
);
```

### 3. Content Analysis

#### Analyze User Content
```typescript
const analysis = await pollenAI.analyze(
  userContent,
  'sentiment' // or 'quality', 'relevance', 'ethics'
);
```

#### Generate Insights
```typescript
const insights = await pollenAI.generateInsights(
  analyticsData,
  'User engagement patterns for wellness content'
);
```

### 4. Content Enhancement

```typescript
// Enhance with different styles
const professional = await pollenAI.enhanceContent(content, 'professional');
const casual = await pollenAI.enhanceContent(content, 'casual');
const technical = await pollenAI.enhanceContent(content, 'technical');
```

### 5. Personalized Recommendations

```typescript
const recommendations = await pollenAI.generateRecommendations(
  { interests: ['wellness', 'technology'], age: 30 },
  'health products'
);
// Returns: string[] of recommendations
```

### 6. Advanced Usage

#### Custom Requests
```typescript
const response = await pollenAI.generate({
  prompt: 'Analyze trends in AI ethics',
  mode: 'analysis',
  type: 'trends',
  context: { timeframe: '2024-2025', focus: 'ethics' },
  use_cache: true,
  compression_level: 'medium'
});

console.log(response.content);
console.log(`Confidence: ${response.confidence}`);
console.log(`Cached: ${response.cached}`);
console.log(`Processing: ${response.processing_time_ms}ms`);
```

#### Check Health
```typescript
const isHealthy = pollenAI.isHealthy(); // true/false
const healthStatus = await pollenAI.checkHealth();
```

#### Get Performance Stats
```typescript
const stats = await pollenAI.getOptimizationStats();
console.log('Cache hit rate:', stats.edge_computing.cache_stats.hit_rate);
console.log('Compression:', stats.edge_computing.compression_ratio);
```

#### Clear Cache
```typescript
const result = await pollenAI.clearCache();
console.log('Cache cleared:', result.status);
```

## Feature Integration Examples

### Feed Content Generation
```typescript
// In Feed component
import { pollenAI } from '@/services/pollenAIUnified';

const refreshFeed = async () => {
  const posts = await Promise.all([
    pollenAI.generateSocial('Sustainable living tips'),
    pollenAI.generateNews('Latest tech innovations'),
    pollenAI.generateWellness('Daily mindfulness practice')
  ]);
  
  setFeedContent(posts);
};
```

### Smart Shop Product Enhancement
```typescript
// Enhance product descriptions
const enhancedProduct = {
  ...product,
  description: await pollenAI.enhanceContent(
    product.description,
    'professional'
  ),
  aiInsights: await pollenAI.analyze(
    product.features,
    'feature_analysis'
  )
};
```

### Community Content Moderation
```typescript
// Analyze posts for content quality and ethics
const moderationResult = await pollenAI.analyze(
  userPost,
  'content_moderation'
);

if (moderationResult.includes('inappropriate')) {
  flagForReview(userPost);
}
```

### Health Research Insights
```typescript
// Generate insights from health data
const healthInsights = await pollenAI.generateInsights(
  {
    steps: 10000,
    sleep: 7.5,
    heartRate: 72,
    mood: 'positive'
  },
  'Wellness patterns and recommendations'
);
```

### Ethics Forum Discussion
```typescript
// Generate AI ethics analysis
const ethicsAnalysis = await pollenAI.generate({
  prompt: 'Analyze the ethical implications of AI in healthcare',
  mode: 'analysis',
  type: 'ethics',
  compression_level: 'low' // More detailed analysis
});
```

## Worker Bot Integration

For background/scheduled tasks, use the Worker Bot:

```typescript
import { workerBotClient } from '@/services/workerBotClient';

// Submit task to Worker Bot (routes to Pollen AI)
const taskId = await workerBotClient.submitTask('content', {
  prompt: 'Generate wellness content',
  type: 'wellness',
  userId: 'continuous_generation'
}, 3); // Priority 3 for background tasks
```

## Performance Optimization

### Caching Strategy
```typescript
// Enable caching for repeated requests
const cachedResponse = await pollenAI.generate({
  prompt: 'Popular wellness tips',
  mode: 'wellness',
  use_cache: true, // Enable cache
  compression_level: 'high' // Aggressive compression for repeated content
});
```

### Compression Levels

- **High**: Use for social media posts, short content
- **Medium**: Default for most content
- **Low**: Use for detailed analysis, technical content

### Batching

Pollen AI automatically batches similar requests within 50ms window. Submit multiple requests in parallel:

```typescript
// These will be batched automatically
const [social, news, wellness] = await Promise.all([
  pollenAI.generateSocial('AI trends'),
  pollenAI.generateNews('Tech news'),
  pollenAI.generateWellness('Health tip')
]);
```

## Error Handling

```typescript
try {
  const content = await pollenAI.generateSocial('AI trends');
  // Use content
} catch (error) {
  console.error('Pollen AI error:', error);
  // Fallback to cached content or show error message
  const fallback = 'Content temporarily unavailable';
}
```

## Environment Configuration

```bash
# .env or .env.local
VITE_POLLEN_AI_URL=http://localhost:8000  # Development
# VITE_POLLEN_AI_URL=https://your-pollen-ai.com  # Production
```

## Migration from Other AI Services

### From OpenAI
```typescript
// Before (OpenAI)
const completion = await openai.chat.completions.create({
  model: "gpt-4",
  messages: [{ role: "user", content: prompt }]
});

// After (Pollen AI)
const content = await generateWithPollenAI(prompt, 'chat');
```

### From Claude/Anthropic
```typescript
// Before (Claude)
const message = await anthropic.messages.create({
  model: "claude-3-opus",
  messages: [{ role: "user", content: prompt }]
});

// After (Pollen AI)
const content = await generateWithPollenAI(prompt, 'chat');
```

## Best Practices

1. **Always use pollenAI service** for AI operations
2. **Enable caching** for repeated content
3. **Choose appropriate compression** level
4. **Use Worker Bot** for background tasks
5. **Check health** before critical operations
6. **Handle errors gracefully** with fallbacks
7. **Monitor performance** via optimization stats

## API Reference

See `POLLEN_AI_DOCUMENTATION.md` for complete API reference and optimization details.

## Support

- **Health Check**: `GET http://localhost:8000/health`
- **Optimization Stats**: `GET http://localhost:8000/optimization/stats`
- **Documentation**: See `POLLEN_AI_DOCUMENTATION.md`

---

**Platform**: Pollen Universe  
**AI Engine**: Pollen AI v3.0.0-Edge-Optimized  
**Integration**: 100% Pollen AI (No OpenAI, Claude, or other services)  
**Last Updated**: October 2025
