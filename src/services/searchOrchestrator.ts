import type { SearchIntent, SearchMetadata, SearchResult, SearchResponse } from '@/types/search';
import pollenAIUnified from './pollenAIUnified';
import { trendScraperSSE } from './trendScraperSSE';
import { wellnessContentService } from './wellnessContent';
import { realWeatherService } from './realWeatherService';

class SearchOrchestrator {
  private cache: Map<string, { data: SearchResponse; timestamp: number }> = new Map();
  private readonly CACHE_TTL = 5 * 60 * 1000;

  async search(query: string, sessionId?: string): Promise<SearchResponse> {
    const cacheKey = `${query.toLowerCase()}_${sessionId || 'anon'}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      return cached.data;
    }

    const intents = this.parseIntent(query);
    const metadata: SearchMetadata = {
      query,
      intent: intents,
      timestamp: Date.now(),
      sessionId
    };

    const results = await this.executeSearch(query, intents, metadata);
    const response: SearchResponse = {
      results: this.rankResults(results),
      metadata,
      totalResults: results.length
    };

    this.cache.set(cacheKey, { data: response, timestamp: Date.now() });
    return response;
  }

  private parseIntent(query: string): SearchIntent[] {
    const lowercaseQuery = query.toLowerCase();
    const intents: SearchIntent[] = [];

    const intentKeywords: Record<SearchIntent, string[]> = {
      news: ['news', 'article', 'headline', 'current events', 'breaking'],
      trends: ['trend', 'trending', 'popular', 'viral', 'hot', 'whats happening'],
      wellness: ['health', 'wellness', 'fitness', 'mental', 'meditation', 'yoga', 'nutrition'],
      shopping: ['buy', 'shop', 'product', 'price', 'deal', 'discount', 'purchase', 'app'],
      music: ['music', 'song', 'audio', 'playlist', 'artist', 'album', 'listen'],
      media: ['image', 'video', 'photo', 'picture', 'visual', 'graphic'],
      entertainment: ['movie', 'tv', 'show', 'game', 'film', 'series', 'entertainment'],
      education: ['learn', 'education', 'course', 'tutorial', 'how to', 'study', 'teach'],
      tools: ['ai detector', 'crop analyzer', 'analyze', 'detect', 'tool'],
      smart_home: ['smart home', 'device', 'light', 'thermostat', 'automation', 'home control'],
      robot: ['robot', 'robotic', 'automation', 'drone', 'fleet'],
      assistant: ['weather', 'time', 'location', 'where', 'when', 'what is'],
      content: ['post', 'feed', 'content', 'social'],
      general: []
    };

    Object.entries(intentKeywords).forEach(([intent, keywords]) => {
      if (keywords.some(keyword => lowercaseQuery.includes(keyword))) {
        intents.push(intent as SearchIntent);
      }
    });

    if (intents.length === 0) {
      intents.push('general', 'content');
    }

    return intents;
  }

  private async executeSearch(
    query: string,
    intents: SearchIntent[],
    metadata: SearchMetadata
  ): Promise<SearchResult[]> {
    const results: SearchResult[] = [];
    const promises: Promise<void>[] = [];

    if (intents.includes('content') || intents.includes('general')) {
      promises.push(this.fetchContentResults(query, results));
    }

    if (intents.includes('news') || intents.includes('trends')) {
      promises.push(this.fetchNewsAndTrends(query, results));
    }

    if (intents.includes('wellness')) {
      promises.push(this.fetchWellnessResults(query, results));
    }

    if (intents.includes('assistant')) {
      promises.push(this.fetchAssistantResults(query, results));
    }

    if (intents.includes('shopping')) {
      promises.push(this.fetchShoppingResults(query, results));
    }

    if (intents.includes('music') || intents.includes('media')) {
      promises.push(this.fetchMediaResults(query, results));
    }

    if (intents.includes('entertainment') || intents.includes('education')) {
      promises.push(this.fetchEntertainmentResults(query, results));
    }

    await Promise.allSettled(promises);

    return results;
  }

  private async fetchContentResults(query: string, results: SearchResult[]): Promise<void> {
    try {
      const response = await pollenAIUnified.generateContent({
        type: 'feed_post',
        context: query,
        mood: 'informative'
      });

      if (response.success) {
        results.push({
          id: `content_${Date.now()}`,
          type: 'content',
          title: 'AI-Generated Content',
          description: response.content.substring(0, 150) + '...',
          content: response.content,
          score: 0.9
        });
      }
    } catch (error) {
      console.error('Content generation error:', error);
    }
  }

  private async fetchNewsAndTrends(query: string, results: SearchResult[]): Promise<void> {
    try {
      const trends = await trendScraperSSE.getLatestTrends();
      
      trends.slice(0, 5).forEach((trend, index) => {
        results.push({
          id: `trend_${trend.id}`,
          type: 'trends',
          title: trend.title,
          description: trend.description || 'Trending topic',
          content: trend,
          metadata: {
            category: trend.category,
            growth: 0,
            volume: 0
          },
          score: 0.8 - (index * 0.05)
        });
      });
    } catch (error) {
      console.error('Trends fetch error:', error);
    }
  }

  private async fetchWellnessResults(query: string, results: SearchResult[]): Promise<void> {
    try {
      const tip = await wellnessContentService.getDailyTip();
      
      results.push({
        id: `wellness_${Date.now()}`,
        type: 'wellness',
        title: tip.title,
        description: tip.content.substring(0, 150) + '...',
        content: tip,
        metadata: {
          category: tip.category,
          emoji: this.getCategoryEmoji(tip.category)
        },
        score: 0.85
      });
    } catch (error) {
      console.error('Wellness fetch error:', error);
    }
  }

  private getCategoryEmoji(category: string): string {
    const emojiMap: Record<string, string> = {
      physical: '💪',
      mental: '🧠',
      nutrition: '🥗',
      productivity: '⚡',
      sleep: '😴'
    };
    return emojiMap[category] || '💚';
  }

  private async fetchAssistantResults(query: string, results: SearchResult[]): Promise<void> {
    try {
      if (query.toLowerCase().includes('weather')) {
        const weather = await realWeatherService.getCurrentWeather();
        
        results.push({
          id: `weather_${Date.now()}`,
          type: 'assistant',
          title: `Weather in ${weather.location}`,
          description: `${weather.temperature}°C - ${weather.condition}`,
          content: weather,
          score: 0.95
        });
      }
    } catch (error) {
      console.error('Assistant fetch error:', error);
    }
  }

  private async fetchShoppingResults(query: string, results: SearchResult[]): Promise<void> {
    try {
      const response = await pollenAIUnified.generateContent({
        type: 'product_description',
        context: query,
        mood: 'enthusiastic'
      });

      if (response.success) {
        results.push({
          id: `shopping_${Date.now()}`,
          type: 'shopping',
          title: 'Product Recommendations',
          description: response.content.substring(0, 150) + '...',
          content: response.content,
          score: 0.75
        });
      }
    } catch (error) {
      console.error('Shopping fetch error:', error);
    }
  }

  private async fetchMediaResults(query: string, results: SearchResult[]): Promise<void> {
    try {
      const response = await pollenAIUnified.generateContent({
        type: 'music_description',
        context: query,
        mood: 'creative'
      });

      if (response.success) {
        results.push({
          id: `media_${Date.now()}`,
          type: 'music',
          title: 'Music & Media Content',
          description: response.content.substring(0, 150) + '...',
          content: response.content,
          score: 0.7
        });
      }
    } catch (error) {
      console.error('Media fetch error:', error);
    }
  }

  private async fetchEntertainmentResults(query: string, results: SearchResult[]): Promise<void> {
    try {
      const response = await pollenAIUnified.generateContent({
        type: 'entertainment_description',
        context: query,
        mood: 'exciting'
      });

      if (response.success) {
        results.push({
          id: `entertainment_${Date.now()}`,
          type: 'entertainment',
          title: 'Entertainment Recommendations',
          description: response.content.substring(0, 150) + '...',
          content: response.content,
          score: 0.72
        });
      }
    } catch (error) {
      console.error('Entertainment fetch error:', error);
    }
  }

  private rankResults(results: SearchResult[]): SearchResult[] {
    return results.sort((a, b) => b.score - a.score);
  }

  clearCache(): void {
    this.cache.clear();
  }
}

export const searchOrchestrator = new SearchOrchestrator();
