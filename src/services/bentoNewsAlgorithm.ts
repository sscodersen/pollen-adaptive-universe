/**
 * Bento News-Type Algorithm for Continuous Post Generation
 * 
 * Intelligently generates diverse, high-quality posts across all categories
 * by combining:
 * - Real-time trend data from Exploding Topics scraper
 * - Pollen AI for content generation
 * - Content quality and diversity algorithms
 * - Personalization based on user preferences
 */

import { pollenAI } from './pollenAIUnified';
import { trendScraperSSE, type ScrapedTrend } from './trendScraperSSE';
import type { ContentType, UnifiedContent } from './unifiedContentEngine';

export interface BentoPostConfig {
  categories: ContentType[];
  postsPerCategory: number;
  refreshInterval: number; // milliseconds
  qualityThreshold: number;
  diversityWeight: number;
  trendWeight: number;
}

export interface BentoPost extends UnifiedContent {
  trendData?: ScrapedTrend;
  bentoScore: number;
  generationMethod: 'trend-based' | 'ai-generated' | 'hybrid';
}

class BentoNewsAlgorithm {
  private config: BentoPostConfig = {
    categories: ['social', 'news', 'entertainment', 'shop', 'music', 'wellness', 'games'],
    postsPerCategory: 5,
    refreshInterval: 15 * 60 * 1000, // 15 minutes
    qualityThreshold: 7.5,
    diversityWeight: 0.6,
    trendWeight: 0.8
  };

  private generatedPosts: Map<string, BentoPost[]> = new Map();
  private trendCache: ScrapedTrend[] = [];
  private isRunning = false;
  private intervalId: NodeJS.Timeout | null = null;

  /**
   * Start continuous post generation
   */
  async startContinuousGeneration(customConfig?: Partial<BentoPostConfig>): Promise<void> {
    if (this.isRunning) {
      console.log('⚠️ Bento algorithm already running');
      return;
    }

    // Merge custom config
    if (customConfig) {
      this.config = { ...this.config, ...customConfig };
    }

    this.isRunning = true;
    console.log('🚀 Starting Bento News Algorithm');
    console.log(`📊 Generating ${this.config.postsPerCategory} posts per category`);
    console.log(`🔄 Refresh interval: ${this.config.refreshInterval / 1000}s`);

    // Connect to trend scraper SSE
    this.connectToTrendScraper();

    // Initial generation
    await this.generateAllCategoryPosts();

    // Setup continuous generation
    this.intervalId = setInterval(async () => {
      await this.generateAllCategoryPosts();
    }, this.config.refreshInterval);
  }

  /**
   * Stop continuous generation
   */
  stopContinuousGeneration(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    trendScraperSSE.disconnect();
    this.isRunning = false;
    console.log('🛑 Stopped Bento News Algorithm');
  }

  /**
   * Connect to trend scraper and cache trends
   */
  private connectToTrendScraper(): void {
    // Subscribe to trend updates
    trendScraperSSE.onTrend((trend) => {
      console.log('📈 New trend received:', trend.topic);
      this.trendCache.push(trend);

      // Keep only recent trends (max 50)
      if (this.trendCache.length > 50) {
        this.trendCache = this.trendCache.slice(-50);
      }

      // Trigger post generation for relevant categories
      this.generatePostsFromTrend(trend);
    });

    // Connect to stream
    trendScraperSSE.connect();

    // Also fetch latest trends immediately
    trendScraperSSE.getLatestTrends().then(trends => {
      this.trendCache = trends;
      console.log(`✅ Loaded ${trends.length} initial trends`);
    });
  }

  /**
   * Generate posts for all categories
   */
  private async generateAllCategoryPosts(): Promise<void> {
    console.log('🎯 Generating posts for all categories...');

    const generationPromises = this.config.categories.map(category =>
      this.generateCategoryPosts(category)
    );

    await Promise.all(generationPromises);

    console.log('✅ All category posts generated');
  }

  /**
   * Generate posts for a specific category
   */
  private async generateCategoryPosts(category: ContentType): Promise<void> {
    const posts: BentoPost[] = [];

    // Get relevant trends for this category
    const relevantTrends = this.getRelevantTrends(category);

    // Generate trend-based posts (60% of posts)
    const trendPostCount = Math.floor(this.config.postsPerCategory * 0.6);
    for (let i = 0; i < Math.min(trendPostCount, relevantTrends.length); i++) {
      const trend = relevantTrends[i];
      const post = await this.generateTrendBasedPost(category, trend);
      if (post && post.bentoScore >= this.config.qualityThreshold) {
        posts.push(post);
      }
    }

    // Generate AI posts to fill remaining slots (40% of posts)
    const remainingCount = this.config.postsPerCategory - posts.length;
    if (remainingCount > 0) {
      const aiPosts = await this.generateAIPosts(category, remainingCount);
      posts.push(...aiPosts);
    }

    // Apply diversity algorithm
    const diversePosts = this.applyDiversityAlgorithm(posts);

    // Store posts
    this.generatedPosts.set(category, diversePosts);
    
    console.log(`✅ Generated ${diversePosts.length} posts for ${category}`);
  }

  /**
   * Generate a post based on a trending topic
   */
  private async generateTrendBasedPost(
    category: ContentType,
    trend: ScrapedTrend
  ): Promise<BentoPost | null> {
    try {
      const prompt = this.buildTrendPrompt(category, trend);
      
      const response = await pollenAI.generate({
        prompt,
        mode: this.mapCategoryToMode(category),
        type: category,
        context: {
          trend: trend.topic,
          growth_rate: trend.growth_rate,
          search_volume: trend.search_volume,
          keywords: trend.keywords
        },
        use_cache: true,
        compression_level: 'medium'
      });

      // Build Bento post
      const bentoScore = this.calculateBentoScore({
        aiConfidence: response.confidence,
        trendGrowth: trend.growth_rate,
        trendVolume: trend.search_volume
      });

      const post: BentoPost = {
        id: `bento-${category}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: category,
        title: this.extractTitle(response.content, trend.topic),
        description: response.content,
        timestamp: new Date().toISOString(),
        significance: Math.min(10, trend.score),
        trending: trend.growth_rate > 150,
        quality: response.confidence * 10,
        views: trend.search_volume,
        engagement: Math.floor(trend.search_volume * 0.05),
        impact: trend.growth_rate > 200 ? 'high' : trend.growth_rate > 100 ? 'medium' : 'low',
        tags: trend.keywords,
        category: trend.category,
        trendData: trend,
        bentoScore,
        generationMethod: 'trend-based'
      };

      return post;
    } catch (error) {
      console.error(`Failed to generate trend post for ${category}:`, error);
      return null;
    }
  }

  /**
   * Generate AI posts without trend data
   */
  private async generateAIPosts(category: ContentType, count: number): Promise<BentoPost[]> {
    const posts: BentoPost[] = [];

    for (let i = 0; i < count; i++) {
      try {
        const prompt = this.buildCategoryPrompt(category);
        
        const response = await pollenAI.generate({
          prompt,
          mode: this.mapCategoryToMode(category),
          type: category,
          use_cache: false,
          compression_level: 'medium'
        });

        const bentoScore = response.confidence * 8; // Base score on AI confidence

        const post: BentoPost = {
          id: `bento-ai-${category}-${Date.now()}-${i}`,
          type: category,
          title: this.extractTitle(response.content),
          description: response.content,
          timestamp: new Date().toISOString(),
          significance: Math.min(10, bentoScore),
          trending: false,
          quality: response.confidence * 10,
          views: Math.floor(Math.random() * 50000) + 10000,
          engagement: Math.floor(Math.random() * 2000) + 500,
          impact: bentoScore > 8 ? 'high' : bentoScore > 6 ? 'medium' : 'low',
          tags: this.generateTags(category),
          category: this.getCategoryName(category),
          bentoScore,
          generationMethod: 'ai-generated'
        };

        posts.push(post);

        // Small delay to prevent overwhelming the AI
        await new Promise(resolve => setTimeout(resolve, 200));
      } catch (error) {
        console.error(`Failed to generate AI post for ${category}:`, error);
      }
    }

    return posts;
  }

  /**
   * Generate posts immediately when a new trend arrives
   */
  private async generatePostsFromTrend(trend: ScrapedTrend): Promise<void> {
    // Map trend category to content types
    const categories = this.mapTrendToCategories(trend.category);

    for (const category of categories) {
      const post = await this.generateTrendBasedPost(category, trend);
      if (post && post.bentoScore >= this.config.qualityThreshold) {
        const existingPosts = this.generatedPosts.get(category) || [];
        
        // Add new post at the beginning (most recent)
        const updatedPosts = [post, ...existingPosts].slice(0, this.config.postsPerCategory);
        this.generatedPosts.set(category, updatedPosts);

        console.log(`✅ Generated hot post for ${category}: ${post.title}`);
      }
    }
  }

  /**
   * Apply diversity algorithm to ensure varied content
   */
  private applyDiversityAlgorithm(posts: BentoPost[]): BentoPost[] {
    const diversePosts: BentoPost[] = [];
    const usedTopics = new Set<string>();

    // Sort by Bento score
    const sortedPosts = posts.sort((a, b) => b.bentoScore - a.bentoScore);

    for (const post of sortedPosts) {
      // Check topic similarity
      const topicWords = post.title.toLowerCase().split(' ');
      const isSimilar = topicWords.some(word => 
        word.length > 4 && usedTopics.has(word)
      );

      if (!isSimilar || diversePosts.length < this.config.postsPerCategory / 2) {
        diversePosts.push(post);
        topicWords.forEach(word => {
          if (word.length > 4) usedTopics.add(word);
        });
      }

      if (diversePosts.length >= this.config.postsPerCategory) {
        break;
      }
    }

    return diversePosts;
  }

  /**
   * Get posts for a specific category
   */
  getPosts(category: ContentType): BentoPost[] {
    return this.generatedPosts.get(category) || [];
  }

  /**
   * Get all generated posts
   */
  getAllPosts(): Map<string, BentoPost[]> {
    return this.generatedPosts;
  }

  // Helper methods

  private getRelevantTrends(category: ContentType): ScrapedTrend[] {
    const categoryKeywords = this.getCategoryKeywords(category);
    
    return this.trendCache
      .filter(trend => {
        const trendText = `${trend.topic} ${trend.description || ''} ${trend.keywords.join(' ')}`.toLowerCase();
        return categoryKeywords.some(keyword => trendText.includes(keyword.toLowerCase()));
      })
      .sort((a, b) => b.growth_rate - a.growth_rate)
      .slice(0, 10);
  }

  private getCategoryKeywords(category: ContentType): string[] {
    const keywordMap: Record<string, string[]> = {
      social: ['social', 'community', 'trending', 'viral', 'discussion'],
      news: ['news', 'breaking', 'update', 'announcement', 'report'],
      entertainment: ['entertainment', 'movie', 'show', 'celebrity', 'streaming'],
      shop: ['product', 'shopping', 'deals', 'ecommerce', 'retail'],
      music: ['music', 'artist', 'album', 'song', 'concert'],
      wellness: ['health', 'wellness', 'fitness', 'mental', 'therapy'],
      games: ['gaming', 'game', 'esports', 'console', 'pc']
    };

    return keywordMap[category] || [];
  }

  private buildTrendPrompt(category: ContentType, trend: ScrapedTrend): string {
    const prompts: Record<string, string> = {
      social: `Create an engaging social media post about: ${trend.topic}. Current growth: ${trend.growth_rate}%. Make it viral-worthy and discussion-provoking.`,
      news: `Write a breaking news article about: ${trend.topic}. This topic is trending with ${trend.growth_rate}% growth. Include key facts and implications.`,
      entertainment: `Create entertaining content about: ${trend.topic}. Make it engaging and fun for a general audience.`,
      shop: `Create a product recommendation post about: ${trend.topic}. Highlight benefits, features, and why it's trending (${trend.growth_rate}% growth).`,
      music: `Generate music-related content about: ${trend.topic}. Focus on artists, genres, or trends in the music industry.`,
      wellness: `Create wellness tips or health insights about: ${trend.topic}. Make it practical and beneficial for users.`,
      games: `Generate gaming content about: ${trend.topic}. Focus on game features, reviews, or gaming trends.`
    };

    return prompts[category] || `Create quality content about: ${trend.topic}`;
  }

  private buildCategoryPrompt(category: ContentType): string {
    const prompts: Record<string, string> = {
      social: 'Generate an interesting social post about current trending topics in technology, culture, or innovation.',
      news: 'Create a news article about recent developments in technology, business, or science.',
      entertainment: 'Generate entertaining content about movies, shows, or pop culture.',
      shop: 'Create a product recommendation for innovative or trending products.',
      music: 'Generate music content about popular artists, albums, or music trends.',
      wellness: 'Create wellness tips or health advice that helps people improve their wellbeing.',
      games: 'Generate gaming content about popular games, esports, or gaming industry news.'
    };

    return prompts[category] || `Generate quality ${category} content`;
  }

  private mapCategoryToMode(category: ContentType): string {
    const modeMap: Record<string, string> = {
      social: 'social',
      news: 'news',
      entertainment: 'entertainment',
      shop: 'shop',
      music: 'chat',
      wellness: 'wellness',
      games: 'chat'
    };

    return modeMap[category] || 'chat';
  }

  private mapTrendToCategories(trendCategory: string): ContentType[] {
    const categoryMap: Record<string, ContentType[]> = {
      'technology': ['social', 'news'],
      'ai-ml': ['social', 'news'],
      'business': ['news', 'shop'],
      'health': ['wellness', 'news'],
      'finance': ['news'],
      'entertainment': ['entertainment'],
      'crypto': ['news', 'social'],
      'ecommerce': ['shop']
    };

    return categoryMap[trendCategory] || ['social'];
  }

  private calculateBentoScore(params: {
    aiConfidence: number;
    trendGrowth: number;
    trendVolume: number;
  }): number {
    const confidenceScore = params.aiConfidence * 10;
    const trendScore = Math.min(10, (params.trendGrowth / 30));
    const volumeScore = Math.min(10, (params.trendVolume / 20000));

    return Number((
      (confidenceScore * 0.4) +
      (trendScore * this.config.trendWeight * 0.4) +
      (volumeScore * 0.2)
    ).toFixed(1));
  }

  private extractTitle(content: string, fallback?: string): string {
    // Extract first sentence or line as title
    const firstLine = content.split('\n')[0];
    const firstSentence = firstLine.split(/[.!?]/)[0];
    
    let title = firstSentence.trim();
    
    // Remove markdown symbols
    title = title.replace(/^[#*\-\d+.)\s]+/, '');
    
    // Limit length
    if (title.length > 80) {
      title = title.substring(0, 77) + '...';
    }
    
    return title || fallback || 'Trending Topic';
  }

  private getCategoryName(category: ContentType): string {
    const nameMap: Record<string, string> = {
      social: 'Social',
      news: 'News',
      entertainment: 'Entertainment',
      shop: 'Shopping',
      music: 'Music',
      wellness: 'Wellness',
      games: 'Gaming'
    };

    return nameMap[category] || category;
  }

  private generateTags(category: ContentType): string[] {
    const tagMap: Record<string, string[]> = {
      social: ['Trending', 'Viral', 'Community'],
      news: ['Breaking', 'Latest', 'Update'],
      entertainment: ['Entertainment', 'Trending', 'Popular'],
      shop: ['Products', 'Deals', 'Shopping'],
      music: ['Music', 'Audio', 'Entertainment'],
      wellness: ['Health', 'Wellness', 'Lifestyle'],
      games: ['Gaming', 'Esports', 'Entertainment']
    };

    return tagMap[category] || ['Trending'];
  }
}

// Export singleton instance
export const bentoNewsAlgorithm = new BentoNewsAlgorithm();
