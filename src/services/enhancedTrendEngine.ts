import { realDataIntegration } from './realDataIntegration';
import { clientAI } from './clientAI';
import { personalizationEngine } from './personalizationEngine';
import { pollenAdaptiveService } from './pollenAdaptiveService';
import { storageService } from './storageService';
import { trendAggregator } from './trendAggregator';

export interface TrendData {
  id: string;
  topic: string;
  score: number;
  sentiment: number;
  source: string;
  timestamp: number;
  category: string;
  keywords: string[];
  momentum: number;
  reach: number;
  engagement: number;
}

export interface GeneratedPost {
  id: string;
  content: string;
  topic: string;
  platform: string;
  engagement_score: number;
  hashtags: string[];
  timestamp: number;
  type: 'social' | 'news' | 'ad' | 'product';
}

export interface ProductRecommendation {
  id: string;
  name: string;
  description: string;
  relevanceScore: number;
  trendTopic: string;
  price: number;
  source: string;
  category: string;
}

export interface TrendAlert {
  id: string;
  topic: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: number;
  dismissed: boolean;
  actionable: boolean;
}

class EnhancedTrendEngine {
  private sseConnection: EventSource | null = null;
  private trends: TrendData[] = [];
  private generatedPosts: GeneratedPost[] = [];
  private recommendations: ProductRecommendation[] = [];
  private alerts: TrendAlert[] = [];
  private listeners: Set<(data: any) => void> = new Set();
  private isRunning = false;

  constructor() {
    this.loadPersistedData();
    this.initializeSSE();
  }

  private async loadPersistedData(): Promise<void> {
    try {
      const [trends, posts, recommendations, alerts] = await Promise.all([
        storageService.getData<TrendData[]>('enhanced_trends'),
        storageService.getData<GeneratedPost[]>('generated_posts'),
        storageService.getData<ProductRecommendation[]>('trend_recommendations'),
        storageService.getData<TrendAlert[]>('trend_alerts')
      ]);

      this.trends = trends || [];
      this.generatedPosts = posts || [];
      this.recommendations = recommendations || [];
      this.alerts = alerts || [];
    } catch (error) {
      console.error('Error loading persisted trend data:', error);
    }
  }

  private async persistData(): Promise<void> {
    try {
      await Promise.all([
        storageService.setData('enhanced_trends', this.trends),
        storageService.setData('generated_posts', this.generatedPosts),
        storageService.setData('trend_recommendations', this.recommendations),
        storageService.setData('trend_alerts', this.alerts)
      ]);
    } catch (error) {
      console.error('Error persisting trend data:', error);
    }
  }

  private initializeSSE(): void {
    // Simulate SSE endpoint for trend updates
    if (this.sseConnection) {
      this.sseConnection.close();
    }

    // In a real implementation, this would connect to an actual SSE endpoint
    // For now, we'll simulate with periodic updates
    this.startPeriodicUpdates();
  }

  private startPeriodicUpdates(): void {
    setInterval(async () => {
      if (!this.isRunning) {
        await this.updateTrends();
      }
    }, 60000); // Update every minute

    // Initial update
    setTimeout(() => this.updateTrends(), 1000);
  }

  private async updateTrends(): Promise<void> {
    if (this.isRunning) return;
    
    this.isRunning = true;
    try {
      // Add timeout to prevent hanging
      const updatePromise = this.performTrendUpdate();
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Trend update timeout')), 30000)
      );
      
      await Promise.race([updatePromise, timeoutPromise]);
    } catch (error) {
      console.error('Error updating trends:', error);
      // Don't let errors bubble up and crash the app
    } finally {
      this.isRunning = false;
    }
  }

  private async performTrendUpdate(): Promise<void> {
    const newTrends = await this.fetchLatestTrends();
    if (newTrends.length === 0) {
      console.warn('No new trends fetched, using cached data');
      return;
    }
    
    this.processTrends(newTrends);
    
    // Run these operations in parallel for better performance
    await Promise.allSettled([
      this.generateContentFromTrends(),
      this.createRecommendations(),
      this.checkForAlerts(),
      this.persistData()
    ]);
    
    this.notifyListeners({ type: 'trends_updated', data: this.trends });
  }

  private async fetchLatestTrends(): Promise<TrendData[]> {
    try {
      // Use real Pollen AI instead of unreliable external APIs
      const { pollenTrendEngine } = await import('./pollenTrendEngine');
      const pollenTrends = await pollenTrendEngine.generateTrends();
      
      console.log(`✅ Generated ${pollenTrends.length} trends using real Pollen AI`);
      return pollenTrends; // processTrends will handle deduplication
    } catch (error) {
      console.error('Error fetching Pollen AI trends:', error);
      return [];
    }
  }

  // External API methods removed - now using 100% reliable Pollen AI

  private async analyzeSentiment(text: string): Promise<any> {
    try {
      return await clientAI.analyzeSentiment(text);
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
      return { compound: 0, pos: 0, neu: 1, neg: 0 };
    }
  }

  private extractKeywords(text: string): string[] {
    const keywords = [
      'AI', 'ML', 'blockchain', 'crypto', 'web3', 'react', 'javascript', 'python',
      'startup', 'innovation', 'technology', 'software', 'developer', 'coding',
      'data', 'cloud', 'security', 'mobile', 'api', 'framework', 'open source'
    ];
    
    return keywords.filter(keyword => 
      text.toLowerCase().includes(keyword.toLowerCase())
    );
  }

  private calculateTrendScore(sentiment: any, keywords: string[], source: string): number {
    let score = Math.abs(sentiment.compound || 0) * 50;
    
    // Boost for relevant keywords
    score += keywords.length * 10;
    
    // Source-based multiplier
    const sourceMultiplier = {
      'hackernews': 1.2,
      'reddit': 1.0,
      'github': 1.1
    }[source] || 1.0;
    
    score *= sourceMultiplier;
    
    // Add randomness for engagement
    score += Math.random() * 20;
    
    return Math.min(score, 100);
  }

  private categorizeContent(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('ai') || lowerText.includes('machine learning') || lowerText.includes('ml')) {
      return 'AI/ML';
    }
    if (lowerText.includes('crypto') || lowerText.includes('blockchain') || lowerText.includes('web3')) {
      return 'Crypto/Web3';
    }
    if (lowerText.includes('startup') || lowerText.includes('funding') || lowerText.includes('ipo')) {
      return 'Business';
    }
    if (lowerText.includes('react') || lowerText.includes('javascript') || lowerText.includes('framework')) {
      return 'Web Development';
    }
    if (lowerText.includes('security') || lowerText.includes('privacy') || lowerText.includes('breach')) {
      return 'Security';
    }
    
    return 'Technology';
  }

  private processTrends(newTrends: TrendData[]): void {
    // Merge with existing trends and remove duplicates
    const trendMap = new Map();
    
    // Add existing trends
    this.trends.forEach(trend => {
      trendMap.set(trend.topic.toLowerCase(), trend);
    });
    
    // Add new trends (will overwrite if duplicate)
    newTrends.forEach(trend => {
      trendMap.set(trend.topic.toLowerCase(), trend);
    });
    
    this.trends = Array.from(trendMap.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 100); // Keep top 100 trends
  }

  private async generateContentFromTrends(): Promise<void> {
    const topTrends = this.trends.slice(0, 10);
    const newPosts: GeneratedPost[] = [];

    for (const trend of topTrends) {
      try {
        // Generate fallback content if AI fails
        let content = await this.generateFallbackContent(trend);
        
        // Try AI generation first
        try {
          const result = await pollenAdaptiveService.curateSocialPost(trend.topic);
          if (result.content && result.content.length > 20) {
            content = result.content;
          }
        } catch (aiError) {
          console.log('AI generation failed, using fallback content');
        }

        const post: GeneratedPost = {
          id: `generated-social-${Date.now()}-${Math.random()}`,
          content,
          topic: trend.topic,
          platform: 'multi-platform',
          engagement_score: trend.score,
          hashtags: trend.keywords.map(k => `#${k.replace(/[^a-zA-Z0-9]/g, '')}`),
          timestamp: Date.now(),
          type: 'social'
        };
        
        newPosts.push(post);
      } catch (error) {
        console.error('Error generating content for trend:', trend.topic, error);
        
        // Generate basic fallback post
        const fallbackPost: GeneratedPost = {
          id: `fallback-${Date.now()}-${Math.random()}`,
          content: this.createBasicFallbackContent(trend),
          topic: trend.topic,
          platform: 'multi-platform',
          engagement_score: trend.score,
          hashtags: trend.keywords.map(k => `#${k.replace(/[^a-zA-Z0-9]/g, '')}`),
          timestamp: Date.now(),
          type: 'social'
        };
        
        newPosts.push(fallbackPost);
      }
    }

    // Add new posts and keep last 200
    this.generatedPosts = [...newPosts, ...this.generatedPosts].slice(0, 200);
    this.notifyListeners({ type: 'posts_generated', data: newPosts });
  }

  private async createRecommendations(): Promise<void> {
    const topTrends = this.trends.slice(0, 15);
    const newRecommendations: ProductRecommendation[] = [];

    for (const trend of topTrends) {
      if (trend.score > 60) {
        // Create product recommendations based on trending topics
        const products = this.generateProductRecommendations(trend);
        newRecommendations.push(...products);
      }
    }

    this.recommendations = [...newRecommendations, ...this.recommendations].slice(0, 50);
    this.notifyListeners({ type: 'recommendations_updated', data: newRecommendations });
  }

  private generateProductRecommendations(trend: TrendData): ProductRecommendation[] {
    const recommendations: ProductRecommendation[] = [];
    const baseProducts = {
      'AI/ML': [
        { name: 'AI Development Kit', price: 299, description: 'Complete AI development toolkit' },
        { name: 'ML Model Training Service', price: 99, description: 'Cloud-based ML training platform' }
      ],
      'Crypto/Web3': [
        { name: 'Crypto Trading Bot', price: 199, description: 'Automated cryptocurrency trading' },
        { name: 'Web3 Development Framework', price: 149, description: 'Tools for Web3 app development' }
      ],
      'Web Development': [
        { name: 'React Component Library', price: 79, description: 'Premium React components' },
        { name: 'Full-Stack Boilerplate', price: 129, description: 'Ready-to-deploy web app template' }
      ]
    };

    const categoryProducts = baseProducts[trend.category as keyof typeof baseProducts] || [
      { name: `${trend.category} Solution`, price: 99, description: `Professional ${trend.category} tools` }
    ];

    categoryProducts.forEach((product, index) => {
      recommendations.push({
        id: `rec-${trend.id}-${index}`,
        name: product.name,
        description: product.description,
        relevanceScore: trend.score,
        trendTopic: trend.topic,
        price: product.price,
        source: 'TrendEngine',
        category: trend.category
      });
    });

    return recommendations;
  }

  private async checkForAlerts(): Promise<void> {
    const criticalTrends = this.trends.filter(trend => trend.score > 85);
    const newAlerts: TrendAlert[] = [];

    for (const trend of criticalTrends) {
      const existingAlert = this.alerts.find(alert => 
        alert.topic === trend.topic && !alert.dismissed
      );

      if (!existingAlert) {
        const alert: TrendAlert = {
          id: `alert-${trend.id}`,
          topic: trend.topic,
          severity: trend.score > 95 ? 'critical' : trend.score > 90 ? 'high' : 'medium',
          message: `Trending: ${trend.topic} (Score: ${trend.score.toFixed(1)})`,
          timestamp: Date.now(),
          dismissed: false,
          actionable: true
        };

        newAlerts.push(alert);
      }
    }

    this.alerts = [...newAlerts, ...this.alerts].slice(0, 50);
    if (newAlerts.length > 0) {
      this.notifyListeners({ type: 'alerts_created', data: newAlerts });
    }
  }

  // Public API methods
  public subscribe(callback: (data: any) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  private notifyListeners(data: any): void {
    this.listeners.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in trend listener:', error);
      }
    });
  }

  public getTrends(): TrendData[] {
    // Apply blacklist on read
    try {
      const { isBlacklistedText } = require('../lib/blacklist');
      return this.trends.filter(t => !isBlacklistedText(t.topic));
    } catch {
      return this.trends;
    }
  }

  public getGeneratedPosts(): GeneratedPost[] {
    try {
      const { isBlacklistedText } = require('../lib/blacklist');
      return this.generatedPosts.filter(p => !isBlacklistedText(p.topic) && !isBlacklistedText(p.content));
    } catch {
      return this.generatedPosts;
    }
  }

  public getRecommendations(): ProductRecommendation[] {
    return this.recommendations;
  }

  public getAlerts(): TrendAlert[] {
    return this.alerts.filter(alert => !alert.dismissed);
  }

  public async dismissAlert(alertId: string): Promise<void> {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.dismissed = true;
      await this.persistData();
      this.notifyListeners({ type: 'alert_dismissed', data: alertId });
    }
  }

  public async forceUpdate(): Promise<void> {
    await this.updateTrends();
  }

  public getTopTrendsByCategory(category: string, limit: number = 10): TrendData[] {
    return this.trends
      .filter(trend => trend.category === category)
      .slice(0, limit);
  }

  public searchTrends(query: string): TrendData[] {
    const lowerQuery = query.toLowerCase();
    return this.trends.filter(trend => 
      trend.topic.toLowerCase().includes(lowerQuery) ||
      trend.keywords.some(k => k.toLowerCase().includes(lowerQuery))
    );
  }

  private humanizeRepoName(repoName: string): string {
    return repoName
      .replace(/[-_]/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }

  private createTopicFromRepoName(repoName: string): string {
    const humanized = this.humanizeRepoName(repoName);
    const techTerms = ['api', 'sdk', 'framework', 'library', 'tool', 'cli', 'app', 'service'];
    const hastech = techTerms.some(term => humanized.toLowerCase().includes(term));
    
    if (hastech) {
      return `New ${humanized} gaining popularity in tech community`;
    }
    return `${humanized} - trending development project`;
  }

  private async generateFallbackContent(trend: TrendData): Promise<string> {
    const templates = [
      `🔥 ${trend.topic} is gaining serious momentum! What do you think about this trend? #trending`,
      `💡 Interesting developments in ${trend.category}: ${trend.topic}. The implications could be huge!`,
      `🚀 ${trend.topic} is exploding right now. Here's why this matters for the future of ${trend.category.toLowerCase()}.`,
      `⚡ Breaking: ${trend.topic} is trending hard. This could change everything we know about ${trend.category.toLowerCase()}.`,
      `🎯 ${trend.topic} caught my attention today. The potential applications are fascinating!`,
      `📈 ${trend.topic} is seeing massive growth. Time to pay attention to this trend!`
    ];

    return templates[Math.floor(Math.random() * templates.length)];
  }

  private createBasicFallbackContent(trend: TrendData): string {
    return `🔥 ${trend.topic} is trending! This ${trend.category.toLowerCase()} topic has a significance score of ${trend.score.toFixed(1)}. What are your thoughts? #trending #${trend.category.toLowerCase().replace(/[^a-zA-Z0-9]/g, '')}`;
  }

  public async generateContentForTopic(topic: string, type: 'social' | 'ad' | 'news' = 'social'): Promise<GeneratedPost | null> {
    try {
      const result = type === 'ad' 
        ? await pollenAdaptiveService.createAdvertisement(topic)
        : await pollenAdaptiveService.curateSocialPost(topic);

      if (result.content) {
        const post: GeneratedPost = {
          id: `manual-${type}-${Date.now()}`,
          content: result.content,
          topic,
          platform: 'multi-platform',
          engagement_score: 75,
          hashtags: [topic.split(' ').slice(0, 3).map(w => `#${w}`)].flat(),
          timestamp: Date.now(),
          type: type as any
        };

        this.generatedPosts.unshift(post);
        await this.persistData();
        this.notifyListeners({ type: 'manual_content_generated', data: post });
        
        return post;
      }
    } catch (error) {
      console.error('Error generating manual content:', error);
    }
    
    return null;
  }
}

export const enhancedTrendEngine = new EnhancedTrendEngine();
