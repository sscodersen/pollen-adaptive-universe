import { realDataIntegration } from './realDataIntegration';
import { clientAI } from './clientAI';
import { personalizationEngine } from './personalizationEngine';
import { pollenAdaptiveService } from './pollenAdaptiveService';
import { storageService } from './storageService';

export interface TrendScore {
  topic: string;
  score: number;
  sentiment: number;
  source: string;
  timestamp: number;
}

export interface TrendAlert {
  id: string;
  topic: string;
  score: number;
  content: string;
  timestamp: number;
  dismissed: boolean;
}

export interface GeneratedContent {
  id: string;
  topic: string;
  type: 'ad' | 'post' | 'article';
  content: string;
  targetAudience: string;
  platform: string;
  timestamp: number;
}

export interface ProductRecommendation {
  id: string;
  name: string;
  description: string;
  relevanceScore: number;
  trendTopic: string;
  source: string;
}

class TrendBasedContentGenerator {
  private trendScores: TrendScore[] = [];
  private alerts: TrendAlert[] = [];
  private generatedContent: GeneratedContent[] = [];
  private isRunning = false;

  constructor() {
    this.loadPersistedData();
  }

  private async loadPersistedData(): Promise<void> {
    try {
      const savedTrends = await storageService.getData<TrendScore[]>('trend_scores');
      const savedAlerts = await storageService.getData<TrendAlert[]>('trend_alerts');
      const savedContent = await storageService.getData<GeneratedContent[]>('generated_content');

      this.trendScores = savedTrends || [];
      this.alerts = savedAlerts || [];
      this.generatedContent = savedContent || [];
    } catch (error) {
      console.error('Error loading persisted trend data:', error);
    }
  }

  private async persistData(): Promise<void> {
    try {
      await Promise.all([
        storageService.setData('trend_scores', this.trendScores),
        storageService.setData('trend_alerts', this.alerts),
        storageService.setData('generated_content', this.generatedContent)
      ]);
    } catch (error) {
      console.error('Error persisting trend data:', error);
    }
  }

  async scrapeTrends(): Promise<string[]> {
    try {
      const [
        hackerNews,
        redditTrends,
        githubTrending,
        devPosts
      ] = await Promise.all([
        realDataIntegration.fetchHackerNews(20),
        realDataIntegration.fetchRedditContent('technology', 15),
        realDataIntegration.fetchGitHubTrending('daily'),
        realDataIntegration.fetchDevToArticles(undefined, 15)
      ]);

      const trends: string[] = [];
      
      // Extract trend topics from various sources
      hackerNews.forEach(story => {
        trends.push(story.title);
        if (story.description) trends.push(story.description);
      });

      redditTrends.forEach(post => {
        trends.push(post.title);
        if (post.description) trends.push(post.description);
      });

      githubTrending.forEach(repo => {
        trends.push(repo.title);
        if (repo.description) trends.push(repo.description);
      });

      devPosts.forEach(article => {
        trends.push(article.title);
        if (article.description) trends.push(article.description);
        if (article.metadata?.tags) {
          article.metadata.tags.forEach((tag: string) => trends.push(tag));
        }
      });

      return trends.filter(trend => trend && trend.length > 10); // Filter meaningful content
    } catch (error) {
      console.error('Error scraping trends:', error);
      return [];
    }
  }

  async analyzeSentiment(text: string): Promise<number> {
    try {
      const result = await clientAI.analyzeSentiment(text);
      // Convert sentiment label to numeric score
      return result.label === 'POSITIVE' ? result.score : -result.score;
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
      return 0;
    }
  }

  async identifyTrends(trends: string[]): Promise<TrendScore[]> {
    const trendScores: TrendScore[] = [];
    const timestamp = Date.now();

    for (const trend of trends) {
      try {
        const sentiment = await this.analyzeSentiment(trend);
        
        // Calculate trend score based on sentiment, length, and keywords
        const keywords = ['AI', 'technology', 'innovation', 'startup', 'crypto', 'web3', 'react', 'javascript'];
        const keywordBonus = keywords.some(keyword => 
          trend.toLowerCase().includes(keyword.toLowerCase())
        ) ? 0.2 : 0;

        const score = Math.abs(sentiment) + keywordBonus;

        if (score > 0.1) {
          trendScores.push({
            topic: trend,
            score,
            sentiment,
            source: 'multi-source',
            timestamp
          });
        }
      } catch (error) {
        console.error('Error processing trend:', trend, error);
      }
    }

    return trendScores.sort((a, b) => b.score - a.score);
  }

  async selectTopics(trendScores: TrendScore[]): Promise<TrendScore[]> {
    try {
      const profile = personalizationEngine.getPersonalizationInsights();
      const userInterests = Object.keys(profile.topInterests);
      
      return trendScores.filter(trend => {
        // Check if trend matches user preferences
        const matchesInterests = userInterests.some(interest =>
          trend.topic.toLowerCase().includes(interest.toLowerCase())
        );
        
        return trend.score > 0.3 || matchesInterests;
      });
    } catch (error) {
      console.error('Error selecting topics:', error);
      return trendScores.filter(trend => trend.score > 0.3);
    }
  }

  async recommendProducts(trendScores: TrendScore[]): Promise<ProductRecommendation[]> {
    const recommendations: ProductRecommendation[] = [];

    for (const trend of trendScores.slice(0, 5)) {
      try {
        // Generate mock product recommendations based on trending topics
        // In a real implementation, this would call an actual product search API
        const mockProducts = [
          {
            title: `Product related to ${trend.topic}`,
            description: `High-quality product for ${trend.topic}`,
            source: 'mock-product-db'
          },
          {
            title: `Premium ${trend.topic} solution`,
            description: `Professional grade ${trend.topic} tools`,
            source: 'mock-product-db'
          }
        ];
        
        mockProducts.forEach((product, index) => {
          recommendations.push({
            id: `prod-${Date.now()}-${index}`,
            name: product.title,
            description: product.description,
            relevanceScore: trend.score,
            trendTopic: trend.topic,
            source: product.source
          });
        });
      } catch (error) {
        console.error('Error generating product recommendations:', error);
      }
    }

    return recommendations;
  }

  async generateContent(topic: string, type: 'ad' | 'post' | 'article' = 'post'): Promise<GeneratedContent | null> {
    try {
      let result;
      
      switch (type) {
        case 'ad':
          result = await pollenAdaptiveService.createAdvertisement(topic);
          break;
        case 'post':
          result = await pollenAdaptiveService.curateSocialPost(topic);
          break;
        case 'article':
          result = await pollenAdaptiveService.solveTask(`Write an article about ${topic}`);
          break;
      }

      const content: GeneratedContent = {
        id: `content-${Date.now()}`,
        topic,
        type,
        content: result.content || result.solution || 'Content generated',
        targetAudience: result.targetAudience || 'General audience',
        platform: result.platform || 'Multi-platform',
        timestamp: Date.now()
      };

      this.generatedContent.push(content);
      await this.persistData();

      return content;
    } catch (error) {
      console.error('Error generating content:', error);
      return null;
    }
  }

  async sendAlert(topic: string, score: number): Promise<TrendAlert> {
    const alert: TrendAlert = {
      id: `alert-${Date.now()}`,
      topic,
      score,
      content: `New trending topic detected: ${topic} (Score: ${score.toFixed(2)})`,
      timestamp: Date.now(),
      dismissed: false
    };

    this.alerts.push(alert);
    await this.persistData();

    // Track as user activity for personalization
    await personalizationEngine.trackBehavior({
      action: 'view',
      contentType: 'entertainment',
      contentId: alert.id,
      metadata: { topic, score, type: 'trend_alert' }
    });

    return alert;
  }

  async run(): Promise<{
    trends: TrendScore[];
    content: GeneratedContent[];
    recommendations: ProductRecommendation[];
    alerts: TrendAlert[];
  }> {
    if (this.isRunning) {
      console.warn('Trend analysis already running');
      return {
        trends: this.trendScores,
        content: this.generatedContent,
        recommendations: [],
        alerts: this.alerts
      };
    }

    this.isRunning = true;

    try {
      console.log('Starting trend analysis...');
      
      // Step 1: Scrape trends from various sources
      const trends = await this.scrapeTrends();
      console.log(`Scraped ${trends.length} trends`);

      // Step 2: Analyze and score trends
      const trendScores = await this.identifyTrends(trends);
      console.log(`Identified ${trendScores.length} trending topics`);

      // Step 3: Select relevant topics based on user preferences
      const selectedTopics = await this.selectTopics(trendScores);
      console.log(`Selected ${selectedTopics.length} relevant topics`);

      // Step 4: Generate product recommendations
      const recommendations = await this.recommendProducts(selectedTopics);
      console.log(`Generated ${recommendations.length} product recommendations`);

      // Step 5: Generate content for top trends
      const newContent: GeneratedContent[] = [];
      for (const trend of selectedTopics.slice(0, 3)) {
        const content = await this.generateContent(trend.topic, 'post');
        if (content) {
          newContent.push(content);
        }
      }

      // Step 6: Send alerts for high-scoring trends
      const newAlerts: TrendAlert[] = [];
      for (const trend of selectedTopics.filter(t => t.score > 0.5)) {
        const alert = await this.sendAlert(trend.topic, trend.score);
        newAlerts.push(alert);
      }

      // Update stored data
      this.trendScores = [...this.trendScores, ...trendScores].slice(-100); // Keep last 100
      await this.persistData();

      console.log('Trend analysis completed');

      return {
        trends: selectedTopics,
        content: newContent,
        recommendations,
        alerts: newAlerts
      };
    } catch (error) {
      console.error('Error in trend analysis:', error);
      return {
        trends: [],
        content: [],
        recommendations: [],
        alerts: []
      };
    } finally {
      this.isRunning = false;
    }
  }

  // Utility methods for UI
  getTrendScores(): TrendScore[] {
    return this.trendScores;
  }

  getAlerts(): TrendAlert[] {
    return this.alerts.filter(alert => !alert.dismissed);
  }

  getGeneratedContent(): GeneratedContent[] {
    return this.generatedContent;
  }

  async dismissAlert(alertId: string): Promise<void> {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.dismissed = true;
      await this.persistData();
    }
  }

  async clearOldData(maxAge: number = 7 * 24 * 60 * 60 * 1000): Promise<void> {
    const cutoff = Date.now() - maxAge;
    
    this.trendScores = this.trendScores.filter(trend => trend.timestamp > cutoff);
    this.alerts = this.alerts.filter(alert => alert.timestamp > cutoff);
    this.generatedContent = this.generatedContent.filter(content => content.timestamp > cutoff);
    
    await this.persistData();
  }
}

export const trendBasedContentGenerator = new TrendBasedContentGenerator();