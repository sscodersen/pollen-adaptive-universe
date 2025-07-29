import { storageService } from './storageService';
import { anonymousAuth, type AnonymousUser } from './anonymousAuth';
import { clientAI } from './clientAI';

export interface UserBehavior {
  id: string;
  userId: string;
  action: 'view' | 'click' | 'like' | 'share' | 'search' | 'generate' | 'save';
  contentId: string;
  contentType: 'music' | 'app' | 'game' | 'entertainment' | 'shop' | 'educational';
  metadata: Record<string, any>;
  timestamp: string;
  duration?: number; // time spent on content in seconds
}

export interface PersonalizationProfile {
  userId: string;
  interests: Array<{ topic: string; score: number; lastUpdated: string }>;
  preferences: {
    contentTypes: Record<string, number>; // preference scores for content types
    sources: Record<string, number>; // preference scores for content sources
    timeOfDay: Record<string, number>; // activity patterns by hour
    sessionLength: number; // average session length
  };
  recommendations: Array<{
    contentId: string;
    score: number;
    reason: string;
    generatedAt: string;
  }>;
  learningModel: {
    version: number;
    lastTrained: string;
    accuracy?: number;
  };
}

class PersonalizationEngine {
  private static instance: PersonalizationEngine;
  private profile: PersonalizationProfile | null = null;
  private behaviors: UserBehavior[] = [];
  private isTraining = false;

  static getInstance(): PersonalizationEngine {
    if (!PersonalizationEngine.instance) {
      PersonalizationEngine.instance = new PersonalizationEngine();
    }
    return PersonalizationEngine.instance;
  }

  async initialize(): Promise<void> {
    const user = anonymousAuth.getCurrentUser();
    if (!user) return;

    // Load existing profile or create new one
    this.profile = await storageService.getData<PersonalizationProfile>(`profile_${user.id}`);
    
    if (!this.profile) {
      this.profile = this.createNewProfile(user.id);
      await this.saveProfile();
    }

    // Load behavior history
    this.behaviors = await storageService.getData<UserBehavior[]>(`behaviors_${user.id}`) || [];
    
    // Start background learning if we have enough data
    if (this.behaviors.length > 10) {
      this.scheduleModelUpdate();
    }
  }

  private createNewProfile(userId: string): PersonalizationProfile {
    return {
      userId,
      interests: [],
      preferences: {
        contentTypes: {},
        sources: {},
        timeOfDay: {},
        sessionLength: 0,
      },
      recommendations: [],
      learningModel: {
        version: 1,
        lastTrained: new Date().toISOString(),
      },
    };
  }

  // Track user behavior
  async trackBehavior(behavior: Omit<UserBehavior, 'id' | 'userId' | 'timestamp'>): Promise<void> {
    const user = anonymousAuth.getCurrentUser();
    if (!user || !this.profile) return;

    const fullBehavior: UserBehavior = {
      id: `behavior_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      userId: user.id,
      timestamp: new Date().toISOString(),
      ...behavior,
    };

    this.behaviors.push(fullBehavior);
    
    // Keep only last 1000 behaviors to manage storage
    if (this.behaviors.length > 1000) {
      this.behaviors = this.behaviors.slice(-1000);
    }

    await this.saveBehaviors();
    
    // Update interests in real-time
    await this.updateInterestsFromBehavior(fullBehavior);
    
    // Schedule model update if enough new data
    if (this.behaviors.length % 20 === 0) {
      this.scheduleModelUpdate();
    }
  }

  // Update interests based on behavior
  private async updateInterestsFromBehavior(behavior: UserBehavior): Promise<void> {
    if (!this.profile) return;

    // Extract topics from content metadata
    const topics = this.extractTopicsFromMetadata(behavior.metadata);
    
    for (const topic of topics) {
      const existingInterest = this.profile.interests.find(i => i.topic === topic);
      
      if (existingInterest) {
        // Update existing interest with decay factor
        const timeDiff = Date.now() - new Date(existingInterest.lastUpdated).getTime();
        const decayFactor = Math.exp(-timeDiff / (1000 * 60 * 60 * 24 * 7)); // weekly decay
        
        existingInterest.score = (existingInterest.score * decayFactor) + this.getActionWeight(behavior.action);
        existingInterest.lastUpdated = new Date().toISOString();
      } else {
        // Add new interest
        this.profile.interests.push({
          topic,
          score: this.getActionWeight(behavior.action),
          lastUpdated: new Date().toISOString(),
        });
      }
    }

    // Update content type preferences
    const contentTypeScore = this.profile.preferences.contentTypes[behavior.contentType] || 0;
    this.profile.preferences.contentTypes[behavior.contentType] = 
      contentTypeScore + this.getActionWeight(behavior.action);

    // Update time of day preferences
    const hour = new Date().getHours().toString();
    const timeScore = this.profile.preferences.timeOfDay[hour] || 0;
    this.profile.preferences.timeOfDay[hour] = timeScore + 1;

    await this.saveProfile();
  }

  private extractTopicsFromMetadata(metadata: Record<string, any>): string[] {
    const topics: string[] = [];
    
    // Extract from various metadata fields
    if (metadata.tags) {
      topics.push(...(Array.isArray(metadata.tags) ? metadata.tags : [metadata.tags]));
    }
    if (metadata.categories) {
      topics.push(...(Array.isArray(metadata.categories) ? metadata.categories : [metadata.categories]));
    }
    if (metadata.genre) {
      topics.push(metadata.genre);
    }
    if (metadata.keywords) {
      topics.push(...(Array.isArray(metadata.keywords) ? metadata.keywords : [metadata.keywords]));
    }

    return topics.filter(topic => typeof topic === 'string' && topic.length > 2);
  }

  private getActionWeight(action: UserBehavior['action']): number {
    const weights = {
      view: 1,
      click: 2,
      like: 3,
      share: 4,
      save: 5,
      search: 2,
      generate: 3,
    };
    return weights[action] || 1;
  }

  // Generate personalized recommendations
  async generateRecommendations(contentItems: any[], limit: number = 10): Promise<any[]> {
    if (!this.profile || this.profile.interests.length === 0) {
      // Return random selection for new users
      return this.shuffleArray(contentItems).slice(0, limit);
    }

    const scoredItems = await Promise.all(
      contentItems.map(async (item) => {
        const score = await this.calculateContentScore(item);
        return { ...item, personalizedScore: score };
      })
    );

    return scoredItems
      .sort((a, b) => b.personalizedScore - a.personalizedScore)
      .slice(0, limit);
  }

  private async calculateContentScore(item: any): Promise<number> {
    if (!this.profile) return Math.random();

    let score = 0;

    // Interest-based scoring
    const itemTopics = this.extractTopicsFromMetadata(item.metadata || {});
    for (const topic of itemTopics) {
      const interest = this.profile.interests.find(i => i.topic.toLowerCase() === topic.toLowerCase());
      if (interest) {
        score += interest.score * 0.4; // 40% weight for interests
      }
    }

    // Content type preference
    const contentTypeScore = this.profile.preferences.contentTypes[item.category] || 0;
    score += (contentTypeScore / 100) * 0.3; // 30% weight for content type

    // Source preference
    const sourceScore = this.profile.preferences.sources[item.source] || 0;
    score += (sourceScore / 100) * 0.2; // 20% weight for source

    // Freshness factor (prefer newer content)
    if (item.publishedAt) {
      const age = Date.now() - new Date(item.publishedAt).getTime();
      const daysSincePublished = age / (1000 * 60 * 60 * 24);
      const freshnessFactor = Math.exp(-daysSincePublished / 7); // Exponential decay over weeks
      score += freshnessFactor * 0.1; // 10% weight for freshness
    }

    // Add some randomization to avoid getting stuck in filter bubbles
    score += Math.random() * 0.1;

    return Math.max(0, score);
  }

  // AI-powered content analysis for better recommendations
  async analyzeContentWithAI(content: string): Promise<{ sentiment: any; categories: any; keywords: string[] }> {
    try {
      const analysis = await clientAI.analyzeContent(content);
      return {
        sentiment: analysis.sentiment,
        categories: analysis.categories,
        keywords: analysis.keywords,
      };
    } catch (error) {
      console.error('AI content analysis failed:', error);
      return {
        sentiment: { label: 'neutral', score: 0.5 },
        categories: { labels: [], scores: [] },
        keywords: [],
      };
    }
  }

  // Smart search with personalization
  async personalizedSearch(query: string, items: any[]): Promise<any[]> {
    if (!this.profile || query.trim().length === 0) {
      return items;
    }

    // First, perform basic text matching
    const queryWords = query.toLowerCase().split(' ');
    const textMatches = items.filter(item => {
      const searchText = `${item.title} ${item.description} ${item.category}`.toLowerCase();
      return queryWords.some(word => searchText.includes(word));
    });

    // Then apply personalization scoring
    const personalizedResults = await this.generateRecommendations(textMatches, textMatches.length);
    
    // Track search behavior
    await this.trackBehavior({
      action: 'search',
      contentId: `search_${query}`,
      contentType: 'educational',
      metadata: { query, resultsCount: personalizedResults.length },
    });

    return personalizedResults;
  }

  // Get user insights
  getPersonalizationInsights(): {
    topInterests: Array<{ topic: string; score: number }>;
    preferredContentTypes: Array<{ type: string; score: number }>;
    activityPatterns: Array<{ hour: string; activity: number }>;
    behaviorSummary: {
      totalActions: number;
      averageSessionTime: number;
      mostActiveHour: string;
    };
  } {
    if (!this.profile) {
      return {
        topInterests: [],
        preferredContentTypes: [],
        activityPatterns: [],
        behaviorSummary: {
          totalActions: 0,
          averageSessionTime: 0,
          mostActiveHour: '12',
        },
      };
    }

    const topInterests = this.profile.interests
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    const preferredContentTypes = Object.entries(this.profile.preferences.contentTypes)
      .map(([type, score]) => ({ type, score }))
      .sort((a, b) => b.score - a.score);

    const activityPatterns = Object.entries(this.profile.preferences.timeOfDay)
      .map(([hour, activity]) => ({ hour, activity }))
      .sort((a, b) => parseInt(a.hour) - parseInt(b.hour));

    const mostActiveHour = Object.entries(this.profile.preferences.timeOfDay)
      .reduce((max, [hour, activity]) => 
        activity > max.activity ? { hour, activity } : max, 
        { hour: '12', activity: 0 }
      ).hour;

    return {
      topInterests,
      preferredContentTypes,
      activityPatterns,
      behaviorSummary: {
        totalActions: this.behaviors.length,
        averageSessionTime: this.profile.preferences.sessionLength,
        mostActiveHour,
      },
    };
  }

  // Schedule model updates
  private scheduleModelUpdate(): void {
    if (this.isTraining) return;

    setTimeout(() => {
      this.updateLearningModel();
    }, 5000); // Delay to avoid blocking UI
  }

  private async updateLearningModel(): Promise<void> {
    if (!this.profile || this.isTraining) return;

    this.isTraining = true;
    console.log('Updating personalization model...');

    try {
      // Simple collaborative filtering simulation
      await this.updateInterestWeights();
      
      this.profile.learningModel.version += 1;
      this.profile.learningModel.lastTrained = new Date().toISOString();
      
      await this.saveProfile();
      console.log('Personalization model updated successfully');
    } catch (error) {
      console.error('Failed to update personalization model:', error);
    } finally {
      this.isTraining = false;
    }
  }

  private async updateInterestWeights(): Promise<void> {
    if (!this.profile) return;

    // Apply time decay to all interests
    const now = Date.now();
    this.profile.interests = this.profile.interests.map(interest => {
      const timeDiff = now - new Date(interest.lastUpdated).getTime();
      const decayFactor = Math.exp(-timeDiff / (1000 * 60 * 60 * 24 * 14)); // 2-week decay
      return {
        ...interest,
        score: interest.score * decayFactor,
      };
    }).filter(interest => interest.score > 0.1); // Remove very low interests
  }

  private shuffleArray<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  private async saveProfile(): Promise<void> {
    if (this.profile) {
      await storageService.setData(`profile_${this.profile.userId}`, this.profile);
    }
  }

  private async saveBehaviors(): Promise<void> {
    const user = anonymousAuth.getCurrentUser();
    if (user) {
      await storageService.setData(`behaviors_${user.id}`, this.behaviors);
    }
  }

  // Export personalization data
  async exportPersonalizationData(): Promise<string> {
    const user = anonymousAuth.getCurrentUser();
    if (!user || !this.profile) return '';

    const exportData = {
      profile: this.profile,
      behaviors: this.behaviors.slice(-100), // Last 100 behaviors
      exportedAt: new Date().toISOString(),
    };

    return JSON.stringify(exportData, null, 2);
  }

  // Reset personalization
  async resetPersonalization(): Promise<void> {
    const user = anonymousAuth.getCurrentUser();
    if (!user) return;

    this.profile = this.createNewProfile(user.id);
    this.behaviors = [];
    
    await Promise.all([
      this.saveProfile(),
      this.saveBehaviors(),
    ]);
  }
}

export const personalizationEngine = PersonalizationEngine.getInstance();
