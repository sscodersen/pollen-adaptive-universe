// App Store Feed Algorithm - Similar to Social Feed but optimized for apps
import { significanceAlgorithm } from './significanceAlgorithm';
import { contentOrchestrator } from './contentOrchestrator';
import { App } from '../components/AppStorePage';

export interface AppFeedStrategy {
  diversityBoost: number;      // 0-1: Preference for diverse app categories
  freeAppWeight: number;       // 0-1: How much to prioritize free apps
  trendingBoost: number;       // 0-1: Boost for trending apps
  qualityThreshold: number;    // 0-10: Minimum quality score
  personalizedWeight: number;  // 0-1: Future: personalization based on user behavior
  freshnessPriority: number;   // 0-1: Preference for recently updated apps
  downloadsWeight: number;     // 0-1: Weight for download count
}

export interface AppFeedFilters {
  category?: string;
  price?: 'free' | 'paid';
  rating?: number;
  trending?: boolean;
  featured?: boolean;
  developer?: string;
}

export interface AppFeedResponse {
  apps: App[];
  metadata: {
    totalApps: number;
    featuredCount: number;
    trendingCount: number;
    freeCount: number;
    avgRating: number;
    avgSignificance: number;
    strategy: AppFeedStrategy;
    timestamp: string;
  };
}

class AppStoreFeed {
  private defaultStrategy: AppFeedStrategy = {
    diversityBoost: 0.8,
    freeAppWeight: 0.3,
    trendingBoost: 0.7,
    qualityThreshold: 7.5,
    personalizedWeight: 0.6,
    freshnessPriority: 0.5,
    downloadsWeight: 0.4
  };

  // Main feed generation - like social feed but for apps
  async generateAppFeed(
    count: number = 20,
    filters?: AppFeedFilters,
    strategy?: Partial<AppFeedStrategy>
  ): Promise<AppFeedResponse> {
    const effectiveStrategy = { ...this.defaultStrategy, ...strategy };
    
    // Generate apps using content orchestrator (would need to extend for apps)
    const apps = await this.generateAppsWithAI(count * 2, effectiveStrategy);

    // Apply filters
    let filteredApps = filters ? this.applyFilters(apps, filters) : apps;

    // Apply feed algorithm ranking
    filteredApps = this.rankAppsForFeed(filteredApps, effectiveStrategy);

    // Ensure category diversity (like feed diversity)
    filteredApps = this.ensureCategoryDiversity(filteredApps, effectiveStrategy.diversityBoost);

    // Apply final selection
    const finalApps = filteredApps.slice(0, count);

    return {
      apps: finalApps,
      metadata: {
        totalApps: filteredApps.length,
        featuredCount: finalApps.filter(a => a.featured).length,
        trendingCount: finalApps.filter(a => a.trending).length,
        freeCount: finalApps.filter(a => a.price === 'Free').length,
        avgRating: this.calculateAverageRating(finalApps),
        avgSignificance: this.calculateAverageSignificance(finalApps),
        strategy: effectiveStrategy,
        timestamp: new Date().toISOString()
      }
    };
  }

  // Feed-style ranking algorithm for apps
  private rankAppsForFeed(apps: App[], strategy: AppFeedStrategy): App[] {
    return apps.sort((a, b) => {
      // 1. Featured apps first (like pinned posts)
      if (a.featured && !b.featured) return -1;
      if (b.featured && !a.featured) return 1;

      // 2. Trending boost (like viral posts)
      const aTrendingScore = a.trending ? strategy.trendingBoost : 0;
      const bTrendingScore = b.trending ? strategy.trendingBoost : 0;

      // 3. Free app appeal (accessibility)
      const aFreeScore = a.price === 'Free' ? strategy.freeAppWeight : 0;
      const bFreeScore = b.price === 'Free' ? strategy.freeAppWeight : 0;

      // 4. Base significance (like post engagement)
      const aSignificance = a.significance * 0.35;
      const bSignificance = b.significance * 0.35;

      // 5. Rating quality (like post quality)
      const aRatingScore = a.rating * 0.2;
      const bRatingScore = b.rating * 0.2;

      // 6. Download popularity (like post views)
      const aDownloadScore = this.normalizeDownloads(a.downloads) * strategy.downloadsWeight;
      const bDownloadScore = this.normalizeDownloads(b.downloads) * strategy.downloadsWeight;

      // 7. Recent updates (freshness)
      const aFreshnessScore = this.calculateFreshness(a.updated) * strategy.freshnessPriority;
      const bFreshnessScore = this.calculateFreshness(b.updated) * strategy.freshnessPriority;

      // 8. Discount appeal (if paid app has discount)
      const aDiscountScore = (a.discount || 0) * 0.05;
      const bDiscountScore = (b.discount || 0) * 0.05;

      const aFinalScore = aSignificance + aTrendingScore + aFreeScore + 
                         aRatingScore + aDownloadScore + aFreshnessScore + aDiscountScore;
      const bFinalScore = bSignificance + bTrendingScore + bFreeScore + 
                         bRatingScore + bDownloadScore + bFreshnessScore + bDiscountScore;

      return bFinalScore - aFinalScore;
    });
  }

  // Generate apps with AI (similar to content generation)
  private async generateAppsWithAI(count: number, strategy: AppFeedStrategy): Promise<App[]> {
    const categories = [
      'Photography', 'Health & Fitness', 'Developer Tools', 'Music', 
      'Games', 'Productivity', 'Education', 'Business', 'Utilities', 'Entertainment'
    ];
    
    const apps: App[] = [];
    
    for (let i = 0; i < count; i++) {
      const category = categories[Math.floor(Math.random() * categories.length)];
      const app = await this.generateSingleApp(category, strategy);
      apps.push(app);
    }
    
    return apps;
  }

  private async generateSingleApp(category: string, strategy: AppFeedStrategy): Promise<App> {
    // This would use AI to generate app concepts (simplified version)
    const appNames = {
      'Photography': ['PhotoMaster Pro', 'LensArt Studio', 'PixelPerfect', 'ViewCraft'],
      'Health & Fitness': ['FitTracker AI', 'HealthHub Pro', 'WellnessCoach', 'ActiveLife'],
      'Developer Tools': ['CodeCraft Pro', 'DevHelper AI', 'GitMaster', 'APITester'],
      'Music': ['SoundStudio Pro', 'BeatMaker AI', 'MusicCraft', 'AudioWave'],
      'Games': ['GameCraft Pro', 'AdventureQuest', 'PuzzleMaster', 'ActionHero'],
      'Productivity': ['TaskMaster Pro', 'WorkFlow AI', 'ProductivePro', 'FocusTime'],
      'Education': ['LearnPro AI', 'StudyHelper', 'KnowledgeHub', 'SkillCraft'],
      'Business': ['BizPro Suite', 'WorkSmart AI', 'BusinessHub', 'ProfessionalPro'],
      'Utilities': ['ToolKit Pro', 'SystemHelper', 'UtilityMaster', 'HandyTools'],
      'Entertainment': ['StreamPro', 'EntertainHub', 'MediaCraft', 'FunTime']
    };

    const names = appNames[category as keyof typeof appNames] || ['Generic App Pro'];
    const name = names[Math.floor(Math.random() * names.length)];
    
    // Generate significance score
    const significanceScore = significanceAlgorithm.calculateSignificance({
      scope: Math.random() * 10,
      intensity: Math.random() * 10,
      originality: Math.random() * 10,
      immediacy: Math.random() * 10,
      practicability: Math.random() * 10,
      positivity: Math.random() * 10,
      credibility: Math.random() * 10
    });

    const isFree = Math.random() > 0.6;
    const price = isFree ? 'Free' : `$${Math.floor(Math.random() * 50) + 1}.99`;
    const hasDiscount = !isFree && Math.random() > 0.7;
    
    return {
      id: `app-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name,
      description: `Advanced ${category.toLowerCase()} app with AI-powered features and professional capabilities.`,
      category,
      developer: `${name.split(' ')[0]} Studios`,
      rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
      reviews: Math.floor(Math.random() * 50000) + 500,
      price,
      originalPrice: hasDiscount ? `$${Math.floor(Math.random() * 30) + parseInt(price.replace('$', '')) + 10}.99` : undefined,
      discount: hasDiscount ? Math.floor(Math.random() * 40) + 10 : 0,
      downloadLink: `https://example.com/app/${name.toLowerCase().replace(/\s+/g, '-')}`,
      iconUrl: '/placeholder-app-icon.png',
      screenshots: [],
      size: `${Math.floor(Math.random() * 200) + 10}.${Math.floor(Math.random() * 9)}MB`,
      version: `${Math.floor(Math.random() * 5) + 1}.${Math.floor(Math.random() * 20)}.${Math.floor(Math.random() * 10)}`,
      updated: this.generateRandomDate(),
      tags: this.generateAppTags(category),
      inStock: true,
      trending: significanceScore > 8.0 || Math.random() > 0.8,
      significance: significanceScore,
      downloads: this.generateDownloadCount(),
      rank: 0,
      featured: significanceScore > 9.0 && Math.random() > 0.7
    };
  }

  // Ensure category diversity like social feed ensures content diversity
  private ensureCategoryDiversity(apps: App[], diversityBoost: number): App[] {
    if (diversityBoost < 0.5) return apps;

    const categorized: { [key: string]: App[] } = {};
    const result: App[] = [];
    
    // Group by category
    apps.forEach(app => {
      if (!categorized[app.category]) {
        categorized[app.category] = [];
      }
      categorized[app.category].push(app);
    });

    // Distribute apps across categories (round-robin style)
    const categories = Object.keys(categorized);
    let maxRounds = Math.max(...Object.values(categorized).map(arr => arr.length));
    
    for (let round = 0; round < maxRounds; round++) {
      for (const category of categories) {
        if (categorized[category][round]) {
          result.push(categorized[category][round]);
        }
      }
    }

    return result;
  }

  private applyFilters(apps: App[], filters: AppFeedFilters): App[] {
    return apps.filter(app => {
      if (filters.category && app.category !== filters.category) return false;
      if (filters.price === 'free' && app.price !== 'Free') return false;
      if (filters.price === 'paid' && app.price === 'Free') return false;
      if (filters.trending !== undefined && app.trending !== filters.trending) return false;
      if (filters.featured !== undefined && app.featured !== filters.featured) return false;
      if (filters.rating && app.rating < filters.rating) return false;
      if (filters.developer && app.developer !== filters.developer) return false;
      
      return true;
    });
  }

  private normalizeDownloads(downloads: string): number {
    // Convert download string to normalized score
    const num = downloads.replace(/[^\d.]/g, '');
    const multiplier = downloads.includes('M') ? 1000000 : 
                      downloads.includes('K') ? 1000 : 1;
    return (parseFloat(num) || 0) * multiplier / 1000000; // Normalize to millions
  }

  private calculateFreshness(updated: string): number {
    // Simple freshness calculation - more recent = higher score
    const now = new Date();
    const updateDate = new Date(updated);
    const diffDays = (now.getTime() - updateDate.getTime()) / (1000 * 60 * 60 * 24);
    
    if (diffDays < 7) return 1.0;
    if (diffDays < 30) return 0.8;
    if (diffDays < 90) return 0.6;
    if (diffDays < 365) return 0.4;
    return 0.2;
  }

  private generateRandomDate(): string {
    const now = new Date();
    const daysAgo = Math.floor(Math.random() * 365);
    const date = new Date(now.getTime() - daysAgo * 24 * 60 * 60 * 1000);
    return date.toISOString().split('T')[0];
  }

  private generateAppTags(category: string): string[] {
    const baseTags = ['Premium', 'Professional', 'AI-Powered'];
    const categoryTags = {
      'Photography': ['Filters', 'Editing', 'RAW'],
      'Health & Fitness': ['Tracking', 'Workout', 'Nutrition'],
      'Developer Tools': ['Code', 'Debug', 'Deploy'],
      'Music': ['Audio', 'Composition', 'Studio'],
      'Games': ['Adventure', 'Multiplayer', 'Casual'],
      'Productivity': ['Tasks', 'Organization', 'Efficiency'],
      'Education': ['Learning', 'Courses', 'Skills'],
      'Business': ['Enterprise', 'Analytics', 'CRM'],
      'Utilities': ['System', 'Maintenance', 'Tools'],
      'Entertainment': ['Streaming', 'Media', 'Fun']
    };
    
    const specific = categoryTags[category as keyof typeof categoryTags] || ['General'];
    return [...baseTags.slice(0, 2), ...specific.slice(0, 2)];
  }

  private generateDownloadCount(): string {
    const base = Math.floor(Math.random() * 999) + 1;
    const multipliers = ['K', 'M'];
    const multiplier = multipliers[Math.floor(Math.random() * multipliers.length)];
    return `${base}${multiplier}`;
  }

  private calculateAverageRating(apps: App[]): number {
    if (apps.length === 0) return 0;
    const sum = apps.reduce((acc, app) => acc + app.rating, 0);
    return Math.round((sum / apps.length) * 100) / 100;
  }

  private calculateAverageSignificance(apps: App[]): number {
    if (apps.length === 0) return 0;
    const sum = apps.reduce((acc, app) => acc + app.significance, 0);
    return Math.round((sum / apps.length) * 100) / 100;
  }

  // Get trending apps (like trending posts)
  async getTrendingApps(count: number = 10): Promise<App[]> {
    const response = await this.generateAppFeed(count * 2, { trending: true }, {
      trendingBoost: 1.0,
      qualityThreshold: 8.0
    });
    
    return response.apps.slice(0, count);
  }

  // Get featured apps (like pinned posts)
  async getFeaturedApps(count: number = 5): Promise<App[]> {
    const response = await this.generateAppFeed(count * 2, { featured: true }, {
      qualityThreshold: 9.0,
      trendingBoost: 0.9
    });
    
    return response.apps.slice(0, count);
  }

  // Get free apps (popular filter)
  async getFreeApps(count: number = 15): Promise<App[]> {
    const response = await this.generateAppFeed(count * 2, { price: 'free' }, {
      freeAppWeight: 1.0,
      qualityThreshold: 7.0
    });
    
    return response.apps.slice(0, count);
  }
}

export const appStoreFeed = new AppStoreFeed();