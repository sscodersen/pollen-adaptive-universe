// App Store Feed Algorithm - Enhanced with Bento Buzz 7-factor significance algorithm
import { significanceAlgorithm } from './significanceAlgorithm';
import { contentOrchestrator } from './contentOrchestrator';
import { pollenAI } from './pollenAI';
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
    
    // Generate apps using enhanced Bento Buzz algorithm
    const apps = await this.generateAppsWithBentoBuzz(count * 2, effectiveStrategy);

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

  // Enhanced app generation using Bento Buzz 7-factor significance algorithm
  private async generateAppsWithBentoBuzz(count: number, strategy: AppFeedStrategy): Promise<App[]> {
    const apps: App[] = [];
    const categories = [
      'Photography', 'Health & Fitness', 'Developer Tools', 'Music', 
      'Games', 'Productivity', 'Education', 'Business', 'Utilities', 'Entertainment',
      'Social', 'Finance', 'Travel', 'News', 'Shopping', 'Weather', 'Sports'
    ];
    
    // Use Bento Buzz algorithm to generate high-significance apps
    for (let i = 0; i < count; i++) {
      const category = categories[Math.floor(Math.random() * categories.length)];
      const prompt = this.generateAppPrompt(category, strategy);
      
      try {
        // Use enhanced pollenAI with Bento Buzz algorithm
        const aiResponse = await pollenAI.generate(prompt, 'app_store');
        const appData = aiResponse.content;
        
        if (appData && typeof appData === 'object') {
          const enhancedApp = this.processAIGeneratedApp(appData, category, strategy);
          if (enhancedApp.significance >= strategy.qualityThreshold) {
            apps.push(enhancedApp);
          }
        }
      } catch (error) {
        // Fallback to enhanced template generation
        const fallbackApp = await this.generateEnhancedTemplateApp(category, strategy);
        if (fallbackApp.significance >= strategy.qualityThreshold) {
          apps.push(fallbackApp);
        }
      }
    }
    
    // Ensure we have enough apps
    while (apps.length < count) {
      const category = categories[Math.floor(Math.random() * categories.length)];
      const fallbackApp = await this.generateEnhancedTemplateApp(category, strategy);
      apps.push(fallbackApp);
    }
    
    return apps;
  }

  // Generate app prompts for Bento Buzz algorithm
  private generateAppPrompt(category: string, strategy: AppFeedStrategy): string {
    const trendingKeywords = strategy.trendingBoost > 0.7 ? ['trending', 'viral', 'popular'] : [];
    const qualityKeywords = strategy.qualityThreshold > 8 ? ['premium', 'professional', 'advanced'] : ['innovative', 'smart'];
    
    const prompts = {
      'Photography': `${qualityKeywords.join(' ')} photo editing app with AI filters and professional tools`,
      'Health & Fitness': `${qualityKeywords.join(' ')} fitness tracking app with AI personal trainer features`,
      'Developer Tools': `${qualityKeywords.join(' ')} code editor with AI assistance and development tools`,
      'Music': `${qualityKeywords.join(' ')} music production app with AI composition capabilities`,
      'Games': `${qualityKeywords.join(' ')} mobile game with innovative gameplay mechanics`,
      'Productivity': `${qualityKeywords.join(' ')} productivity app with smart automation features`,
      'Education': `${qualityKeywords.join(' ')} learning app with AI tutoring and personalization`,
      'Business': `${qualityKeywords.join(' ')} business management app with analytics dashboard`,
      'Utilities': `${qualityKeywords.join(' ')} utility app with system optimization tools`,
      'Entertainment': `${qualityKeywords.join(' ')} entertainment app with streaming and social features`,
      'Social': `${qualityKeywords.join(' ')} social networking app with unique community features`,
      'Finance': `${qualityKeywords.join(' ')} finance app with investment tracking and AI insights`,
      'Travel': `${qualityKeywords.join(' ')} travel planning app with AI recommendations`,
      'News': `${qualityKeywords.join(' ')} news app with personalized content curation`,
      'Shopping': `${qualityKeywords.join(' ')} shopping app with AR try-on features`,
      'Weather': `${qualityKeywords.join(' ')} weather app with hyper-local forecasting`,
      'Sports': `${qualityKeywords.join(' ')} sports tracking app with performance analytics`
    };
    
    const basePrompt = prompts[category as keyof typeof prompts] || `${qualityKeywords.join(' ')} ${category.toLowerCase()} application`;
    return `${trendingKeywords.join(' ')} ${basePrompt}`;
  }

  // Process AI-generated app data with Bento Buzz enhancements
  private processAIGeneratedApp(appData: any, category: string, strategy: AppFeedStrategy): App {
    const isFree = strategy.freeAppWeight > 0.5 ? Math.random() > 0.4 : Math.random() > 0.6;
    const price = appData.price || (isFree ? 'Free' : `$${(Math.random() * 30 + 4.99).toFixed(2)}`);
    const hasDiscount = !isFree && appData.discount && appData.discount > 0;
    
    return {
      id: appData.id || `app-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name: appData.name || `${category} Pro`,
      description: appData.description || `Advanced ${category.toLowerCase()} application with cutting-edge features.`,
      category: appData.category || category,
      developer: appData.developer || `${appData.name?.split(' ')[0] || category} Studios`,
      rating: appData.rating || Number((Math.random() * 1.5 + 3.5).toFixed(1)),
      reviews: appData.reviews || Math.floor(Math.random() * 100000) + 1000,
      price: typeof price === 'string' ? price : `$${price}`,
      originalPrice: hasDiscount ? appData.originalPrice : undefined,
      discount: appData.discount || 0,
      downloadLink: appData.downloadLink || `https://apps.pollen.ai/${(appData.name || 'app').toLowerCase().replace(/\s+/g, '-')}`,
      iconUrl: appData.iconUrl || '/placeholder-app-icon.png',
      screenshots: appData.screenshots || ['/placeholder.svg', '/placeholder.svg'],
      size: appData.size || `${Math.floor(Math.random() * 150 + 25)}.${Math.floor(Math.random() * 9)}MB`,
      version: appData.version || `${Math.floor(Math.random() * 3 + 1)}.${Math.floor(Math.random() * 10)}.0`,
      updated: this.generateRandomDate(),
      tags: appData.tags || this.generateAppTags(category),
      inStock: true,
      trending: appData.trending || appData.significance > 8.5,
      significance: appData.significance || 8.0,
      downloads: this.generateDownloadCount(),
      rank: 0,
      featured: appData.significance > 9.0 || appData.editors_choice === true
    };
  }

  // Enhanced template app generation with Bento Buzz factors
  private async generateEnhancedTemplateApp(category: string, strategy: AppFeedStrategy): Promise<App> {
    const enhancedNames = {
      'Photography': ['Lens AI Pro', 'PhotoCraft Studio', 'Visual Master', 'PixelGenius'],
      'Health & Fitness': ['FitAI Trainer', 'HealthSync Pro', 'WellBeing Coach', 'ActiveMind'],
      'Developer Tools': ['CodeForge AI', 'DevMaster Pro', 'GitFlow Studio', 'API Wizard'],
      'Music': ['BeatCraft AI', 'SoundForge Pro', 'MusicMind Studio', 'AudioGenius'],
      'Games': ['GameCraft Pro', 'PlayMaster', 'QuestForge', 'ActionCraft'],
      'Productivity': ['TaskAI Pro', 'WorkFlow Master', 'ProductiveGenius', 'FocusCraft'],
      'Education': ['LearnAI Pro', 'StudyMaster', 'KnowledgeCraft', 'SkillForge'],
      'Business': ['BizAI Suite', 'WorkMaster Pro', 'BusinessCraft', 'ProfessionalAI'],
      'Utilities': ['ToolCraft Pro', 'SystemAI', 'UtilityMaster', 'HandyCraft'],
      'Entertainment': ['StreamCraft Pro', 'EntertainAI', 'MediaMaster', 'FunForge'],
      'Social': ['SocialCraft Pro', 'ConnectAI', 'CommunityMaster', 'NetworkForge'],
      'Finance': ['FinanceAI Pro', 'MoneyMaster', 'WealthCraft', 'InvestForge'],
      'Travel': ['TravelAI Pro', 'JourneyMaster', 'ExploreCraft', 'WanderForge'],
      'News': ['NewsAI Pro', 'InfoMaster', 'NewsCraft', 'MediaForge'],
      'Shopping': ['ShopAI Pro', 'BuyMaster', 'CommerceCraft', 'RetailForge'],
      'Weather': ['WeatherAI Pro', 'ClimateMaster', 'ForecastCraft', 'SkyForge'],
      'Sports': ['SportsAI Pro', 'AthleteMaster', 'FitnessCraft', 'SportForge']
    };

    const names = enhancedNames[category as keyof typeof enhancedNames] || ['AppCraft Pro'];
    const name = names[Math.floor(Math.random() * names.length)];
    
    // Calculate enhanced significance with Bento Buzz factors
    const bentoBuzzFactors = this.calculateBentoBuzzFactors(name, category, strategy);
    const significanceScore = significanceAlgorithm.calculateSignificance(bentoBuzzFactors);

    const isFree = strategy.freeAppWeight > 0.5 ? Math.random() > 0.3 : Math.random() > 0.6;
    const basePrice = Math.random() * 40 + 4.99;
    const price = isFree ? 'Free' : `$${basePrice.toFixed(2)}`;
    const hasDiscount = !isFree && Math.random() > 0.7;
    
    return {
      id: `app-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name,
      description: this.generateEnhancedDescription(name, category, significanceScore),
      category,
      developer: `${name.split(' ')[0]} Studios`,
      rating: Number((Math.random() * 1.5 + 3.5 + significanceScore / 10).toFixed(1)),
      reviews: Math.floor((significanceScore * 10000) + Math.random() * 50000) + 1000,
      price,
      originalPrice: hasDiscount ? `$${(basePrice * 1.4).toFixed(2)}` : undefined,
      discount: hasDiscount ? Math.floor(Math.random() * 40) + 15 : 0,
      downloadLink: `https://apps.pollen.ai/${name.toLowerCase().replace(/\s+/g, '-')}`,
      iconUrl: '/placeholder-app-icon.png',
      screenshots: ['/placeholder.svg', '/placeholder.svg', '/placeholder.svg'],
      size: `${Math.floor(Math.random() * 150 + 30)}.${Math.floor(Math.random() * 9)}MB`,
      version: `${Math.floor(Math.random() * 4 + 1)}.${Math.floor(Math.random() * 15)}.${Math.floor(Math.random() * 10)}`,
      updated: this.generateRandomDate(),
      tags: this.generateAppTags(category),
      inStock: true,
      trending: significanceScore > 8.5 || Math.random() > (1.0 - strategy.trendingBoost),
      significance: Math.max(significanceScore, strategy.qualityThreshold),
      downloads: this.generateDownloadCount(),
      rank: 0,
      featured: significanceScore > 9.0 && Math.random() > 0.7
    };
  }

  // Calculate Bento Buzz factors for apps
  private calculateBentoBuzzFactors(name: string, category: string, strategy: AppFeedStrategy) {
    const keywords = name.toLowerCase();
    
    // Enhanced scope based on app category and potential reach
    const scope = category === 'Social' || category === 'Productivity' ? 9 : 
                 category === 'Games' || category === 'Entertainment' ? 8 : 7;

    // Higher intensity for trending or premium apps
    const intensity = strategy.trendingBoost > 0.7 ? 8.5 : 
                     strategy.qualityThreshold > 8 ? 8 : 7;

    // App originality based on AI features and innovation
    const originalityKeywords = ['ai', 'pro', 'master', 'craft', 'forge', 'genius'];
    const originality = Math.min(10, originalityKeywords.filter(k => keywords.includes(k)).length * 1.5 + 7);

    // Apps have high immediacy (always available for download)
    const immediacy = 9;

    // High practicability for apps (direct value to users)
    const practicability = category === 'Productivity' || category === 'Utilities' ? 9.5 :
                          category === 'Education' || category === 'Health & Fitness' ? 9 : 8;

    // Positive aspects (apps generally solve problems)
    const positivity = strategy.freeAppWeight > 0.5 ? 8.5 : 8;

    // High credibility for curated app store content
    const credibility = 8.8;

    return { scope, intensity, originality, immediacy, practicability, positivity, credibility };
  }

  // Generate enhanced app descriptions
  private generateEnhancedDescription(name: string, category: string, significance: number): string {
    const qualityLevel = significance > 9 ? 'Revolutionary' :
                        significance > 8.5 ? 'Premium' :
                        significance > 8 ? 'Advanced' : 'Professional';
    
    const descriptions = {
      'Photography': `${qualityLevel} photo editing experience with AI-powered enhancement tools, professional filters, and intuitive controls that transform your photography workflow.`,
      'Health & Fitness': `${qualityLevel} fitness companion featuring personalized workout plans, health tracking, and AI-driven insights to help you achieve your wellness goals.`,
      'Developer Tools': `${qualityLevel} development environment with intelligent code assistance, seamless integrations, and powerful debugging tools for modern software development.`,
      'Music': `${qualityLevel} music creation suite with AI composition assistance, professional mixing tools, and comprehensive audio processing capabilities.`,
      'Games': `${qualityLevel} gaming experience featuring innovative mechanics, stunning visuals, and engaging gameplay that pushes the boundaries of mobile entertainment.`,
      'Productivity': `${qualityLevel} productivity solution with smart automation, seamless synchronization, and powerful organizational tools to optimize your workflow.`,
      'Education': `${qualityLevel} learning platform with personalized curricula, interactive content, and AI tutoring to accelerate your educational journey.`,
      'Business': `${qualityLevel} business management suite with comprehensive analytics, workflow automation, and professional tools for modern enterprises.`,
      'Utilities': `${qualityLevel} utility collection with system optimization, file management, and productivity enhancements for power users.`,
      'Entertainment': `${qualityLevel} entertainment platform with curated content, social features, and personalized recommendations for endless enjoyment.`
    };

    return descriptions[category as keyof typeof descriptions] || 
           `${qualityLevel} ${category.toLowerCase()} application with cutting-edge features and exceptional user experience.`;
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