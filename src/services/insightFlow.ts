/**
 * InsightFlow Algorithm - Advanced Content Significance Rating System
 * 
 * Evaluates content using 7-factor significance rating:
 * 1. Relevance - How relevant is the content to current trends and user interests
 * 2. Quality - Content quality and production value
 * 3. Innovation - How innovative or unique is the content
 * 4. Timeliness - How timely and current is the content
 * 5. Utility - Practical value and usefulness to users
 * 6. Engagement - Predicted engagement potential
 * 7. Credibility - Source reliability and content accuracy
 * 
 * Only content with significance > 7.0 is shown to users
 */

export interface ContentItem {
  id: string;
  title: string;
  description: string;
  content?: string;
  category: string;
  tags?: string[];
  source?: string;
  author?: string;
  publishedAt?: string;
  views?: number;
  likes?: number;
  shares?: number;
  price?: number;
  rating?: number;
  type: 'product' | 'app' | 'content' | 'service' | 'tool';
}

export interface SignificanceFactors {
  relevance: number;      // 0-10: How relevant to current trends/interests
  quality: number;        // 0-10: Content/product quality
  innovation: number;     // 0-10: How innovative/unique
  timeliness: number;     // 0-10: How current/timely
  utility: number;        // 0-10: Practical usefulness
  engagement: number;     // 0-10: Predicted engagement potential  
  credibility: number;    // 0-10: Source reliability/accuracy
}

export interface ScoredContent extends ContentItem {
  significance: number;
  factors: SignificanceFactors;
  reasoning: string;
  confidence: number;
}

class InsightFlowAlgorithm {
  private readonly SIGNIFICANCE_THRESHOLD = 7.0;
  private readonly TRENDING_KEYWORDS = [
    'ai', 'artificial intelligence', 'machine learning', 'automation', 'blockchain',
    'sustainability', 'climate', 'renewable', 'innovation', 'technology',
    'productivity', 'efficiency', 'optimization', 'real-time', 'analytics',
    'social', 'community', 'collaboration', 'creator', 'content generation'
  ];

  /**
   * Calculate relevance score based on trends, keywords, and user interests
   */
  private calculateRelevance(item: ContentItem): number {
    let score = 5.0; // Base score
    
    const textToAnalyze = `${item.title} ${item.description} ${item.tags?.join(' ') || ''}`.toLowerCase();
    
    // Check for trending keywords
    const trendingMatches = this.TRENDING_KEYWORDS.filter(keyword => 
      textToAnalyze.includes(keyword)
    ).length;
    score += Math.min(trendingMatches * 0.8, 3.0);
    
    // Recent content gets relevance boost
    if (item.publishedAt) {
      const daysSincePublished = (Date.now() - new Date(item.publishedAt).getTime()) / (1000 * 60 * 60 * 24);
      if (daysSincePublished < 7) score += 1.5;
      else if (daysSincePublished < 30) score += 0.8;
    }
    
    // Category-specific boosts
    if (['ai', 'technology', 'productivity', 'innovation'].includes(item.category.toLowerCase())) {
      score += 1.2;
    }
    
    return Math.min(score, 10);
  }

  /**
   * Calculate quality score based on content characteristics
   */
  private calculateQuality(item: ContentItem): number {
    let score = 6.0; // Base score
    
    // Title and description quality indicators
    const titleWords = item.title.split(' ').length;
    const descWords = item.description.split(' ').length;
    
    if (titleWords >= 3 && titleWords <= 12) score += 1.0;
    if (descWords >= 20) score += 1.0;
    if (descWords >= 50) score += 0.5;
    
    // Rating-based quality (if available)
    if (item.rating && item.rating >= 4.0) score += 1.5;
    if (item.rating && item.rating >= 4.5) score += 0.5;
    
    // Content completeness
    if (item.tags && item.tags.length >= 3) score += 0.5;
    if (item.content && item.content.length > 100) score += 0.5;
    
    // Engagement metrics as quality indicators
    if (item.views && item.views > 1000) score += 0.5;
    if (item.likes && item.likes > 100) score += 0.5;
    
    return Math.min(score, 10);
  }

  /**
   * Calculate innovation score based on uniqueness and creativity
   */
  private calculateInnovation(item: ContentItem): number {
    let score = 5.0; // Base score
    
    const textToAnalyze = `${item.title} ${item.description}`.toLowerCase();
    
    // Innovation keywords
    const innovationKeywords = [
      'breakthrough', 'revolutionary', 'cutting-edge', 'innovative', 'novel',
      'first-of-its-kind', 'groundbreaking', 'pioneering', 'disruptive',
      'next-generation', 'advanced', 'experimental', 'prototype', 'beta'
    ];
    
    const innovationMatches = innovationKeywords.filter(keyword => 
      textToAnalyze.includes(keyword)
    ).length;
    score += Math.min(innovationMatches * 0.7, 2.5);
    
    // Category-specific innovation scoring
    if (['ai', 'technology', 'research', 'experimental'].includes(item.category.toLowerCase())) {
      score += 1.0;
    }
    
    // Recency can indicate innovation
    if (item.publishedAt) {
      const daysSincePublished = (Date.now() - new Date(item.publishedAt).getTime()) / (1000 * 60 * 60 * 24);
      if (daysSincePublished < 30) score += 1.0;
    }
    
    return Math.min(score, 10);
  }

  /**
   * Calculate timeliness score based on recency and current relevance
   */
  private calculateTimeliness(item: ContentItem): number {
    let score = 5.0; // Base score
    
    if (item.publishedAt) {
      const daysSincePublished = (Date.now() - new Date(item.publishedAt).getTime()) / (1000 * 60 * 60 * 24);
      
      if (daysSincePublished < 1) score += 3.0;        // Today
      else if (daysSincePublished < 7) score += 2.0;   // This week
      else if (daysSincePublished < 30) score += 1.0;  // This month
      else if (daysSincePublished < 90) score += 0.5;  // Last 3 months
      else score -= 1.0; // Older content loses timeliness
    }
    
    // Real-time indicators
    const textToAnalyze = `${item.title} ${item.description}`.toLowerCase();
    const timelinessKeywords = [
      'live', 'real-time', 'now', 'latest', 'breaking', 'update',
      'current', 'today', 'this week', '2024', '2025', 'recent'
    ];
    
    const timelinessMatches = timelinessKeywords.filter(keyword => 
      textToAnalyze.includes(keyword)
    ).length;
    score += Math.min(timelinessMatches * 0.5, 2.0);
    
    return Math.min(score, 10);
  }

  /**
   * Calculate utility score based on practical value and usefulness
   */
  private calculateUtility(item: ContentItem): number {
    let score = 5.0; // Base score
    
    const textToAnalyze = `${item.title} ${item.description}`.toLowerCase();
    
    // Utility keywords
    const utilityKeywords = [
      'tool', 'solution', 'helps', 'improves', 'optimize', 'efficient',
      'productive', 'useful', 'practical', 'guide', 'tutorial', 'how-to',
      'automation', 'save time', 'streamline', 'enhance', 'boost'
    ];
    
    const utilityMatches = utilityKeywords.filter(keyword => 
      textToAnalyze.includes(keyword)
    ).length;
    score += Math.min(utilityMatches * 0.6, 3.0);
    
    // Type-specific utility scoring
    if (item.type === 'tool') score += 2.0;
    if (item.type === 'service') score += 1.5;
    if (item.type === 'app') score += 1.0;
    
    // Price as utility indicator (reasonable pricing)
    if (item.price !== undefined) {
      if (item.price === 0) score += 1.0; // Free tools are highly useful
      if (item.price > 0 && item.price < 50) score += 0.5; // Affordable
    }
    
    return Math.min(score, 10);
  }

  /**
   * Calculate engagement score based on predicted user interaction
   */
  private calculateEngagement(item: ContentItem): number {
    let score = 5.0; // Base score
    
    // Historical engagement metrics
    if (item.views) score += Math.min(Math.log10(item.views) / 2, 2.0);
    if (item.likes) score += Math.min(Math.log10(item.likes), 1.5);
    if (item.shares) score += Math.min(item.shares / 10, 1.5);
    
    const textToAnalyze = `${item.title} ${item.description}`.toLowerCase();
    
    // Engagement-driving keywords
    const engagementKeywords = [
      'amazing', 'incredible', 'must-see', 'viral', 'trending', 'popular',
      'interactive', 'engaging', 'fun', 'exciting', 'game-changing',
      'share', 'discover', 'explore', 'create', 'build', 'design'
    ];
    
    const engagementMatches = engagementKeywords.filter(keyword => 
      textToAnalyze.includes(keyword)
    ).length;
    score += Math.min(engagementMatches * 0.4, 2.0);
    
    // Visual content tends to be more engaging
    if (item.category.toLowerCase().includes('visual') || 
        item.category.toLowerCase().includes('video') ||
        item.category.toLowerCase().includes('image')) {
      score += 1.0;
    }
    
    return Math.min(score, 10);
  }

  /**
   * Calculate credibility score based on source reliability and content accuracy
   */
  private calculateCredibility(item: ContentItem): number {
    let score = 6.0; // Base score
    
    // Source credibility
    if (item.source) {
      const trustedSources = [
        'github', 'arxiv', 'nature', 'science', 'ieee', 'acm',
        'google', 'microsoft', 'openai', 'anthropic', 'meta',
        'university', 'research', 'official', 'verified'
      ];
      
      const sourceText = item.source.toLowerCase();
      const trustedMatches = trustedSources.filter(trusted => 
        sourceText.includes(trusted)
      ).length;
      score += Math.min(trustedMatches * 0.8, 2.0);
    }
    
    // Rating as credibility indicator
    if (item.rating && item.rating >= 4.0) score += 1.0;
    
    // Author credibility (if available)
    if (item.author) {
      const authorText = item.author.toLowerCase();
      if (authorText.includes('verified') || authorText.includes('expert')) {
        score += 1.0;
      }
    }
    
    // Content quality indicators for credibility
    const textToAnalyze = `${item.title} ${item.description}`.toLowerCase();
    const credibilityKeywords = [
      'research', 'study', 'data', 'evidence', 'peer-reviewed',
      'scientific', 'verified', 'tested', 'proven', 'analysis'
    ];
    
    const credibilityMatches = credibilityKeywords.filter(keyword => 
      textToAnalyze.includes(keyword)
    ).length;
    score += Math.min(credibilityMatches * 0.5, 1.5);
    
    return Math.min(score, 10);
  }

  /**
   * Calculate overall significance score using weighted factors
   */
  private calculateSignificance(factors: SignificanceFactors): number {
    // Weighted calculation - adjust weights based on content type and context
    const weights = {
      relevance: 0.20,    // 20% - How relevant to current context
      quality: 0.18,      // 18% - Content quality
      innovation: 0.15,   // 15% - Innovation factor
      timeliness: 0.12,   // 12% - How current/timely
      utility: 0.15,      // 15% - Practical value
      engagement: 0.10,   // 10% - Engagement potential
      credibility: 0.10   // 10% - Source credibility
    };
    
    const significance = 
      factors.relevance * weights.relevance +
      factors.quality * weights.quality +
      factors.innovation * weights.innovation +
      factors.timeliness * weights.timeliness +
      factors.utility * weights.utility +
      factors.engagement * weights.engagement +
      factors.credibility * weights.credibility;
    
    return Math.round(significance * 100) / 100; // Round to 2 decimal places
  }

  /**
   * Generate reasoning for the significance score
   */
  private generateReasoning(item: ContentItem, factors: SignificanceFactors, significance: number): string {
    const reasons = [];
    
    if (factors.relevance >= 8) reasons.push("highly relevant to current trends");
    if (factors.quality >= 8) reasons.push("exceptional content quality");
    if (factors.innovation >= 8) reasons.push("innovative and groundbreaking");
    if (factors.timeliness >= 8) reasons.push("very timely and current");
    if (factors.utility >= 8) reasons.push("extremely useful and practical");
    if (factors.engagement >= 8) reasons.push("high engagement potential");
    if (factors.credibility >= 8) reasons.push("highly credible source");
    
    if (reasons.length === 0) {
      if (significance >= 7) reasons.push("solid overall score across multiple factors");
      else reasons.push("below significance threshold for display");
    }
    
    return `Score: ${significance}/10. ${reasons.join(", ")}.`;
  }

  /**
   * Analyze a single content item and return scored result
   */
  public async analyzeContent(item: ContentItem): Promise<ScoredContent> {
    const factors: SignificanceFactors = {
      relevance: this.calculateRelevance(item),
      quality: this.calculateQuality(item),
      innovation: this.calculateInnovation(item),
      timeliness: this.calculateTimeliness(item),
      utility: this.calculateUtility(item),
      engagement: this.calculateEngagement(item),
      credibility: this.calculateCredibility(item)
    };
    
    const significance = this.calculateSignificance(factors);
    const reasoning = this.generateReasoning(item, factors, significance);
    
    // Confidence based on data availability and factor consistency
    const factorValues = Object.values(factors);
    const avgFactor = factorValues.reduce((sum, val) => sum + val, 0) / factorValues.length;
    const variance = factorValues.reduce((sum, val) => sum + Math.pow(val - avgFactor, 2), 0) / factorValues.length;
    const confidence = Math.max(0.6, Math.min(0.95, 1 - (variance / 10))); // Lower variance = higher confidence
    
    return {
      ...item,
      significance,
      factors,
      reasoning,
      confidence
    };
  }

  /**
   * Analyze multiple content items and return only those above significance threshold
   */
  public async analyzeContentBatch(items: ContentItem[]): Promise<ScoredContent[]> {
    const scoredItems = await Promise.all(
      items.map(item => this.analyzeContent(item))
    );
    
    // Filter by significance threshold and sort by score
    return scoredItems
      .filter(item => item.significance > this.SIGNIFICANCE_THRESHOLD)
      .sort((a, b) => b.significance - a.significance);
  }

  /**
   * Get content recommendations based on significance scoring
   */
  public async getRecommendations(items: ContentItem[], limit: number = 20): Promise<ScoredContent[]> {
    const significantItems = await this.analyzeContentBatch(items);
    return significantItems.slice(0, limit);
  }

  /**
   * Get significance threshold
   */
  public getThreshold(): number {
    return this.SIGNIFICANCE_THRESHOLD;
  }
}

// Export singleton instance
export const insightFlow = new InsightFlowAlgorithm();