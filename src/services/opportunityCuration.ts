import { pollenAI } from './pollenAI';

export interface Opportunity {
  id: string;
  type: 'real-estate' | 'app' | 'product' | 'news' | 'trend' | 'travel' | 'lifestyle' | 'investment';
  title: string;
  description: string;
  relevanceScore: number;
  qualityScore: number;
  urgency: 'low' | 'medium' | 'high';
  category: string;
  tags: string[];
  source: string;
  dateAdded: string;
  expiryDate?: string;
  aiInsights: string[];
  estimatedValue?: string;
  actionRequired?: string;
  imageUrl?: string;
}

export interface TrendOpportunity extends Opportunity {
  trendingScore: number;
  momentum: 'rising' | 'stable' | 'declining';
  relatedTopics: string[];
  predictedPeak: string;
}

export interface RealEstateOpportunity extends Opportunity {
  location: string;
  priceRange: string;
  roi: number;
  marketTrend: 'bullish' | 'neutral' | 'bearish';
  keyFeatures: string[];
}

class OpportunityCurationService {
  private opportunities: Opportunity[] = [];

  async getCuratedOpportunities(
    type?: Opportunity['type'],
    limit: number = 20
  ): Promise<Opportunity[]> {
    if (this.opportunities.length === 0) {
      await this.loadOpportunities();
    }

    let filtered = type 
      ? this.opportunities.filter(o => o.type === type)
      : this.opportunities;

    return filtered
      .sort((a, b) => {
        const scoreA = (a.relevanceScore * 0.5) + (a.qualityScore * 0.5);
        const scoreB = (b.relevanceScore * 0.5) + (b.qualityScore * 0.5);
        return scoreB - scoreA;
      })
      .slice(0, limit);
  }

  async getTrendingOpportunities(): Promise<TrendOpportunity[]> {
    return [
      {
        id: 'trend-1',
        type: 'trend',
        title: 'AI-Powered Personal Health Assistants',
        description: 'Rising demand for AI health monitoring tools that provide personalized wellness recommendations.',
        relevanceScore: 9.5,
        qualityScore: 9.2,
        urgency: 'high',
        category: 'Health Tech',
        tags: ['AI', 'healthcare', 'wearables', 'personalization'],
        source: 'Market Research AI',
        dateAdded: new Date().toISOString(),
        aiInsights: [
          'Market expected to grow 45% annually',
          'Strong consumer demand in 25-45 age group',
          'Integration with existing health apps is key differentiator'
        ],
        trendingScore: 9.3,
        momentum: 'rising',
        relatedTopics: ['Wearable tech', 'Preventive healthcare', 'Machine learning'],
        predictedPeak: '2026 Q2'
      },
      {
        id: 'trend-2',
        type: 'trend',
        title: 'Sustainable Fashion Marketplaces',
        description: 'Growing consumer preference for eco-friendly clothing and circular fashion economy platforms.',
        relevanceScore: 8.8,
        qualityScore: 8.9,
        urgency: 'medium',
        category: 'Sustainability',
        tags: ['fashion', 'sustainability', 'marketplace', 'circular-economy'],
        source: 'Trend Analysis Bot',
        dateAdded: new Date().toISOString(),
        aiInsights: [
          'Gen Z driving 60% of sustainable fashion purchases',
          'Resale market projected to double by 2027',
          'Brand transparency is critical success factor'
        ],
        trendingScore: 8.7,
        momentum: 'rising',
        relatedTopics: ['Vintage fashion', 'Upcycling', 'Eco-conscious consumers'],
        predictedPeak: '2025 Q4'
      }
    ];
  }

  async getRealEstateOpportunities(): Promise<RealEstateOpportunity[]> {
    return [
      {
        id: 're-1',
        type: 'real-estate',
        title: 'Emerging Tech Hub Properties - Austin, TX',
        description: 'Investment opportunity in rapidly growing tech corridor with strong rental demand.',
        relevanceScore: 9.0,
        qualityScore: 8.8,
        urgency: 'high',
        category: 'Commercial Real Estate',
        tags: ['tech-hub', 'high-growth', 'rental-income'],
        source: 'Real Estate AI',
        dateAdded: new Date().toISOString(),
        aiInsights: [
          '23% property value increase over 3 years',
          'Major tech companies expanding presence',
          'Below-market entry points still available'
        ],
        location: 'Austin, TX',
        priceRange: '$450K - $750K',
        roi: 8.5,
        marketTrend: 'bullish',
        keyFeatures: ['High rental demand', 'Tech sector growth', 'Strong infrastructure']
      }
    ];
  }

  async evaluateOpportunity(opportunity: Partial<Opportunity>): Promise<{
    relevanceScore: number;
    qualityScore: number;
    insights: string[];
    recommended: boolean;
  }> {
    try {
      const response = await pollenAI.generate(
        `Evaluate this opportunity: ${opportunity.title}. ${opportunity.description}`,
        'evaluation',
        { type: opportunity.type }
      );

      const relevanceScore = response.confidence * 10;
      const qualityScore = this.calculateQualityScore(opportunity);

      return {
        relevanceScore,
        qualityScore,
        insights: this.generateInsights(opportunity, relevanceScore),
        recommended: relevanceScore > 7 && qualityScore > 7
      };
    } catch (error) {
      return {
        relevanceScore: 7.5,
        qualityScore: 7.5,
        insights: ['Standard opportunity - requires further analysis'],
        recommended: true
      };
    }
  }

  private async loadOpportunities(): Promise<void> {
    this.opportunities = [
      {
        id: 'opp-1',
        type: 'app',
        title: 'Voice-Controlled Smart Home App',
        description: 'New platform integrating all smart home devices with advanced voice AI.',
        relevanceScore: 9.0,
        qualityScore: 8.7,
        urgency: 'high',
        category: 'Smart Home',
        tags: ['IoT', 'voice-AI', 'automation'],
        source: 'App Store Trends',
        dateAdded: new Date().toISOString(),
        aiInsights: ['First-mover advantage in unified control', 'Strong VC interest'],
        estimatedValue: '$2M - $5M market opportunity',
        actionRequired: 'Beta testing signup closes in 7 days'
      },
      {
        id: 'opp-2',
        type: 'product',
        title: 'Biodegradable Phone Cases',
        description: 'Eco-friendly phone protection made from plant-based materials.',
        relevanceScore: 8.5,
        qualityScore: 9.0,
        urgency: 'medium',
        category: 'Sustainable Tech',
        tags: ['eco-friendly', 'accessories', 'innovation'],
        source: 'Product Hunt',
        dateAdded: new Date().toISOString(),
        aiInsights: ['Growing demand for sustainable accessories', '40% cheaper than competitors'],
        estimatedValue: 'Early bird pricing available'
      },
      {
        id: 'opp-3',
        type: 'travel',
        title: 'Off-Season Mediterranean Cruises',
        description: 'Luxury cruises at 60% discount during shoulder season with perfect weather.',
        relevanceScore: 8.0,
        qualityScore: 8.5,
        urgency: 'high',
        category: 'Travel Deals',
        tags: ['cruise', 'luxury', 'discount'],
        source: 'Travel AI',
        dateAdded: new Date().toISOString(),
        expiryDate: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString(),
        aiInsights: ['Less crowded destinations', 'Same quality at fraction of peak price'],
        actionRequired: 'Book within 14 days'
      },
      {
        id: 'opp-4',
        type: 'news',
        title: 'New AI Regulation Framework Announced',
        description: 'Government unveils comprehensive AI governance guidelines affecting tech startups.',
        relevanceScore: 9.2,
        qualityScore: 9.5,
        urgency: 'high',
        category: 'Policy & Regulation',
        tags: ['AI-regulation', 'compliance', 'policy'],
        source: 'News Aggregator AI',
        dateAdded: new Date().toISOString(),
        aiInsights: ['Compliance deadline in 6 months', 'Affects all AI-powered businesses'],
        actionRequired: 'Review requirements and plan compliance strategy'
      },
      {
        id: 'opp-5',
        type: 'lifestyle',
        title: 'Personalized Meal Prep Services',
        description: 'AI-customized meal plans delivered weekly based on health goals and preferences.',
        relevanceScore: 8.3,
        qualityScore: 8.4,
        urgency: 'low',
        category: 'Health & Wellness',
        tags: ['nutrition', 'meal-prep', 'personalization'],
        source: 'Lifestyle Trends',
        dateAdded: new Date().toISOString(),
        aiInsights: ['20% discount for first-time users', 'Highly rated by nutritionists'],
        estimatedValue: 'Save 5 hours weekly'
      }
    ];
  }

  private calculateQualityScore(opportunity: Partial<Opportunity>): number {
    let score = 5;
    
    if (opportunity.source && opportunity.source.includes('AI')) score += 1.5;
    if (opportunity.tags && opportunity.tags.length > 3) score += 1;
    if (opportunity.urgency === 'high') score += 1.5;
    if (opportunity.aiInsights && opportunity.aiInsights.length > 2) score += 1;
    
    return Math.min(10, score);
  }

  private generateInsights(opportunity: Partial<Opportunity>, score: number): string[] {
    const insights: string[] = [];

    if (score > 8.5) {
      insights.push('High-value opportunity with strong potential');
    }
    if (opportunity.urgency === 'high') {
      insights.push('Time-sensitive - immediate action recommended');
    }
    if (opportunity.type === 'trend') {
      insights.push('Early adoption could provide competitive advantage');
    }

    return insights;
  }
}

export const opportunityCurationService = new OpportunityCurationService();
