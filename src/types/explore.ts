export type ExploreSection = 'trending' | 'real-estate' | 'content-creator' | 'all';

export interface ContentCreationRequest {
  topic: string;
  platform?: 'instagram' | 'tiktok' | 'youtube' | 'facebook';
  targetAudience?: string;
  tone?: 'professional' | 'casual' | 'enthusiastic' | 'educational';
}

export interface TrendingData {
  id: string;
  title: string;
  description: string;
  trendingScore: number;
  momentum: 'rising' | 'stable' | 'declining';
  relatedTopics: string[];
  aiInsights: string[];
  predictedPeak: string;
  category: string;
}

export interface RealEstateData {
  id: string;
  title: string;
  location: string;
  priceRange: string;
  roi: number;
  marketTrend: 'bullish' | 'neutral' | 'bearish';
  keyFeatures: string[];
  aiInsights: string[];
}

export interface AIContentSuggestion {
  id: string;
  type: 'post' | 'article' | 'video' | 'infographic';
  title: string;
  description: string;
  estimatedTime: string;
  keywords: string[];
}

export interface ExploreCategory {
  id: string;
  name: string;
  icon: string;
  description: string;
  count?: number;
}
