export type FeedTab = 'all' | 'wellness' | 'agriculture' | 'social-impact' | 'opportunities';

export interface BaseCardProps {
  id: string;
  title: string;
  description: string;
  timestamp?: string;
}

export interface VerificationCardData extends BaseCardProps {
  accuracy: number;
  isSecure: boolean;
}

export interface WellnessCardData extends BaseCardProps {
  duration: string;
  category: string;
  tags: string[];
  impactLevel: number;
}

export interface AgricultureCardData extends BaseCardProps {
  soilPH: number;
  nitrogenLevel: string;
  weatherAlert?: string;
  cropStatus: 'healthy' | 'warning' | 'critical';
}

export interface SocialImpactCardData extends BaseCardProps {
  fundingGoal: number;
  currentFunding: number;
  backers: number;
  qualityScore: number;
  votes: number;
  isVoted: boolean;
}

export interface OpportunityCardData extends BaseCardProps {
  type: string;
  relevanceScore: number;
  urgency: 'low' | 'medium' | 'high';
  aiInsight?: string;
  tags: string[];
}
