
import { SignificanceFactors } from '../services/insightFlow';

export interface Product {
  id: string;
  name: string;
  description: string;
  price: string;
  originalPrice?: string;
  discount?: number;
  rating: number;
  reviews: number;
  category: string;
  brand: string;
  tags: string[];
  link: string;
  inStock: boolean;
  trending: boolean;
  significance: number;
  features: string[];
  seller: string;
  views: number;
  rank: number;
  quality: number;
  impact: 'low' | 'medium' | 'high' | 'premium';
  // InsightFlow algorithm properties
  insightFlowScore?: number;
  insightFlowReasoning?: string;
  insightFlowFactors?: SignificanceFactors;
}
