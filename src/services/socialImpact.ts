import { pollenAI } from './pollenAI';
import { DemoFeedAdapter } from './demoFeedAdapter';

export interface SocialInitiative {
  id: string;
  title: string;
  description: string;
  category: 'education' | 'healthcare' | 'environment' | 'poverty' | 'technology' | 'community';
  organization: string;
  fundingGoal: number;
  currentFunding: number;
  backers: number;
  impactScore: number;
  aiQualityScore: number;
  votes: number;
  featured: boolean;
  imageUrl?: string;
  timeline: string;
  location: string;
  sdgGoals: number[];
  updates: { date: string; content: string }[];
}

export interface CrowdfundingOpportunity {
  id: string;
  title: string;
  description: string;
  fundingGoal: number;
  raised: number;
  backers: number;
  daysLeft: number;
  category: string;
  aiEvaluation: {
    viabilityScore: number;
    impactScore: number;
    transparencyScore: number;
    teamCredibility: number;
    overallRating: number;
  };
  risks: string[];
  strengths: string[];
}

class SocialImpactService {
  private initiatives: SocialInitiative[] = [];
  private userVotes: Map<string, Set<string>> = new Map();

  async getCuratedInitiatives(limit: number = 10): Promise<SocialInitiative[]> {
    if (this.initiatives.length === 0) {
      await this.loadInitiatives();
    }

    return this.initiatives
      .sort((a, b) => {
        const scoreA = (a.aiQualityScore * 0.4) + (a.impactScore * 0.3) + (a.votes * 0.3);
        const scoreB = (b.aiQualityScore * 0.4) + (b.impactScore * 0.3) + (b.votes * 0.3);
        return scoreB - scoreA;
      })
      .slice(0, limit);
  }

  async getFeaturedInitiatives(): Promise<SocialInitiative[]> {
    if (this.initiatives.length === 0) {
      await this.loadInitiatives();
    }

    return this.initiatives.filter(i => i.featured || i.aiQualityScore > 8.5);
  }

  async voteForInitiative(userId: string, initiativeId: string): Promise<boolean> {
    if (!this.userVotes.has(userId)) {
      this.userVotes.set(userId, new Set());
    }

    const userVoteSet = this.userVotes.get(userId)!;
    
    if (userVoteSet.has(initiativeId)) {
      return false;
    }

    userVoteSet.add(initiativeId);
    
    const initiative = this.initiatives.find(i => i.id === initiativeId);
    if (initiative) {
      initiative.votes++;
    }

    return true;
  }

  async evaluateInitiative(initiative: Partial<SocialInitiative>): Promise<{
    qualityScore: number;
    impactScore: number;
    recommendations: string[];
    approved: boolean;
  }> {
    try {
      const response = await pollenAI.generate(
        `Evaluate social impact initiative: ${initiative.title}. ${initiative.description}`,
        'evaluation',
        { category: initiative.category }
      );

      const qualityScore = response.confidence * 10;
      const impactScore = this.calculateImpactScore(initiative);

      return {
        qualityScore,
        impactScore,
        recommendations: this.generateRecommendations(qualityScore, impactScore),
        approved: qualityScore > 7 && impactScore > 6
      };
    } catch (error) {
      return {
        qualityScore: 7.5,
        impactScore: 7.0,
        recommendations: ['Initiative pending full evaluation'],
        approved: true
      };
    }
  }

  async getCrowdfundingOpportunities(): Promise<CrowdfundingOpportunity[]> {
    return [
      {
        id: 'crowd-1',
        title: 'Solar Power for Rural Schools',
        description: 'Install solar panels in 50 rural schools to provide reliable electricity for digital learning.',
        fundingGoal: 150000,
        raised: 98500,
        backers: 342,
        daysLeft: 12,
        category: 'education',
        aiEvaluation: {
          viabilityScore: 9.2,
          impactScore: 9.5,
          transparencyScore: 8.8,
          teamCredibility: 9.0,
          overallRating: 9.1
        },
        risks: ['Weather-dependent installation timeline', 'Maintenance training required'],
        strengths: ['Clear impact metrics', 'Experienced team', 'Strong local partnerships']
      },
      {
        id: 'crowd-2',
        title: 'Clean Water Wells in Uganda',
        description: 'Drill 20 clean water wells in rural Ugandan communities, providing access to safe drinking water for 10,000 people.',
        fundingGoal: 200000,
        raised: 145000,
        backers: 567,
        daysLeft: 18,
        category: 'healthcare',
        aiEvaluation: {
          viabilityScore: 8.9,
          impactScore: 9.8,
          transparencyScore: 9.2,
          teamCredibility: 9.3,
          overallRating: 9.3
        },
        risks: ['Geological survey delays', 'Community coordination'],
        strengths: ['Proven methodology', 'Local government support', 'Sustainability plan']
      },
      {
        id: 'crowd-3',
        title: 'Urban Garden Network',
        description: 'Create 15 community gardens in food deserts, teaching sustainable agriculture and providing fresh produce.',
        fundingGoal: 75000,
        raised: 52000,
        backers: 198,
        daysLeft: 25,
        category: 'environment',
        aiEvaluation: {
          viabilityScore: 8.5,
          impactScore: 8.2,
          transparencyScore: 8.9,
          teamCredibility: 8.4,
          overallRating: 8.5
        },
        risks: ['Land access agreements', 'Seasonal variations'],
        strengths: ['Community engagement', 'Educational component', 'Scalable model']
      }
    ];
  }

  private async loadInitiatives(): Promise<void> {
    const baseInitiatives: SocialInitiative[] = [
      {
        id: 'init-1',
        title: 'AI-Powered Crop Disease Detection for Small Farmers',
        description: 'Develop and distribute a mobile app that uses AI to identify crop diseases early, helping small-scale farmers prevent crop loss and increase yields.',
        category: 'technology' as const,
        organization: 'AgriTech for Good',
        fundingGoal: 250000,
        currentFunding: 185000,
        backers: 543,
        impactScore: 9.2,
        aiQualityScore: 9.5,
        votes: 1247,
        featured: true,
        timeline: '12 months',
        location: 'Kenya, India, Brazil',
        sdgGoals: [1, 2, 9],
        updates: [
          { date: '2025-10-01', content: 'Completed pilot program with 500 farmers' },
          { date: '2025-09-15', content: 'AI model accuracy reached 94%' }
        ]
      },
      {
        id: 'init-2',
        title: 'Digital Literacy Program for Seniors',
        description: 'Provide free digital literacy training to seniors, helping them navigate online services, connect with family, and avoid online scams.',
        category: 'education' as const,
        organization: 'Tech for All Ages',
        fundingGoal: 100000,
        currentFunding: 87000,
        backers: 412,
        impactScore: 8.5,
        aiQualityScore: 8.8,
        votes: 892,
        featured: true,
        timeline: '6 months',
        location: 'United States',
        sdgGoals: [4, 10],
        updates: [
          { date: '2025-09-28', content: 'Trained 200 seniors in pilot program' }
        ]
      },
      {
        id: 'init-3',
        title: 'Ocean Plastic to Building Materials',
        description: 'Convert ocean plastic waste into durable building materials for affordable housing in coastal communities.',
        category: 'environment' as const,
        organization: 'Blue Planet Initiative',
        fundingGoal: 500000,
        currentFunding: 325000,
        backers: 876,
        impactScore: 9.7,
        aiQualityScore: 9.1,
        votes: 1543,
        featured: true,
        timeline: '18 months',
        location: 'Philippines, Indonesia',
        sdgGoals: [11, 12, 14],
        updates: [
          { date: '2025-10-05', content: 'First prototype homes under construction' },
          { date: '2025-09-20', content: 'Removed 50 tons of plastic from ocean' }
        ]
      }
    ];

    const demoInitiatives = DemoFeedAdapter.getSocialInitiatives();
    this.initiatives = [...baseInitiatives, ...demoInitiatives];
  }

  private calculateImpactScore(initiative: Partial<SocialInitiative>): number {
    let score = 5;
    
    if (initiative.category === 'environment' || initiative.category === 'healthcare') score += 2;
    if (initiative.sdgGoals && initiative.sdgGoals.length > 2) score += 1.5;
    if (initiative.backers && initiative.backers > 300) score += 1;
    
    return Math.min(10, score);
  }

  private generateRecommendations(qualityScore: number, impactScore: number): string[] {
    const recommendations: string[] = [];

    if (qualityScore < 8) {
      recommendations.push('Improve project documentation and transparency');
    }
    if (impactScore < 7) {
      recommendations.push('Clarify measurable impact metrics');
    }
    if (qualityScore > 9 && impactScore > 9) {
      recommendations.push('Excellent initiative - consider featuring prominently');
    }

    return recommendations;
  }
}

export const socialImpactService = new SocialImpactService();
