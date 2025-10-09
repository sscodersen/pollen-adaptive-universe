import { storageService } from './storageService';
import { SocialContent } from './unifiedContentEngine';
import { WellnessTip } from './wellnessContent';
import { SocialInitiative } from './socialImpact';
import { Opportunity } from './opportunityCuration';

export class DemoFeedAdapter {
  static getFeedItems(): SocialContent[] {
    const demoFeedItems = storageService.getItem('demo_feed_items', []);
    
    return demoFeedItems.map((item: any) => ({
      id: item.id,
      type: 'social' as const,
      title: item.title,
      description: item.content,
      content: item.content,
      category: item.category || 'general',
      user: {
        name: item.type === 'wellness_tip' ? 'Wellness Bot' : item.type === 'social_impact' ? 'Community Impact' : 'Opportunity Bot',
        username: item.type === 'wellness_tip' ? 'wellness' : item.type === 'social_impact' ? 'impact' : 'opportunities',
        avatar: item.type === 'wellness_tip' ? 'bg-gradient-to-r from-green-400 to-blue-500' : item.type === 'social_impact' ? 'bg-gradient-to-r from-purple-400 to-pink-500' : 'bg-gradient-to-r from-orange-400 to-yellow-500',
        verified: item.verified || true,
        rank: Math.floor(Math.random() * 30) + 70,
        badges: ['AI Generated', item.type === 'opportunity' && item.verified ? 'Verified' : 'Curated'].filter(Boolean)
      },
      timestamp: new Date(Date.now() - Math.floor(Math.random() * 7200000)).toLocaleString(),
      views: item.views || Math.floor(Math.random() * 10000) + 1000,
      engagement: item.likes || Math.floor(Math.random() * 1000) + 100,
      significance: (item.aiScore || 8) / 10,
      quality: Math.floor(item.aiScore || 80),
      trending: item.aiScore > 9 || false,
      impact: item.aiScore > 9 ? 'high' : item.aiScore > 8 ? 'medium' : 'low',
      contentType: item.type === 'wellness_tip' ? 'discussion' : item.type === 'social_impact' ? 'social' : 'news',
      tags: [item.category, item.type].filter(Boolean),
      readTime: '2 min'
    }));
  }

  static getWellnessTips(): WellnessTip[] {
    const demoFeedItems = storageService.getItem('demo_feed_items', []);
    const wellnessItems = demoFeedItems.filter((item: any) => item.type === 'wellness_tip');
    
    return wellnessItems.map((item: any) => ({
      id: item.id,
      title: item.title,
      content: item.content,
      category: item.category.includes('mental') ? 'mental' : item.category.includes('fitness') ? 'physical' : item.category.includes('sleep') ? 'sleep' : 'nutrition',
      duration: '2-5 min',
      difficulty: 'easy' as const,
      impact: Math.floor(item.aiScore || 8),
      tags: [item.category]
    }));
  }

  static getSocialInitiatives(): SocialInitiative[] {
    const demoFeedItems = storageService.getItem('demo_feed_items', []);
    const socialItems = demoFeedItems.filter((item: any) => item.type === 'social_impact');
    
    return socialItems.map((item: any, index: number) => {
      let category: 'education' | 'healthcare' | 'environment' | 'poverty' | 'technology' | 'community' = 'community';
      
      if (item.category === 'environment') category = 'environment';
      else if (item.category === 'health') category = 'healthcare';
      else if (item.category === 'education') category = 'education';
      else if (item.category === 'volunteer') category = 'poverty';
      
      return {
        id: item.id,
        title: item.title,
        description: item.content,
        category,
        organization: 'Community Initiative',
        fundingGoal: item.fundingProgress ? 50000 : 0,
        currentFunding: item.fundingProgress ? Math.floor(50000 * (item.fundingProgress / 100)) : 0,
        backers: Math.floor(Math.random() * 200) + 50,
        impactScore: item.aiScore || 8.5,
        aiQualityScore: item.aiScore || 8.5,
        votes: Math.floor(Math.random() * 500) + 100,
        featured: item.aiScore > 9 || false,
        timeline: '6 months',
        location: 'Local',
        sdgGoals: [1, 3, 11],
        updates: [
          { date: new Date().toISOString(), content: 'Initiative launched successfully' }
        ]
      };
    });
  }

  static getOpportunities(): Opportunity[] {
    const demoOpportunities = storageService.getItem('demo_opportunities', []);
    
    return demoOpportunities.map((item: any) => ({
      id: item.id,
      type: item.type === 'job' ? 'app' : item.type === 'grant' ? 'investment' : item.type === 'education' ? 'lifestyle' : item.type === 'apprenticeship' ? 'app' : 'news',
      title: item.title,
      description: item.description,
      relevanceScore: item.aiScore || 8.0,
      qualityScore: item.verified ? 9.0 : 7.5,
      urgency: item.aiScore > 9 ? 'high' : item.aiScore > 8 ? 'medium' : 'low',
      category: item.tags?.[0] || 'general',
      tags: item.tags || [],
      source: item.company,
      dateAdded: new Date(Date.now() - Math.floor(Math.random() * 7200000)).toISOString(),
      expiryDate: new Date(Date.now() + Math.floor(Math.random() * 2592000000)).toISOString(),
      aiInsights: [
        `${item.verified ? 'Verified opportunity' : 'Curated opportunity'}`,
        `Located in ${item.location}`,
        `${item.salary}`
      ],
      estimatedValue: item.salary,
      actionRequired: 'Apply now'
    }));
  }
}
