
import { significanceAlgorithm } from './significanceAlgorithm';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  type: 'news' | 'entertainment' | 'product' | 'social' | 'analytics' | 'automation' | 'workspace' | 'ads';
  significance: number;
  url: string;
  metadata: any;
  relevanceScore: number;
  crossDomainConnections?: string[];
}

interface CrossDomainInsight {
  sourceType: string;
  targetType: string;
  connection: string;
  significance: number;
  actionable: boolean;
}

class GlobalSearchService {
  private searchIndex: Map<string, SearchResult[]> = new Map();
  private crossDomainMap: Map<string, CrossDomainInsight[]> = new Map();
  
  constructor() {
    this.initializeSearchIndex();
    this.initializeCrossDomainIntelligence();
  }

  private initializeSearchIndex() {
    const sampleData: SearchResult[] = [
      {
        id: '1',
        title: 'AI Breakthrough in Quantum Computing',
        description: 'Revolutionary quantum-AI hybrid achieves 1000x speedup in complex problem solving, enabling real-time financial modeling and climate prediction',
        type: 'news',
        significance: 9.4,
        url: '/search',
        metadata: { category: 'technology', trending: true, impact: 'global' },
        relevanceScore: 0,
        crossDomainConnections: ['entertainment', 'automation', 'workspace']
      },
      {
        id: '2', 
        title: 'Neural Symphony #47 - Adaptive Soundscapes',
        description: 'AI-composed ambient music that adapts to your productivity cycles and environment, boosting focus by 340% in workplace studies',
        type: 'entertainment',
        significance: 8.7,
        url: '/entertainment',
        metadata: { category: 'music', duration: '30 min', productivity: true },
        relevanceScore: 0,
        crossDomainConnections: ['workspace', 'analytics']
      },
      {
        id: '3',
        title: 'Quantum Task Orchestrator',
        description: 'AI-powered workflow automation that predicts task dependencies and optimizes team productivity across multiple time zones',
        type: 'automation',
        significance: 9.1,
        url: '/tasks',
        metadata: { category: 'productivity', aiPowered: true, teamSize: 'unlimited' },
        relevanceScore: 0,
        crossDomainConnections: ['workspace', 'analytics', 'social']
      },
      {
        id: '4',
        title: 'Sustainable Neural Interface Devices',
        description: 'Eco-friendly brain-computer interfaces with 99.7% accuracy and solar charging, revolutionizing accessibility technology',
        type: 'product',
        significance: 8.9,
        url: '/ads',
        metadata: { category: 'sustainability', trending: true, accessibility: true },
        relevanceScore: 0,
        crossDomainConnections: ['news', 'workspace']
      },
      {
        id: '5',
        title: 'Real-time Collaborative Intelligence Hub',
        description: 'Multi-domain workspace where teams coordinate across social, creative, and analytical tasks with AI mediation',
        type: 'workspace',
        significance: 8.8,
        url: '/workspace',
        metadata: { category: 'collaboration', realTime: true, aiMediated: true },
        relevanceScore: 0,
        crossDomainConnections: ['social', 'analytics', 'automation']
      },
      {
        id: '6',
        title: 'Predictive Social Intelligence Networks',
        description: 'AI analyzes social patterns to predict viral content, optimal posting times, and community engagement strategies',
        type: 'social',
        significance: 8.3,
        url: '/social',
        metadata: { category: 'networking', predictive: true, engagement: 'high' },
        relevanceScore: 0,
        crossDomainConnections: ['analytics', 'ads', 'entertainment']
      },
      {
        id: '7',
        title: 'Cross-Platform Performance Analytics',
        description: 'Unified analytics dashboard tracking productivity, creativity, social engagement, and task completion across all domains',
        type: 'analytics',
        significance: 8.6,
        url: '/analytics',
        metadata: { category: 'insights', unified: true, realTime: true },
        relevanceScore: 0,
        crossDomainConnections: ['workspace', 'social', 'automation']
      },
      {
        id: '8',
        title: 'Significance-Driven Ad Intelligence',
        description: 'Advertisement creation using 7-factor significance algorithm to ensure only high-impact, meaningful content reaches audiences',
        type: 'ads',
        significance: 8.4,
        url: '/ads',
        metadata: { category: 'marketing', significanceBased: true, ethical: true },
        relevanceScore: 0,
        crossDomainConnections: ['analytics', 'social', 'news']
      }
    ];

    // Index content by keywords and cross-domain connections
    sampleData.forEach(item => {
      const keywords = this.extractKeywords(item.title + ' ' + item.description);
      keywords.forEach(keyword => {
        if (!this.searchIndex.has(keyword)) {
          this.searchIndex.set(keyword, []);
        }
        this.searchIndex.get(keyword)!.push(item);
      });
    });
  }

  private initializeCrossDomainIntelligence() {
    const insights: CrossDomainInsight[] = [
      {
        sourceType: 'news',
        targetType: 'entertainment',
        connection: 'AI trends influencing personalized content creation and adaptive media experiences',
        significance: 8.7,
        actionable: true
      },
      {
        sourceType: 'analytics',
        targetType: 'automation',
        connection: 'Performance data driving intelligent workflow optimization and predictive task scheduling',
        significance: 9.0,
        actionable: true
      },
      {
        sourceType: 'social',
        targetType: 'ads',
        connection: 'Social engagement patterns informing ethical, significance-based advertising strategies',
        significance: 8.4,
        actionable: true
      },
      {
        sourceType: 'workspace',
        targetType: 'entertainment',
        connection: 'Productivity cycles integrated with adaptive content to maintain optimal focus and creativity',
        significance: 8.2,
        actionable: true
      },
      {
        sourceType: 'automation',
        targetType: 'social',
        connection: 'Automated community management enhancing authentic human connections and engagement',
        significance: 7.9,
        actionable: true
      }
    ];

    insights.forEach(insight => {
      const key = `${insight.sourceType}-${insight.targetType}`;
      if (!this.crossDomainMap.has(key)) {
        this.crossDomainMap.set(key, []);
      }
      this.crossDomainMap.get(key)!.push(insight);
    });
  }

  private extractKeywords(text: string): string[] {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 2);
  }

  async search(query: string, limit: number = 10): Promise<SearchResult[]> {
    if (!query.trim()) return [];

    const queryWords = this.extractKeywords(query);
    const resultMap = new Map<string, SearchResult>();
    
    // Find matching content with cross-domain relevance boost
    queryWords.forEach(word => {
      const matches = this.searchIndex.get(word) || [];
      matches.forEach(match => {
        if (resultMap.has(match.id)) {
          resultMap.get(match.id)!.relevanceScore += 1;
        } else {
          resultMap.set(match.id, { ...match, relevanceScore: 1 });
        }
      });
    });

    // Boost scores for cross-domain connections
    Array.from(resultMap.values()).forEach(result => {
      if (result.crossDomainConnections && result.crossDomainConnections.length > 0) {
        result.relevanceScore += result.crossDomainConnections.length * 0.5;
      }
    });

    // Sort by combined relevance, significance, and cross-domain intelligence
    return Array.from(resultMap.values())
      .sort((a, b) => {
        const scoreA = a.relevanceScore * 0.4 + a.significance * 0.4 + (a.crossDomainConnections?.length || 0) * 0.2;
        const scoreB = b.relevanceScore * 0.4 + b.significance * 0.4 + (b.crossDomainConnections?.length || 0) * 0.2;
        return scoreB - scoreA;
      })
      .slice(0, limit);
  }

  async getInsights(): Promise<any> {
    return {
      totalContent: 15847,
      highSignificance: 432,
      crossDomainConnections: Array.from(this.crossDomainMap.values()).flat().length,
      trendingTopics: [
        'AI-powered cross-domain intelligence',
        'sustainable technology with high significance',
        'quantum computing breakthrough applications',
        'personalized productivity and entertainment fusion',
        'ethical significance-based advertising',
        'real-time collaborative intelligence platforms'
      ],
      crossDomainInsights: Array.from(this.crossDomainMap.values()).flat().slice(0, 5),
      systemHealth: {
        significanceAccuracy: 98.7,
        crossDomainSynergy: 94.3,
        userEngagement: 96.8,
        aiLearningRate: 97.2
      }
    };
  }

  getCrossDomainConnections(sourceType: string): CrossDomainInsight[] {
    const connections: CrossDomainInsight[] = [];
    this.crossDomainMap.forEach((insights, key) => {
      if (key.startsWith(sourceType + '-')) {
        connections.push(...insights);
      }
    });
    return connections.sort((a, b) => b.significance - a.significance);
  }
}

export const globalSearch = new GlobalSearchService();
