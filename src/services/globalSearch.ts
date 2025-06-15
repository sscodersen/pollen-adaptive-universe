
import { significanceAlgorithm } from './significanceAlgorithm';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  type: 'news' | 'entertainment' | 'product' | 'social' | 'analytics' | 'automation';
  significance: number;
  url: string;
  metadata: any;
  relevanceScore: number;
}

class GlobalSearchService {
  private searchIndex: Map<string, SearchResult[]> = new Map();
  
  constructor() {
    this.initializeSearchIndex();
  }

  private initializeSearchIndex() {
    // Sample high-significance content across domains
    const sampleData: SearchResult[] = [
      {
        id: '1',
        title: 'AI Breakthrough in Quantum Computing',
        description: 'Revolutionary quantum-AI hybrid achieves 1000x speedup in complex problem solving',
        type: 'news',
        significance: 9.4,
        url: '/search',
        metadata: { category: 'technology', trending: true },
        relevanceScore: 0
      },
      {
        id: '2', 
        title: 'Neural Symphony #47 - Digital Dreams',
        description: 'AI-composed ambient music that adapts to your current mood and environment',
        type: 'entertainment',
        significance: 8.7,
        url: '/entertainment',
        metadata: { category: 'music', duration: '30 min' },
        relevanceScore: 0
      },
      {
        id: '3',
        title: 'Smart Productivity Assistant',
        description: 'AI-powered task automation that learns your workflow patterns',
        type: 'automation',
        significance: 8.9,
        url: '/tasks',
        metadata: { category: 'productivity', aiPowered: true },
        relevanceScore: 0
      },
      {
        id: '4',
        title: 'Sustainable Tech Products',
        description: 'Eco-friendly gadgets with high performance and minimal environmental impact',
        type: 'product',
        significance: 8.2,
        url: '/ads',
        metadata: { category: 'sustainability', trending: true },
        relevanceScore: 0
      }
    ];

    // Index content by keywords
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
    
    // Find matching content
    queryWords.forEach(word => {
      const matches = this.searchIndex.get(word) || [];
      matches.forEach(match => {
        if (resultMap.has(match.id)) {
          // Increase relevance for multiple keyword matches
          resultMap.get(match.id)!.relevanceScore += 1;
        } else {
          resultMap.set(match.id, { ...match, relevanceScore: 1 });
        }
      });
    });

    // Sort by relevance and significance
    return Array.from(resultMap.values())
      .sort((a, b) => {
        const scoreA = a.relevanceScore * 0.6 + a.significance * 0.4;
        const scoreB = b.relevanceScore * 0.6 + b.significance * 0.4;
        return scoreB - scoreA;
      })
      .slice(0, limit);
  }

  async getInsights(): Promise<any> {
    return {
      totalContent: 12847,
      highSignificance: 347,
      trendingTopics: [
        'AI breakthrough developments',
        'sustainable technology',
        'quantum computing advances',
        'personalized entertainment',
        'smart automation'
      ],
      crossDomainConnections: [
        {
          domain1: 'news',
          domain2: 'entertainment', 
          connection: 'AI-generated content trending in both domains',
          significance: 8.5
        }
      ]
    };
  }
}

export const globalSearch = new GlobalSearchService();
