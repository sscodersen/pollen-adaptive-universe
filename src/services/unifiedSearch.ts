import { storageService } from './storageService';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  category: 'music' | 'apps' | 'games' | 'entertainment' | 'shop' | 'content';
  type: string;
  relevance: number;
  url?: string;
  metadata?: any;
}

export class UnifiedSearch {
  private static instance: UnifiedSearch;
  private searchIndex: Map<string, SearchResult[]> = new Map();

  static getInstance(): UnifiedSearch {
    if (!UnifiedSearch.instance) {
      UnifiedSearch.instance = new UnifiedSearch();
    }
    return UnifiedSearch.instance;
  }

  async search(query: string, category?: string): Promise<SearchResult[]> {
    storageService.trackEvent('search', { query, category });
    
    const normalizedQuery = query.toLowerCase().trim();
    if (!normalizedQuery) return [];

    // Get cached results first
    const cacheKey = `search_${normalizedQuery}_${category || 'all'}`;
    const cached = await storageService.getCache<SearchResult[]>(cacheKey);
    if (cached) return cached;

    // Perform search across all categories
    const results = await this.performSearch(normalizedQuery, category);
    
    // Cache results
    await storageService.setCache(cacheKey, results, 300000); // 5 minutes
    
    return results;
  }

  private async performSearch(query: string, category?: string): Promise<SearchResult[]> {
    const allResults: SearchResult[] = [];

    // Mock search data - in real implementation, this would search actual content
    const mockData = this.getMockSearchData();
    
    for (const item of mockData) {
      if (category && item.category !== category) continue;
      
      const relevance = this.calculateRelevance(query, item);
      if (relevance > 0) {
        allResults.push({ ...item, relevance });
      }
    }

    // Sort by relevance
    allResults.sort((a, b) => b.relevance - a.relevance);
    
    return allResults.slice(0, 50); // Limit results
  }

  private calculateRelevance(query: string, item: SearchResult): number {
    const queryWords = query.split(' ');
    let score = 0;

    for (const word of queryWords) {
      if (item.title.toLowerCase().includes(word)) score += 10;
      if (item.description.toLowerCase().includes(word)) score += 5;
      if (item.category.toLowerCase().includes(word)) score += 3;
      if (item.type.toLowerCase().includes(word)) score += 2;
    }

    return score;
  }

  private getMockSearchData(): SearchResult[] {
    return [
      // Music
      { id: '1', title: 'Digital Dreams', description: 'AI-generated ambient electronic track', category: 'music', type: 'track', relevance: 0 },
      { id: '2', title: 'Neural Symphony', description: 'Classical AI composition', category: 'music', type: 'album', relevance: 0 },
      
      // Apps
      { id: '3', title: 'TaskMaster Pro', description: 'Advanced productivity and task management', category: 'apps', type: 'productivity', relevance: 0 },
      { id: '4', title: 'PhotoEdit AI', description: 'AI-powered image editing suite', category: 'apps', type: 'creativity', relevance: 0 },
      
      // Games
      { id: '5', title: 'Space Explorer VR', description: 'Virtual reality space exploration game', category: 'games', type: 'vr', relevance: 0 },
      { id: '6', title: 'Puzzle Master', description: 'Brain-training puzzle collection', category: 'games', type: 'puzzle', relevance: 0 },
      
      // Entertainment
      { id: '7', title: 'AI Documentary Series', description: 'Educational content about artificial intelligence', category: 'entertainment', type: 'documentary', relevance: 0 },
      { id: '8', title: 'Tech Podcast Daily', description: 'Latest technology news and discussions', category: 'entertainment', type: 'podcast', relevance: 0 },
      
      // Shop
      { id: '9', title: 'Smart Home Hub', description: 'Central control for all smart devices', category: 'shop', type: 'electronics', relevance: 0 },
      { id: '10', title: 'AI Learning Course', description: 'Complete machine learning curriculum', category: 'shop', type: 'education', relevance: 0 }
    ];
  }
}

export const unifiedSearch = UnifiedSearch.getInstance();