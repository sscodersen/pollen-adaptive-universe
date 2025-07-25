import { storageService } from './storageService';
import { unifiedSearch } from './unifiedSearch';

interface SearchOptions {
  category?: string;
  filters?: string[];
  sortBy?: 'relevance' | 'date' | 'popularity';
  limit?: number;
}

interface SearchSuggestion {
  text: string;
  category: string;
  frequency: number;
}

export class SearchEngine {
  private static instance: SearchEngine;
  private searchCache = new Map<string, any>();
  private searchSuggestions: SearchSuggestion[] = [];

  static getInstance(): SearchEngine {
    if (!SearchEngine.instance) {
      SearchEngine.instance = new SearchEngine();
    }
    return SearchEngine.instance;
  }

  async search(query: string, options: SearchOptions = {}) {
    const cacheKey = `${query}-${JSON.stringify(options)}`;
    
    // Check cache first
    if (this.searchCache.has(cacheKey)) {
      return this.searchCache.get(cacheKey);
    }

    // Track search
    await this.trackSearch(query, options.category);

    // Perform search using unified search
    const results = await unifiedSearch.search(query, options.category);

    // Apply additional filters and sorting
    let filteredResults = results;
    
    if (options.filters?.length) {
      filteredResults = results.filter(item => 
        options.filters!.some(filter => 
          item.title.toLowerCase().includes(filter.toLowerCase()) ||
          item.description.toLowerCase().includes(filter.toLowerCase())
        )
      );
    }

    if (options.sortBy === 'date') {
      filteredResults.sort((a, b) => 
        new Date(b.metadata?.date || 0).getTime() - new Date(a.metadata?.date || 0).getTime()
      );
    } else if (options.sortBy === 'popularity') {
      filteredResults.sort((a, b) => (b.metadata?.views || 0) - (a.metadata?.views || 0));
    }

    if (options.limit) {
      filteredResults = filteredResults.slice(0, options.limit);
    }

    // Cache results
    this.searchCache.set(cacheKey, filteredResults);
    
    return filteredResults;
  }

  async getSearchSuggestions(partial: string): Promise<SearchSuggestion[]> {
    const searchHistory = await storageService.getItem('searchHistory') || [];
    
    // Filter suggestions based on partial input
    const matchingSuggestions = this.searchSuggestions
      .filter(suggestion => 
        suggestion.text.toLowerCase().includes(partial.toLowerCase())
      )
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 5);

    // Include recent searches
    const recentMatches = (searchHistory as string[])
      .filter((search: string) => 
        search.toLowerCase().includes(partial.toLowerCase())
      )
      .slice(0, 3)
      .map((search: string) => ({
        text: search,
        category: 'recent',
        frequency: 0
      }));

    return [...matchingSuggestions, ...recentMatches];
  }

  async getPopularSearches(): Promise<string[]> {
    return this.searchSuggestions
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10)
      .map(s => s.text);
  }

  private async trackSearch(query: string, category?: string): Promise<void> {
    // Update search history
    const searchHistory = await storageService.getItem('searchHistory') || [];
    const newHistory = [query, ...(searchHistory as string[]).filter((h: string) => h !== query)].slice(0, 50);
    await storageService.setItem('searchHistory', newHistory);

    // Update search suggestions
    const existingSuggestion = this.searchSuggestions.find(s => s.text === query);
    if (existingSuggestion) {
      existingSuggestion.frequency++;
    } else {
      this.searchSuggestions.push({
        text: query,
        category: category || 'general',
        frequency: 1
      });
    }

    // Analytics
    storageService.trackEvent('search', { query, category });
  }

  clearCache(): void {
    this.searchCache.clear();
  }
}

export const searchEngine = SearchEngine.getInstance();