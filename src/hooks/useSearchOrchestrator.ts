import { useState, useCallback, useRef, useEffect } from 'react';
import { searchOrchestrator } from '@/services/searchOrchestrator';
import type { SearchResult } from '@/types/search';

interface UseSearchOrchestratorReturn {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  results: SearchResult[];
  isLoading: boolean;
  error: string | null;
  executeSearch: (query: string) => Promise<void>;
  clearResults: () => void;
}

export const useSearchOrchestrator = (sessionId?: string): UseSearchOrchestratorReturn => {
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceTimeout = useRef<NodeJS.Timeout | null>(null);
  const abortController = useRef<AbortController | null>(null);

  const executeSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    if (abortController.current) {
      abortController.current.abort();
    }

    abortController.current = new AbortController();

    setIsLoading(true);
    setError(null);

    try {
      const response = await searchOrchestrator.search(query, sessionId);
      setResults(response.results);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Search failed';
      setError(errorMessage);
      console.error('Search error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const debouncedSearch = useCallback((query: string) => {
    if (debounceTimeout.current) {
      clearTimeout(debounceTimeout.current);
    }

    debounceTimeout.current = setTimeout(() => {
      if (query.trim()) {
        executeSearch(query);
      }
    }, 500);
  }, [executeSearch]);

  const clearResults = useCallback(() => {
    setResults([]);
    setSearchQuery('');
    setError(null);
  }, []);

  useEffect(() => {
    return () => {
      if (debounceTimeout.current) {
        clearTimeout(debounceTimeout.current);
      }
      if (abortController.current) {
        abortController.current.abort();
      }
    };
  }, []);

  const handleSearchChange = useCallback((query: string) => {
    setSearchQuery(query);
    debouncedSearch(query);
  }, [debouncedSearch]);

  return {
    searchQuery,
    setSearchQuery: handleSearchChange,
    results,
    isLoading,
    error,
    executeSearch,
    clearResults
  };
};
