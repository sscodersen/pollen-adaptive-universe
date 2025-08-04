import React, { useState, useRef, useEffect } from 'react';
import { Search, X, Clock, TrendingUp } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { searchEngine } from '@/services/searchEngine';
import { useEnhancedApp } from '@/contexts/EnhancedAppContext';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  category: string;
  relevance: number;
}

interface SearchProps {
  onResultSelect?: (result: SearchResult) => void;
  placeholder?: string;
  autoFocus?: boolean;
}

export const GlobalSearch: React.FC<SearchProps> = ({
  onResultSelect,
  placeholder = "Search across all content...",
  autoFocus = false
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  
  const { dispatch } = useEnhancedApp();
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (resultsRef.current && !resultsRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      setShowResults(false);
      return;
    }

    setIsSearching(true);
    try {
      const searchResults = await searchEngine.search(searchQuery, { limit: 10 });
      setResults(searchResults);
      setShowResults(true);
      
      // Track search
      dispatch({ type: 'ADD_SEARCH', payload: searchQuery });
    } catch (error) {
      console.error('Search failed:', error);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSuggestionSearch = async (partial: string) => {
    if (partial.length > 1) {
      try {
        const searchSuggestions = await searchEngine.getSearchSuggestions(partial);
        setSuggestions(searchSuggestions);
      } catch (error) {
        console.error('Failed to get suggestions:', error);
      }
    } else {
      setSuggestions([]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    setSelectedIndex(-1);
    
    if (value.trim()) {
      handleSuggestionSearch(value);
      // Debounce search
      const searchTimer = setTimeout(() => handleSearch(value), 300);
      return () => clearTimeout(searchTimer);
    } else {
      setResults([]);
      setSuggestions([]);
      setShowResults(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showResults) return;

    const totalItems = results.length + suggestions.length;
    
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % totalItems);
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => prev <= 0 ? totalItems - 1 : prev - 1);
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0) {
          if (selectedIndex < suggestions.length) {
            const suggestion = suggestions[selectedIndex];
            setQuery(suggestion.text);
            handleSearch(suggestion.text);
          } else {
            const result = results[selectedIndex - suggestions.length];
            handleResultSelect(result);
          }
        } else if (query.trim()) {
          handleSearch(query);
        }
        break;
      case 'Escape':
        setShowResults(false);
        setSelectedIndex(-1);
        inputRef.current?.blur();
        break;
    }
  };

  const handleResultSelect = (result: SearchResult) => {
    dispatch({ type: 'ADD_ACTIVITY', payload: { type: 'search_result_clicked', resultId: result.id, title: result.title } });
    onResultSelect?.(result);
    setShowResults(false);
    setQuery('');
  };

  const clearSearch = () => {
    setQuery('');
    setResults([]);
    setSuggestions([]);
    setShowResults(false);
    inputRef.current?.focus();
  };

  return (
    <div className="relative w-full max-w-2xl" ref={resultsRef}>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          ref={inputRef}
          value={query}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowResults(query.trim().length > 0)}
          placeholder={placeholder}
          className="pl-10 pr-10 bg-surface-secondary border-border focus:border-primary"
        />
        {query && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearSearch}
            className="absolute right-1 top-1/2 transform -translate-y-1/2 w-8 h-8 p-0"
          >
            <X className="w-4 h-4" />
          </Button>
        )}
      </div>

      {showResults && (suggestions.length > 0 || results.length > 0) && (
        <Card className="absolute top-full left-0 right-0 mt-2 bg-surface-secondary border-border shadow-lg z-50 max-h-96 overflow-hidden">
          <CardContent className="p-0">
            {/* Suggestions */}
            {suggestions.length > 0 && (
              <div className="border-b border-border">
                <div className="px-3 py-2 text-xs font-medium text-muted-foreground bg-surface-tertiary">
                  Suggestions
                </div>
                {suggestions.map((suggestion, index) => (
                  <button
                    key={`suggestion-${index}`}
                    onClick={() => {
                      setQuery(suggestion.text);
                      handleSearch(suggestion.text);
                    }}
                    className={`w-full px-3 py-2 text-left hover:bg-surface-tertiary transition-colors flex items-center gap-2 ${
                      selectedIndex === index ? 'bg-surface-tertiary' : ''
                    }`}
                  >
                    {suggestion.category === 'recent' ? (
                      <Clock className="w-4 h-4 text-muted-foreground" />
                    ) : (
                      <TrendingUp className="w-4 h-4 text-muted-foreground" />
                    )}
                    <span className="text-sm">{suggestion.text}</span>
                    {suggestion.category !== 'recent' && (
                      <Badge variant="outline" className="ml-auto text-xs">
                        {suggestion.category}
                      </Badge>
                    )}
                  </button>
                ))}
              </div>
            )}

            {/* Results */}
            {results.length > 0 && (
              <div className="max-h-64 overflow-y-auto">
                <div className="px-3 py-2 text-xs font-medium text-muted-foreground bg-surface-tertiary">
                  Results
                </div>
                {results.map((result, index) => (
                  <button
                    key={result.id}
                    onClick={() => handleResultSelect(result)}
                    className={`w-full px-3 py-3 text-left hover:bg-surface-tertiary transition-colors ${
                      selectedIndex === suggestions.length + index ? 'bg-surface-tertiary' : ''
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-1 min-w-0">
                        <h3 className="text-sm font-medium text-foreground truncate">
                          {result.title}
                        </h3>
                        <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
                          {result.description}
                        </p>
                      </div>
                      <Badge variant="outline" className="shrink-0 text-xs">
                        {result.category}
                      </Badge>
                    </div>
                  </button>
                ))}
              </div>
            )}

            {isSearching && (
              <div className="px-3 py-4 text-center">
                <div className="text-sm text-muted-foreground">Searching...</div>
              </div>
            )}

            {!isSearching && query.trim() && results.length === 0 && suggestions.length === 0 && (
              <div className="px-3 py-4 text-center">
                <div className="text-sm text-muted-foreground">No results found</div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};