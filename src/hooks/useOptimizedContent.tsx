
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { contentCurator } from '../services/contentCurator';

interface ContentItem {
  id: string;
  title: string;
  description: string;
  content: string;
  category: string;
  significance: number;
  timestamp: number;
}

export const useOptimizedContent = (category: string, limit: number = 20) => {
  const [content, setContent] = useState<ContentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const lastUpdateRef = useRef(0);

  const loadContent = useCallback(async () => {
    const now = Date.now();
    // Avoid too frequent updates
    if (now - lastUpdateRef.current < 30000) return;

    setLoading(true);
    try {
      const newContent = await contentCurator.scrapeAndCurateContent(
        category as 'news' | 'shop' | 'entertainment', 
        limit
      );
      
      const mappedContent = newContent.map(item => ({
        id: item.id,
        title: item.title,
        description: item.description,
        content: item.content,
        category: item.category,
        significance: item.significance,
        timestamp: item.timestamp
      }));

      setContent(mappedContent);
      lastUpdateRef.current = now;
    } catch (error) {
      console.error(`Error loading ${category} content:`, error);
    } finally {
      setLoading(false);
    }
  }, [category, limit]);

  const sortedContent = useMemo(() => 
    [...content].sort((a, b) => b.significance - a.significance),
    [content]
  );

  const highSignificanceContent = useMemo(() =>
    sortedContent.filter(item => item.significance > 7.0),
    [sortedContent]
  );

  useEffect(() => {
    loadContent();
    const interval = setInterval(loadContent, 60000); // Every minute
    return () => clearInterval(interval);
  }, [loadContent]);

  return {
    content: sortedContent,
    highSignificanceContent,
    loading,
    refresh: loadContent
  };
};
