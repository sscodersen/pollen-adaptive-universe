import { useState, useEffect, useCallback, useMemo } from 'react';
import { contentOrchestrator } from '../../services/contentOrchestrator';
import { SocialContent } from '../../services/unifiedContentEngine';
import { enhancedTrendEngine, GeneratedPost } from '../../services/enhancedTrendEngine';
import { cleanText, truncateText } from '@/lib/textUtils';
import { isBlacklistedText } from '@/lib/blacklist';

interface UseSocialFeedDataProps {
  activities?: SocialContent[];
  filter?: string;
  searchQuery?: string;
}

export const useSocialFeedData = ({ 
  activities, 
  filter = "all", 
  searchQuery = "" 
}: UseSocialFeedDataProps) => {
  const [posts, setPosts] = useState<SocialContent[]>(activities || []);
  const [trendPosts, setTrendPosts] = useState<GeneratedPost[]>([]);
  const [loading, setLoading] = useState(true);

  // Hash function for deterministic content generation
  const hashString = useCallback((s: string) => {
    let h = 0; 
    for (let i = 0; i < s.length; i++) { 
      h = ((h << 5) - h) + s.charCodeAt(i); 
      h |= 0; 
    }
    return Math.abs(h);
  }, []);

  // Generate original summary for trend posts
  const makeOriginalSummary = useCallback((p: GeneratedPost): string => {
    const impact = p.engagement_score > 80 ? 'high' : p.engagement_score > 60 ? 'medium' : 'rising';
    const stripped = (p.content || '').replace(/^\([^)]*\)\s*/,'');
    const cleaned = cleanText(stripped).replace(/\s*(is\s+trending|is\s+surging)[.!…]*$/i, '');
    const clip = truncateText(cleaned, 140);
    const tags = p.hashtags.slice(0, 2).join(' ');
    const options = [
      `${p.topic}: ${clip}`.trim(),
      `Signal check • ${p.topic}. ${tags}`.trim(),
      `${p.topic} — ${clip}`.trim(),
    ];
    return options[hashString(p.id) % options.length];
  }, [hashString]);

  // Convert trend posts to social content format - memoized
  const convertTrendPostsToSocialContent = useCallback((trendPosts: GeneratedPost[]): SocialContent[] => {
    return trendPosts.map(post => ({
      id: post.id,
      type: 'social' as const,
      title: post.topic,
      description: makeOriginalSummary(post),
      content: makeOriginalSummary(post),
      category: 'trending',
      user: {
        name: 'TrendBot',
        username: 'trendbot',
        avatar: 'bg-gradient-to-r from-purple-500 to-pink-500',
        verified: true,
        rank: 98,
        badges: ['AI Generated', 'Trending']
      },
      timestamp: new Date(post.timestamp).toLocaleString(),
      views: Math.floor(Math.random() * 5000) + 1000,
      engagement: Math.floor(Math.random() * 1000) + 500,
      significance: post.engagement_score / 10,
      quality: Math.floor(post.engagement_score),
      trending: true,
      impact: post.engagement_score > 80 ? 'high' : post.engagement_score > 60 ? 'medium' : 'low',
      contentType: post.type === 'social' ? 'social' : post.type === 'news' ? 'news' : 'discussion',
      tags: post.hashtags.map(h => h.replace('#', '')),
      readTime: '2 min'
    }));
  }, [makeOriginalSummary]);

  // Load posts from content orchestrator
  const loadPosts = useCallback(async () => {
    if (activities && activities.length > 0) {
      setPosts(activities);
      setLoading(false);
      return;
    }
    
    setLoading(true);
    try {
      const strategy = {
        diversity: 0.9,
        freshness: 0.8,
        personalization: 0.6,
        qualityThreshold: 6.0,
        trendingBoost: 1.5
      };
      
      const { content: newPosts } = await contentOrchestrator.generateContent({
        type: 'social',
        count: 15,
        strategy
      });
      
      const trendGeneratedPosts = enhancedTrendEngine.getGeneratedPosts().slice(0, 10);
      setPosts(newPosts as SocialContent[]);
      setTrendPosts(trendGeneratedPosts);
    } catch (error) {
      console.error('Failed to load posts:', error);
    }
    setLoading(false);
  }, [activities]);

  // Combine all posts - memoized
  const allPosts = useMemo(() => {
    return [
      ...convertTrendPostsToSocialContent(trendPosts),
      ...posts
    ];
  }, [convertTrendPostsToSocialContent, trendPosts, posts]);

  // Filter out blacklisted content - memoized
  const nonBlacklistedPosts = useMemo(() => {
    return allPosts.filter(p => 
      !isBlacklistedText(p.title || '') && 
      !isBlacklistedText(p.description || '') && 
      !p.tags?.some(t => isBlacklistedText(t))
    );
  }, [allPosts]);

  // Apply filters and search - memoized
  const filteredPosts = useMemo(() => {
    return nonBlacklistedPosts.filter(post => {
      const matchesFilter = filter === 'all' || 
        (filter === 'trending' && post.trending) || 
        (filter === 'high-impact' && (post.significance || 0) > 8);
      
      const matchesSearch = !searchQuery || 
        (post.description || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
        post.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (post.category || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
        (post.user?.name || '').toLowerCase().includes(searchQuery.toLowerCase());
      
      return matchesFilter && matchesSearch;
    });
  }, [nonBlacklistedPosts, filter, searchQuery]);

  // Setup subscriptions and intervals
  useEffect(() => {
    loadPosts();

    // Prime trend posts immediately
    setTrendPosts(enhancedTrendEngine.getGeneratedPosts().slice(0, 10));

    // Subscribe to trend engine updates
    const unsubscribe = enhancedTrendEngine.subscribe((data) => {
      if (data.type === 'posts_generated' || data.type === 'manual_content_generated') {
        setTrendPosts(enhancedTrendEngine.getGeneratedPosts().slice(0, 10));
      }
    });
    
    // Reduced frequency to improve performance
    const interval = setInterval(loadPosts, 120000); // 2 minutes instead of 45 seconds
    
    return () => {
      clearInterval(interval);
      unsubscribe();
    };
  }, [loadPosts]);

  return {
    posts: filteredPosts,
    loading,
    refetch: loadPosts,
    postsCount: filteredPosts.length
  };
};