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
  const [continuousGeneration, setContinuousGeneration] = useState(true);

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

  // Continuous content generation from multiple platform sources
  const generateContinuousContent = useCallback(async () => {
    if (!continuousGeneration) return;
    
    try {
      // Generate content from different platform events and sources
      const contentSources = [
        { type: 'social', query: 'trending discussions and insights', count: 3 },
        { type: 'news', query: 'latest technology and innovation news', count: 2 },
        { type: 'entertainment', query: 'entertainment and creative content', count: 2 },
        { type: 'shop', query: 'featured products and innovations', count: 2 },
        { type: 'general', query: 'explore diverse topics and discoveries', count: 3 }
      ];
      
      const newContent: SocialContent[] = [];
      
      for (const source of contentSources) {
        const { content } = await contentOrchestrator.generateContent({
          type: source.type as any,
          count: source.count,
          query: source.query,
          strategy: {
            diversity: 0.95,
            freshness: 1.0,
            personalization: 0.7,
            qualityThreshold: 7.0,
            trendingBoost: 1.2
          },
          realtime: true
        });
        
        // Convert content to social format with platform event context
        const socialContent = content.map((item: any, idx: number) => ({
          id: `${source.type}-${Date.now()}-${idx}`,
          type: 'social' as const,
          title: item.title || `${source.type.charAt(0).toUpperCase() + source.type.slice(1)} Update`,
          description: item.description || item.content || 'Platform generated content',
          content: item.content || item.description,
          category: source.type,
          user: {
            name: `${source.type.charAt(0).toUpperCase() + source.type.slice(1)}Bot`,
            username: `${source.type}bot`,
            avatar: `bg-gradient-to-r ${getSourceColor(source.type)}`,
            verified: true,
            rank: 95 + Math.floor(Math.random() * 5),
            badges: ['AI Generated', 'Live Feed', source.type.charAt(0).toUpperCase() + source.type.slice(1)]
          },
          timestamp: new Date().toLocaleString(),
          views: Math.floor(Math.random() * 3000) + 500,
          engagement: Math.floor(Math.random() * 800) + 200,
          significance: item.significance || (8.0 + Math.random() * 2),
          quality: Math.floor((item.significance || 8.0) * 10),
          trending: item.significance > 8.5,
          impact: item.significance > 9.0 ? 'high' : item.significance > 7.5 ? 'medium' : 'low',
          contentType: source.type === 'news' ? 'news' : source.type === 'entertainment' ? 'entertainment' : 'social',
          tags: item.tags || [source.type, 'trending', 'live'],
          readTime: '1-3 min'
        }));
        
        newContent.push(...socialContent as SocialContent[]);
      }
      
      // Add new content to existing posts (prepend for chronological order)
      setPosts(prevPosts => [...newContent, ...prevPosts.slice(0, 50)]); // Keep max 50 posts
      
    } catch (error) {
      console.error('Continuous content generation failed:', error);
    }
  }, [continuousGeneration]);

  // Helper function to get source-specific colors
  const getSourceColor = (sourceType: string) => {
    const colors = {
      social: 'from-blue-500 to-cyan-500',
      news: 'from-red-500 to-orange-500',
      entertainment: 'from-purple-500 to-pink-500',
      shop: 'from-green-500 to-teal-500',
      general: 'from-gray-500 to-slate-500'
    };
    return colors[sourceType as keyof typeof colors] || colors.general;
  };

  // Load initial posts from content orchestrator
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
    
    // Setup continuous content generation from platform events
    const continuousTimer = setTimeout(() => {
      generateContinuousContent();
      // Set up recurring generation every 45 seconds
      const recurringTimer = setInterval(generateContinuousContent, 45000);
      return () => clearInterval(recurringTimer);
    }, 8000); // Start after 8 seconds

    // Reduced frequency for regular reload to improve performance
    const interval = setInterval(loadPosts, 120000); // 2 minutes instead of 45 seconds
    
    return () => {
      unsubscribe();
      clearInterval(interval);
      clearTimeout(continuousTimer);
    };
  }, [loadPosts]);

  return {
    posts: filteredPosts,
    loading,
    refetch: loadPosts,
    postsCount: filteredPosts.length
  };
};