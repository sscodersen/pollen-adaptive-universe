import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { SocialContent } from '../../services/unifiedContentEngine';
import { SocialFeedHeader } from './SocialFeedHeader';
import { PostCard } from './PostCard';
import { useSocialFeedData } from './useSocialFeedData';
import { PremiumBannerAd, NativeFeedAd, InlineAd } from '@/components/ads/AdComponents';
import { Loader2 } from 'lucide-react';

export interface SocialFeedProps {
  activities?: SocialContent[];
  isGenerating?: boolean;
  filter?: string;
}

export const SocialFeed = ({ activities, isGenerating = false, filter = "all" }: SocialFeedProps) => {
  const [searchQuery, setSearchQuery] = useState('');
  const loadMoreRef = useRef<HTMLDivElement>(null);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  
  // Use custom hook for data management - optimized with memoization
  const { posts, loading, refetch, postsCount } = useSocialFeedData({
    activities,
    filter,
    searchQuery
  });

  // Memoized search handler to prevent unnecessary re-renders
  const handleSearchChange = useCallback((query: string) => {
    setSearchQuery(query);
  }, []);

  // Memoized post click handler
  const handlePostClick = useCallback((post: SocialContent) => {
    console.log('Post clicked:', post.id);
    // Add analytics or navigation logic here
  }, []);

  // Handle vote changes
  const handleVoteChange = useCallback((postId: string, votes: any) => {
    console.log('Vote updated for post:', postId, votes);
  }, []);

  // Infinite scroll implementation
  useEffect(() => {
    if (!loadMoreRef.current || loading || isLoadingMore) return;

    const observer = new IntersectionObserver(
      async (entries) => {
        const [entry] = entries;
        if (entry.isIntersecting && posts.length > 0) {
          setIsLoadingMore(true);
          await refetch();
          setIsLoadingMore(false);
        }
      },
      { threshold: 0.1, rootMargin: '100px' }
    );

    observer.observe(loadMoreRef.current);

    return () => {
      if (loadMoreRef.current) {
        observer.unobserve(loadMoreRef.current);
      }
    };
  }, [posts.length, loading, isLoadingMore, refetch]);

  // Memoized ad positions for consistent placement
  const adPositions = useMemo(() => {
    const positions = [];
    const totalPosts = posts.length;
    
    // Add ads every 6 posts for optimal user experience
    for (let i = 6; i < totalPosts; i += 6) {
      positions.push(i);
    }
    
    return positions;
  }, [posts.length]);

  // Loading state
  if (loading && posts.length === 0) {
    return (
      <div className="flex-1 bg-background min-h-0 flex flex-col">
        <SocialFeedHeader 
          searchQuery={searchQuery}
          onSearchChange={handleSearchChange}
          postsCount={0}
          isLoading={true}
        />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-400">Loading real AI-generated content...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-background min-h-0 flex flex-col">
      <SocialFeedHeader 
        searchQuery={searchQuery}
        onSearchChange={handleSearchChange}
        postsCount={postsCount}
        isLoading={loading}
      />
      
      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <div className="p-6">
          {/* Premium Banner Ad */}
          <div className="mb-6">
            <PremiumBannerAd placement="feed-top" />
          </div>

          {/* Posts Feed */}
          <div className="space-y-6">
            {posts.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-gray-400 mb-4">
                  {searchQuery ? 'No posts match your search criteria.' : 'No posts available.'}
                </p>
                {searchQuery && (
                  <button 
                    onClick={() => setSearchQuery('')}
                    className="text-primary hover:text-primary/80 underline"
                  >
                    Clear search
                  </button>
                )}
              </div>
            ) : (
              posts.map((post, index) => (
                <React.Fragment key={post.id}>
                  <PostCard 
                    post={post} 
                    onPostClick={handlePostClick}
                    onVoteChange={handleVoteChange}
                  />
                  
                  {/* Insert ads at calculated positions */}
                  {adPositions.includes(index + 1) && (
                    <div className="my-6">
                      {(index + 1) % 12 === 0 ? (
                        <NativeFeedAd index={index} />
                      ) : (
                        <InlineAd compact={false} />
                      )}
                    </div>
                  )}
                </React.Fragment>
              ))
            )}
          </div>

          {/* Infinite Scroll Trigger */}
          {posts.length > 0 && (
            <div ref={loadMoreRef} className="text-center mt-8 py-8">
              {isLoadingMore && (
                <div className="flex items-center justify-center space-x-2 text-primary">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Loading more content...</span>
                </div>
              )}
            </div>
          )}

          {/* Real-time Status Indicator */}
          <div className="text-center mt-6 text-sm text-gray-500">
            <p>Powered by real Pollen AI • Last updated: {new Date().toLocaleTimeString()}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// Export as default to maintain compatibility
export default SocialFeed;