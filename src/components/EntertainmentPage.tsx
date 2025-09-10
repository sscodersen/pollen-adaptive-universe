
import React, { useState, useEffect, useMemo } from 'react';
import { Film, Play, Star, Clock, Eye, Headphones, Camera, Gamepad2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ContentRequestBar } from "@/components/ContentRequestBar";
import { AdSpace } from "@/components/AdSpace";
import { contentOrchestrator } from '../services/contentOrchestrator';

const staticContent = [
  {
    id: 1,
    title: 'The Quantum Paradox',
    type: 'Movie',
    genre: 'Sci-Fi Thriller',
    duration: '2h 18m',
    rating: 4.8,
    views: '2.4M',
    thumbnail: 'bg-gradient-to-br from-purple-600 to-blue-600',
    description: 'A physicist discovers parallel universes are bleeding into reality through quantum experiments.',
    trending: true
  },
  {
    id: 2,
    title: 'Echoes of Tomorrow',
    type: 'Series',
    genre: 'Drama',
    duration: '8 episodes',
    rating: 4.6,
    views: '1.8M',
    thumbnail: 'bg-gradient-to-br from-emerald-500 to-cyan-500',
    description: 'A group of time travelers try to prevent an apocalyptic future.',
    trending: false
  },
  {
    id: 3,
    title: 'Digital Souls',
    type: 'Documentary',
    genre: 'Technology',
    duration: '1h 45m',
    rating: 4.9,
    views: '890K',
    thumbnail: 'bg-gradient-to-br from-orange-500 to-red-500',
    description: 'Exploring the intersection of consciousness and artificial intelligence.',
    trending: true
  },
  {
    id: 4,
    title: 'Neon Dreams',
    type: 'Music Video',
    genre: 'Electronic',
    duration: '4m 32s',
    rating: 4.7,
    views: '3.2M',
    thumbnail: 'bg-gradient-to-br from-pink-500 to-purple-500',
    description: 'A cyberpunk journey through sound and visual artistry.',
    trending: true
  }
];

const categories = [
  { name: 'Movies', icon: Film, count: 1234 },
  { name: 'Series', icon: Play, count: 567 },
  { name: 'Music', icon: Headphones, count: 2890 },
  { name: 'Documentaries', icon: Camera, count: 445 },
  { name: 'Gaming', icon: Gamepad2, count: 1678 }
];

export function EntertainmentPage() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [entertainmentContent, setEntertainmentContent] = useState(staticContent);
  const [userGeneratedContent, setUserGeneratedContent] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  const handleContentRequest = (contentData: any) => {
    const newContent = {
      id: Date.now(),
      title: contentData.content.title || 'Generated Content',
      type: contentData.content.type || 'Movie',
      genre: contentData.content.genre || 'Entertainment',
      duration: contentData.content.duration || '2h 0m',
      rating: contentData.confidence,
      views: `${(Math.random() * 2 + 0.5).toFixed(1)}M`,
      thumbnail: 'bg-gradient-to-br from-cyan-500 to-purple-500',
      description: contentData.content.description || 'User-generated entertainment content',
      trending: true,
      userGenerated: true
    };
    
    setUserGeneratedContent(prev => [newContent, ...prev]);
  };

  useEffect(() => {
    const generateContent = async () => {
      setIsLoading(true);
      try {
        const { content: trendingContent } = await contentOrchestrator.generateContent({
          type: 'entertainment',
          count: 8,
          strategy: {
            diversity: 0.8,
            freshness: 0.9,
            qualityThreshold: 7,
            personalization: 0.5,
            trendingBoost: 1.3
          }
        });

        const formattedContent = trendingContent.map((item: any, index: number) => ({
          id: index + 1,
          title: item.title,
          type: item.contentType === 'movie' ? 'Movie' : 
                item.contentType === 'series' ? 'Series' :
                item.contentType === 'documentary' ? 'Documentary' : 'Music Video',
          genre: item.genre,
          duration: item.duration,
          rating: item.rating,
          views: `${(Math.random() * 5 + 0.5).toFixed(1)}M`,
          thumbnail: item.thumbnail,
          description: item.description,
          trending: item.trending
        }));

        setEntertainmentContent([...formattedContent, ...staticContent]);
      } catch (error) {
        console.error('Failed to generate entertainment content:', error);
        setEntertainmentContent(staticContent);
      } finally {
        setIsLoading(false);
        setIsInitialLoad(false);
      }
    };

    generateContent();
    const interval = setInterval(generateContent, 5 * 60 * 1000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  // Combine all content and apply filtering
  const allContent = useMemo(() => {
    return [...userGeneratedContent, ...entertainmentContent];
  }, [userGeneratedContent, entertainmentContent]);

  const filteredContent = useMemo(() => {
    if (selectedCategory === 'all') {
      return allContent;
    }

    const categoryTypeMap: { [key: string]: string[] } = {
      'movies': ['Movie'],
      'series': ['Series'],
      'music': ['Music Video'],
      'documentaries': ['Documentary'],
      'gaming': ['Game', 'Gaming']
    };

    const allowedTypes = categoryTypeMap[selectedCategory] || [selectedCategory];
    return allContent.filter(content => 
      allowedTypes.some(type => 
        content.type.toLowerCase().includes(type.toLowerCase())
      )
    );
  }, [allContent, selectedCategory]);

  // Create skeleton loading component
  const ContentSkeleton = () => (
    <div className="glass-card overflow-hidden">
      <Skeleton className="h-48 w-full bg-white/10" />
      <div className="p-4 space-y-3">
        <div className="flex gap-2">
          <Skeleton className="h-4 w-16 bg-white/10" />
          <Skeleton className="h-4 w-20 bg-white/10" />
        </div>
        <Skeleton className="h-5 w-3/4 bg-white/10" />
        <Skeleton className="h-8 w-full bg-white/10" />
        <div className="flex justify-between">
          <Skeleton className="h-4 w-16 bg-white/10" />
          <Skeleton className="h-4 w-16 bg-white/10" />
        </div>
        <div className="flex justify-between items-center">
          <Skeleton className="h-4 w-12 bg-white/10" />
          <Skeleton className="h-8 w-20 bg-white/10" />
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex-1 liquid-gradient-animated min-h-screen">
      {/* Header with liquid glass design */}
      <div className="glass-nav p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <Film className="w-8 h-8 text-cyan-400" />
            Entertainment
          </h1>
          <p className="text-white/60">Discover movies, series, music, and more curated content</p>
        </div>
      </div>

      {/* Content Request Bar */}
      <div className="max-w-6xl mx-auto px-6 py-4">
        <ContentRequestBar 
          mode="entertainment" 
          onContentGenerated={handleContentRequest}
          placeholder="Generate entertainment content like movies, series, music videos..."
        />
      </div>

      <div className="max-w-6xl mx-auto p-6">
        {/* Categories with liquid glass design */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-6">Browse Categories</h2>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <button
              onClick={() => setSelectedCategory('all')}
              className={`glass-card p-6 hover:scale-105 transition-all duration-300 group ${
                selectedCategory === 'all' ? 'ring-2 ring-cyan-400 bg-cyan-500/10' : ''
              }`}
              data-testid="category-all"
            >
              <div className="flex flex-col items-center gap-3">
                <div className="p-3 liquid-gradient-accent rounded-xl group-hover:scale-110 transition-transform">
                  <Star className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">All</h3>
                  <p className="text-sm text-white/60">{allContent.length} items</p>
                </div>
              </div>
            </button>
            {categories.map((category, index) => (
              <button
                key={index}
                onClick={() => setSelectedCategory(category.name.toLowerCase())}
                className={`glass-card p-6 hover:scale-105 transition-all duration-300 group ${
                  selectedCategory === category.name.toLowerCase() ? 'ring-2 ring-cyan-400 bg-cyan-500/10' : ''
                }`}
                data-testid={`category-${category.name.toLowerCase()}`}
              >
                <div className="flex flex-col items-center gap-3">
                  <div className="p-3 liquid-gradient-accent rounded-xl group-hover:scale-110 transition-transform">
                    <category.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">{category.name}</h3>
                    <p className="text-sm text-white/60">{category.count} items</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Inline Ad Space */}
        <AdSpace size="native" position="inline" category="lifestyle" className="mb-8" />

        {/* Featured Content */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white">
              {selectedCategory === 'all' ? 'Trending Now' : `${selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1)} Content`}
              {isLoading && !isInitialLoad && (
                <span className="ml-2 text-sm text-white/60">(Refreshing...)</span>
              )}
            </h2>
            <div className="flex items-center gap-2">
              <span className="text-sm text-white/60">{filteredContent.length} items</span>
              <Button className="glass-button text-white border-white/10 hover:scale-105">
                View All
              </Button>
            </div>
          </div>
          
          {isInitialLoad ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {Array(8).fill(0).map((_, index) => (
                <ContentSkeleton key={index} />
              ))}
            </div>
          ) : filteredContent.length === 0 ? (
            <div className="glass-card p-12 text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-white/10 rounded-full flex items-center justify-center">
                <Film className="w-8 h-8 text-white/60" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">No content found</h3>
              <p className="text-white/60 mb-4">
                No {selectedCategory === 'all' ? 'content' : selectedCategory} available at the moment.
              </p>
              <Button 
                onClick={() => setSelectedCategory('all')}
                className="glass-button text-white border-white/10 hover:scale-105"
              >
                View All Content
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {filteredContent.map((content) => (
              <div 
                key={content.id} 
                className="glass-card overflow-hidden hover:scale-105 transition-all duration-300 group cursor-pointer liquid-border"
                data-testid={`entertainment-content-${content.id}`}
              >
                {/* Thumbnail */}
                <div className={`${content.thumbnail} h-48 relative flex items-center justify-center group-hover:scale-110 transition-transform overflow-hidden`}>
                  <Play className="w-12 h-12 text-white/80 group-hover:text-white group-hover:scale-125 transition-all" />
                  {content.trending && (
                    <Badge className="absolute top-3 left-3 liquid-gradient-warm hover:liquid-gradient-warm border-none">
                      Trending
                    </Badge>
                  )}
                  {/* Hover overlay */}
                  <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
                
                {/* Content Info */}
                <div className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-medium text-white liquid-gradient-secondary px-2 py-1 rounded-lg">
                      {content.type}
                    </span>
                    <span className="text-xs text-white/60">{content.genre}</span>
                    {content.userGenerated && (
                      <Badge className="text-xs liquid-gradient-warm border-none">
                        Generated
                      </Badge>
                    )}
                  </div>
                  
                  <h3 className="font-semibold text-white mb-2 group-hover:text-cyan-300 transition-colors">
                    {content.title}
                  </h3>
                  
                  <p className="text-sm text-white/60 mb-3 line-clamp-2">
                    {content.description}
                  </p>
                  
                  <div className="flex items-center justify-between text-sm text-white/50 mb-3">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {content.duration}
                    </div>
                    <div className="flex items-center gap-1">
                      <Eye className="w-4 h-4" />
                      {content.views}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <Star className="w-4 h-4 text-yellow-400 fill-current" />
                      <span className="text-sm font-medium text-white">{content.rating}</span>
                    </div>
                    <Button size="sm" className="glass-button hover:scale-105">
                      Watch Now
                    </Button>
                  </div>
                </div>
              </div>
              ))}
            </div>
          )}
        </div>

        {/* Recommended Playlists */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Curated Playlists</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { name: 'Future Sci-Fi', count: 24, color: 'from-blue-500 to-purple-500' },
              { name: 'Mind-Bending Docs', count: 18, color: 'from-emerald-500 to-cyan-500' },
              { name: 'Electronic Vibes', count: 156, color: 'from-pink-500 to-orange-500' }
            ].map((playlist, index) => (
              <div key={index} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-colors">
                <div className={`w-16 h-16 bg-gradient-to-br ${playlist.color} rounded-lg mb-4 flex items-center justify-center`}>
                  <Play className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{playlist.name}</h3>
                <p className="text-gray-400 mb-4">{playlist.count} items</p>
                <Button variant="outline" size="sm" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                  Explore Playlist
                </Button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
