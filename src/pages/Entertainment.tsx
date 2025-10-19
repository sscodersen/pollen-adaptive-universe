import React, { useState, useEffect, useCallback } from 'react';
import { Film, Music, Tv, Gamepad2, Play, Star, Clock, TrendingUp, Sparkles, Heart, Share2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { contentOrchestrator } from '@/services/contentOrchestrator';
import { EntertainmentContent } from '@/services/unifiedContentEngine';
import { personalizationEngine } from '@/services/personalizationEngine';

const ENTERTAINMENT_CATEGORIES = [
  { id: 'all', label: 'All', icon: Sparkles },
  { id: 'movie', label: 'Movies', icon: Film },
  { id: 'series', label: 'TV Shows', icon: Tv },
  { id: 'music_video', label: 'Music Videos', icon: Music },
  { id: 'documentary', label: 'Documentaries', icon: Gamepad2 }
];

const Entertainment = () => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [content, setContent] = useState<EntertainmentContent[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadContent();
  }, [selectedCategory]);

  const loadContent = async () => {
    setLoading(true);
    try {
      const query = selectedCategory === 'all'
        ? 'trending entertainment content and recommendations'
        : `top ${selectedCategory} recommendations and trending content`;

      const response = await contentOrchestrator.generateContent<EntertainmentContent>({
        type: 'entertainment',
        query,
        count: 24,
        strategy: {
          diversity: 0.95,
          freshness: 0.9,
          personalization: 0.85,
          qualityThreshold: 8.0,
          trendingBoost: 0.8
        }
      });

      const personalized = await personalizationEngine.generateRecommendations(
        response.content,
        24
      );

      setContent(personalized);
    } catch (error) {
      console.error('Failed to load entertainment:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFavorite = useCallback((contentId: string) => {
    setFavorites(prev => {
      const newSet = new Set(prev);
      if (newSet.has(contentId)) {
        newSet.delete(contentId);
      } else {
        newSet.add(contentId);
      }
      return newSet;
    });

    personalizationEngine.trackBehavior({
      action: 'like',
      contentId,
      contentType: 'entertainment',
      metadata: { category: selectedCategory }
    });
  }, [selectedCategory]);

  const filteredContent = searchQuery
    ? content.filter(item =>
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : content;

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl">
              <Film className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                Entertainment Hub
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                AI-curated movies, shows, and more
              </p>
            </div>
          </div>

          {/* Search */}
          <Input
            type="text"
            placeholder="Search entertainment..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="max-w-2xl h-12"
          />
        </div>

        {/* Categories */}
        <div className="flex gap-2 mb-8 overflow-x-auto scrollbar-thin pb-2">
          {ENTERTAINMENT_CATEGORIES.map(category => (
            <Button
              key={category.id}
              variant={selectedCategory === category.id ? 'default' : 'outline'}
              onClick={() => setSelectedCategory(category.id)}
              className="whitespace-nowrap"
            >
              <category.icon className="w-4 h-4 mr-2" />
              {category.label}
            </Button>
          ))}
        </div>

        {/* Content Grid */}
        {loading ? (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
            {[...Array(12)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="bg-gray-300 dark:bg-gray-700 rounded-xl aspect-[2/3]" />
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
            {filteredContent.map(item => (
              <div
                key={item.id}
                className="group relative bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-xl overflow-hidden hover:shadow-2xl hover:scale-105 transition-all duration-300"
              >
                {/* Thumbnail */}
                <div className="relative aspect-[2/3] bg-gradient-to-br from-purple-400 to-pink-400">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Play className="w-12 h-12 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                  
                  {/* Rating Badge */}
                  <div className="absolute top-2 left-2 px-2 py-1 bg-black/70 backdrop-blur-sm rounded-lg flex items-center gap-1">
                    <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                    <span className="text-xs text-white font-medium">{item.rating}</span>
                  </div>

                  {/* Favorite Button */}
                  <button
                    onClick={() => handleFavorite(item.id)}
                    className="absolute top-2 right-2 p-2 bg-black/50 backdrop-blur-sm rounded-lg hover:bg-black/70 transition-colors"
                  >
                    <Heart
                      className={`w-4 h-4 ${favorites.has(item.id) ? 'fill-red-500 text-red-500' : 'text-white'}`}
                    />
                  </button>
                </div>

                {/* Info */}
                <div className="p-3">
                  <h3 className="font-semibold text-sm text-gray-900 dark:text-white line-clamp-2 mb-1">
                    {item.title}
                  </h3>
                  <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span className="capitalize">{item.genre}</span>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {item.duration}
                    </div>
                  </div>
                  {item.trending && (
                    <div className="mt-2 px-2 py-1 bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded flex items-center gap-1">
                      <TrendingUp className="w-3 h-3 text-orange-600" />
                      <span className="text-xs font-medium text-orange-600">Trending</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Entertainment;
