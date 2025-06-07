
import React from 'react';
import { Play, Star, Trophy, TrendingUp, Clock } from 'lucide-react';

interface ContentItem {
  id: string;
  title: string;
  description: string;
  type: 'video' | 'audio' | 'story' | 'game' | 'music' | 'interactive';
  content: string;
  duration: string;
  category: string;
  tags: string[];
  significance: number;
  trending: boolean;
  views: number;
  likes: number;
  shares: number;
  comments: number;
  rating: number;
  difficulty?: string;
  thumbnail?: string;
}

interface ContentCardProps {
  item: ContentItem;
  onClick: (item: ContentItem) => void;
}

export const ContentCard = ({ item, onClick }: ContentCardProps) => {
  const getRatingColor = (rating: number) => {
    if (rating >= 4.5) return 'text-green-400';
    if (rating >= 4.0) return 'text-yellow-400';
    return 'text-orange-400';
  };

  const getDifficultyBadge = (difficulty?: string) => {
    if (!difficulty) return null;
    const colors = {
      'Easy': 'bg-green-500/20 text-green-300 border-green-500/30',
      'Medium': 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
      'Hard': 'bg-red-500/20 text-red-300 border-red-500/30',
      'Adaptive': 'bg-purple-500/20 text-purple-300 border-purple-500/30'
    };
    return colors[difficulty as keyof typeof colors] || colors.Medium;
  };

  return (
    <div
      onClick={() => onClick(item)}
      className="group relative bg-gray-900/50 backdrop-blur-sm border border-gray-800/50 rounded-xl overflow-hidden hover:border-cyan-500/50 transition-all duration-300 cursor-pointer hover:scale-[1.02]"
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-800/50">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-white group-hover:text-cyan-300 transition-colors line-clamp-2">
              {item.title}
            </h3>
            <p className="text-gray-400 text-sm mt-1 line-clamp-2">{item.description}</p>
          </div>
          <div className="ml-3 flex-shrink-0">
            <Play className="w-8 h-8 text-cyan-400 group-hover:text-cyan-300 transition-colors" />
          </div>
        </div>

        <div className="flex items-center space-x-3 text-sm">
          <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded-full text-xs border border-purple-500/30">
            {item.category}
          </span>
          <div className="flex items-center space-x-1 text-gray-400">
            <Clock className="w-3 h-3" />
            <span>{item.duration}</span>
          </div>
          {item.trending && (
            <div className="flex items-center space-x-1 px-2 py-1 bg-red-500/20 text-red-300 rounded-full text-xs border border-red-500/30">
              <TrendingUp className="w-3 h-3" />
              <span>Trending</span>
            </div>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="p-4">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4 text-gray-400">
            <span>{item.views.toLocaleString()} views</span>
            <span>{item.likes.toLocaleString()} likes</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className={`flex items-center space-x-1 ${getRatingColor(item.rating)}`}>
              <Star className="w-4 h-4 fill-current" />
              <span className="font-medium">{item.rating}</span>
            </div>
            <div className={`px-2 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${
              item.significance > 8 
                ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
            }`}>
              <Trophy className="w-3 h-3" />
              <span>{item.significance.toFixed(1)}</span>
            </div>
          </div>
        </div>

        {item.difficulty && (
          <div className="mt-3">
            <div className={`inline-flex px-2 py-1 rounded-full text-xs font-medium border ${getDifficultyBadge(item.difficulty)}`}>
              {item.difficulty}
            </div>
          </div>
        )}

        {/* Tags */}
        <div className="flex flex-wrap gap-1 mt-3">
          {item.tags.slice(0, 3).map((tag, index) => (
            <span 
              key={index} 
              className={`px-2 py-1 rounded-full text-xs border ${
                tag === 'trending' || tag === 'viral' 
                  ? 'bg-red-500/10 text-red-400 border-red-500/20'
                  : tag === 'new' || tag === 'custom'
                  ? 'bg-green-500/10 text-green-400 border-green-500/20'
                  : 'bg-gray-700/30 text-gray-400 border-gray-600/30'
              }`}
            >
              #{tag}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};
