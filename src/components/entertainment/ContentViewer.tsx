
import React from 'react';
import { Star, Trophy, TrendingUp } from 'lucide-react';

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

interface ContentViewerProps {
  content: ContentItem;
  onBack: () => void;
}

export const ContentViewer = ({ content, onBack }: ContentViewerProps) => {
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
    <div className="flex-1 bg-gray-950">
      {/* Content Header */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <button
              onClick={onBack}
              className="flex items-center space-x-2 text-cyan-400 hover:text-cyan-300 transition-colors"
            >
              <span>← Back to Entertainment</span>
            </button>
            <div className="flex items-center space-x-3">
              <div className={`px-3 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${
                content.significance > 8 
                  ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                  : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
              }`}>
                <Trophy className="w-3 h-3" />
                <span>{content.significance.toFixed(1)} Quality</span>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${getRatingColor(content.rating)} bg-gray-800/50 border border-gray-700/50`}>
                <Star className="w-3 h-3 fill-current" />
                <span>{content.rating}/5.0</span>
              </div>
              {content.difficulty && (
                <div className={`px-3 py-1 rounded-full text-xs font-medium border ${getDifficultyBadge(content.difficulty)}`}>
                  {content.difficulty}
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center space-x-6 text-gray-400 text-sm">
            <span>{content.views.toLocaleString()} views</span>
            <span>{content.likes.toLocaleString()} likes</span>
            <span>{content.shares.toLocaleString()} shares</span>
            <span>{content.comments.toLocaleString()} comments</span>
          </div>
        </div>
      </div>

      {/* Content Display */}
      <div className="p-6 max-w-4xl mx-auto">
        <div className="mb-6">
          <div className="flex items-center space-x-4 mb-4">
            <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm border border-purple-500/30">
              {content.category}
            </span>
            <span className="text-gray-400 text-sm">{content.duration}</span>
            {content.trending && (
              <div className="flex items-center space-x-1 px-3 py-1 bg-red-500/20 text-red-300 rounded-full text-sm border border-red-500/30">
                <TrendingUp className="w-3 h-3" />
                <span>Trending</span>
              </div>
            )}
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">{content.title}</h1>
          <p className="text-xl text-gray-300 leading-relaxed">{content.description}</p>
        </div>

        <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-8">
          <div className="prose prose-invert max-w-none">
            <div className="text-gray-200 leading-relaxed whitespace-pre-line">
              {content.content}
            </div>
          </div>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mt-8 pt-6 border-t border-gray-800/50">
          {content.tags.map((tag, index) => (
            <span 
              key={index} 
              className={`px-3 py-1 rounded-full text-sm border ${
                tag === 'trending' || tag === 'viral' 
                  ? 'bg-red-500/20 text-red-300 border-red-500/30'
                  : tag === 'new' || tag === 'custom'
                  ? 'bg-green-500/20 text-green-300 border-green-500/30'
                  : 'bg-gray-700/50 text-gray-300 border-gray-600/50'
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
