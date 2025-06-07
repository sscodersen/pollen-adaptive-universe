
import React, { useState } from 'react';
import { Star, Trophy, TrendingUp, Play, Volume2, BookOpen, Gamepad2, Music, Sparkles } from 'lucide-react';

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
  const [isPlaying, setIsPlaying] = useState(false);
  const [userChoice, setUserChoice] = useState<string | null>(null);

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

  const getContentIcon = () => {
    switch (content.type) {
      case 'video': return <Play className="w-6 h-6" />;
      case 'audio': return <Volume2 className="w-6 h-6" />;
      case 'story': return <BookOpen className="w-6 h-6" />;
      case 'game': return <Gamepad2 className="w-6 h-6" />;
      case 'music': return <Music className="w-6 h-6" />;
      case 'interactive': return <Sparkles className="w-6 h-6" />;
      default: return <Play className="w-6 h-6" />;
    }
  };

  const handleChoice = (choice: string) => {
    setUserChoice(choice);
  };

  const renderInteractiveContent = () => {
    if (content.type === 'story' && content.content.includes('Choice')) {
      const parts = content.content.split('\n\n');
      return (
        <div className="space-y-6">
          {parts.map((part, index) => {
            if (part.includes('Choice')) {
              const choices = part.split('\n').filter(line => line.startsWith('Choice'));
              return (
                <div key={index} className="space-y-4">
                  <p className="text-gray-200 leading-relaxed mb-4">
                    {part.split('\n').filter(line => !line.startsWith('Choice')).join('\n')}
                  </p>
                  <div className="space-y-3">
                    {choices.map((choice, choiceIndex) => (
                      <button
                        key={choiceIndex}
                        onClick={() => handleChoice(choice)}
                        className={`w-full text-left p-4 rounded-lg border transition-all ${
                          userChoice === choice
                            ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                            : 'bg-gray-800/50 border-gray-700/50 text-gray-300 hover:bg-gray-700/50'
                        }`}
                      >
                        {choice}
                      </button>
                    ))}
                  </div>
                  {userChoice && (
                    <div className="mt-4 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                      <p className="text-green-400 text-sm">
                        You chose: {userChoice.replace('Choice ', '')}
                      </p>
                      <p className="text-gray-300 text-sm mt-2">
                        Your choice shapes the narrative! The story continues based on your decision...
                      </p>
                    </div>
                  )}
                </div>
              );
            }
            return (
              <p key={index} className="text-gray-200 leading-relaxed mb-4">
                {part}
              </p>
            );
          })}
        </div>
      );
    }

    if (content.type === 'game') {
      return (
        <div className="space-y-6">
          <div className="text-gray-200 leading-relaxed whitespace-pre-line mb-6">
            {content.content}
          </div>
          <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-6">
            <div className="text-center">
              <div className="w-full h-64 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 rounded-lg flex items-center justify-center mb-4">
                <Gamepad2 className="w-16 h-16 text-cyan-400" />
              </div>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 mx-auto ${
                  isPlaying
                    ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                    : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                }`}
              >
                <Gamepad2 className="w-5 h-5" />
                <span>{isPlaying ? 'Pause Game' : 'Start Game'}</span>
              </button>
              {isPlaying && (
                <p className="text-gray-400 text-sm mt-4">
                  Game simulation active! The AI is adapting to your play style...
                </p>
              )}
            </div>
          </div>
        </div>
      );
    }

    if (content.type === 'music' || content.type === 'audio') {
      return (
        <div className="space-y-6">
          <div className="text-gray-200 leading-relaxed whitespace-pre-line mb-6">
            {content.content}
          </div>
          <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-6">
            <div className="text-center">
              <div className="w-full h-32 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-lg flex items-center justify-center mb-4">
                <div className="flex space-x-2">
                  {[...Array(5)].map((_, i) => (
                    <div
                      key={i}
                      className={`w-2 bg-cyan-400 rounded-full ${
                        isPlaying ? 'animate-pulse' : ''
                      }`}
                      style={{
                        height: `${Math.random() * 40 + 20}px`,
                        animationDelay: `${i * 0.1}s`
                      }}
                    ></div>
                  ))}
                </div>
              </div>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 mx-auto ${
                  isPlaying
                    ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                    : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                }`}
              >
                {content.type === 'music' ? <Music className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                <span>{isPlaying ? 'Pause' : 'Play'}</span>
              </button>
              {isPlaying && (
                <p className="text-gray-400 text-sm mt-4">
                  Now playing: {content.title} • {content.duration}
                </p>
              )}
            </div>
          </div>
        </div>
      );
    }

    if (content.type === 'video') {
      return (
        <div className="space-y-6">
          <div className="text-gray-200 leading-relaxed whitespace-pre-line mb-6">
            {content.content}
          </div>
          <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-6">
            <div className="text-center">
              <div className="w-full h-64 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg flex items-center justify-center mb-4">
                <Play className="w-16 h-16 text-cyan-400" />
              </div>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 mx-auto ${
                  isPlaying
                    ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                    : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                }`}
              >
                <Play className="w-5 h-5" />
                <span>{isPlaying ? 'Pause Video' : 'Play Video'}</span>
              </button>
              {isPlaying && (
                <p className="text-gray-400 text-sm mt-4">
                  Video playing: {content.title} • {content.duration}
                </p>
              )}
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="text-gray-200 leading-relaxed whitespace-pre-line">
        {content.content}
      </div>
    );
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
          <div className="flex items-center space-x-3 mb-4">
            <div className="text-cyan-400">
              {getContentIcon()}
            </div>
            <h1 className="text-4xl font-bold text-white">{content.title}</h1>
          </div>
          <p className="text-xl text-gray-300 leading-relaxed">{content.description}</p>
        </div>

        <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-8">
          <div className="prose prose-invert max-w-none">
            {renderInteractiveContent()}
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
