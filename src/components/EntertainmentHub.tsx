
import React, { useState, useEffect } from 'react';
import { Play, Pause, Star, Clock, Users, Gamepad2, Film, Music, Sparkles } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface Entertainment {
  id: string;
  title: string;
  description: string;
  type: 'game' | 'video' | 'audio' | 'interactive';
  genre: string;
  duration: string;
  rating: number;
  players: number;
  thumbnail: string;
  content: string;
  isGenerating?: boolean;
}

interface EntertainmentHubProps {
  isGenerating?: boolean;
}

export const EntertainmentHub = ({ isGenerating = true }: EntertainmentHubProps) => {
  const [content, setContent] = useState<Entertainment[]>([]);
  const [generatingContent, setGeneratingContent] = useState(false);
  const [selectedType, setSelectedType] = useState('all');
  const [selectedGenre, setSelectedGenre] = useState('all');

  const types = [
    { id: 'all', name: 'All', icon: Sparkles },
    { id: 'game', name: 'Games', icon: Gamepad2 },
    { id: 'video', name: 'Videos', icon: Film },
    { id: 'interactive', name: 'Interactive', icon: Star },
    { id: 'audio', name: 'Audio', icon: Music }
  ];

  const genres = ['all', 'adventure', 'puzzle', 'strategy', 'creative', 'educational', 'simulation'];

  useEffect(() => {
    if (!isGenerating) return;

    const generateContent = async () => {
      if (generatingContent) return;
      
      setGeneratingContent(true);
      try {
        const entertainmentTypes = ['interactive story game', 'puzzle adventure', 'creative sandbox', 'educational simulation'];
        const randomType = entertainmentTypes[Math.floor(Math.random() * entertainmentTypes.length)];
        
        const response = await pollenAI.generate(
          `Create an engaging ${randomType} experience with unique mechanics`,
          "entertainment"
        );
        
        const newContent: Entertainment = {
          id: Date.now().toString(),
          title: generateTitle(randomType),
          description: response.content.slice(0, 200) + '...',
          type: getContentType(randomType),
          genre: getGenre(randomType),
          duration: generateDuration(),
          rating: Math.random() * 2 + 3, // 3-5 stars
          players: Math.floor(Math.random() * 10) + 1,
          thumbnail: generateThumbnail(),
          content: response.content,
          isGenerating: false
        };
        
        setContent(prev => [newContent, ...prev.slice(0, 11)]);
      } catch (error) {
        console.error('Failed to generate entertainment:', error);
      }
      setGeneratingContent(false);
    };

    generateContent();
    const interval = setInterval(generateContent, 40000);
    return () => clearInterval(interval);
  }, [isGenerating, generatingContent]);

  const generateTitle = (type: string) => {
    const titles = [
      'Quantum Maze Explorer',
      'Neural Network Detective',
      'Creative Code Canvas',
      'Time Paradox Solver',
      'Memory Palace Builder',
      'Pattern Recognition Quest',
      'Logic Loop Adventure',
      'Cognitive Challenge Arena',
      'Synaptic Symphony',
      'Digital Dreamscape'
    ];
    return titles[Math.floor(Math.random() * titles.length)];
  };

  const getContentType = (type: string): Entertainment['type'] => {
    if (type.includes('game') || type.includes('puzzle')) return 'game';
    if (type.includes('story')) return 'interactive';
    if (type.includes('video')) return 'video';
    if (type.includes('audio')) return 'audio';
    return 'interactive';
  };

  const getGenre = (type: string) => {
    if (type.includes('puzzle')) return 'puzzle';
    if (type.includes('creative')) return 'creative';
    if (type.includes('adventure')) return 'adventure';
    if (type.includes('simulation')) return 'simulation';
    return 'adventure';
  };

  const generateDuration = () => {
    const durations = ['5-15 min', '15-30 min', '30-60 min', '1-2 hours', 'Ongoing'];
    return durations[Math.floor(Math.random() * durations.length)];
  };

  const generateThumbnail = () => {
    const colors = [
      'bg-gradient-to-br from-purple-500 to-pink-500',
      'bg-gradient-to-br from-blue-500 to-cyan-500',
      'bg-gradient-to-br from-green-500 to-blue-500',
      'bg-gradient-to-br from-orange-500 to-red-500',
      'bg-gradient-to-br from-indigo-500 to-purple-500'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  const filteredContent = content.filter(item => {
    if (selectedType !== 'all' && item.type !== selectedType) return false;
    if (selectedGenre !== 'all' && item.genre !== selectedGenre) return false;
    return true;
  });

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Entertainment Hub</h1>
            <p className="text-gray-400">On-demand generated experiences and interactive content</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingContent && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Creating...</span>
              </div>
            )}
            <span className="text-sm text-gray-400">{filteredContent.length} experiences</span>
          </div>
        </div>

        {/* Type Filter */}
        <div className="flex space-x-3 mb-4">
          {types.map((type) => (
            <button
              key={type.id}
              onClick={() => setSelectedType(type.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedType === type.id
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 hover:text-white'
              }`}
            >
              <type.icon className="w-4 h-4" />
              <span>{type.name}</span>
            </button>
          ))}
        </div>

        {/* Genre Filter */}
        <div className="flex space-x-2 overflow-x-auto">
          {genres.map((genre) => (
            <button
              key={genre}
              onClick={() => setSelectedGenre(genre)}
              className={`px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap transition-colors ${
                selectedGenre === genre
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                  : 'bg-gray-700/30 text-gray-400 border border-gray-600/30 hover:bg-gray-600/30'
              }`}
            >
              {genre.charAt(0).toUpperCase() + genre.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Content Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredContent.map((item) => (
            <div key={item.id} className="bg-gray-800/50 rounded-xl border border-gray-700/50 overflow-hidden hover:bg-gray-800/70 transition-all group">
              {/* Thumbnail */}
              <div className={`h-48 ${item.thumbnail} flex items-center justify-center relative`}>
                <div className="absolute inset-0 bg-black/20" />
                <div className="relative z-10 flex items-center justify-center">
                  <Play className="w-12 h-12 text-white opacity-80 group-hover:opacity-100 group-hover:scale-110 transition-all" />
                </div>
                
                {/* Type Badge */}
                <div className="absolute top-3 left-3 px-2 py-1 bg-black/50 backdrop-blur-sm rounded-full">
                  <span className="text-xs text-white font-medium capitalize">{item.type}</span>
                </div>
                
                {/* Rating */}
                <div className="absolute top-3 right-3 flex items-center space-x-1 px-2 py-1 bg-black/50 backdrop-blur-sm rounded-full">
                  <Star className="w-3 h-3 text-yellow-400 fill-current" />
                  <span className="text-xs text-white font-medium">{item.rating.toFixed(1)}</span>
                </div>
              </div>

              {/* Content */}
              <div className="p-5">
                <h3 className="text-lg font-bold text-white mb-2 group-hover:text-purple-300 transition-colors">
                  {item.title}
                </h3>
                
                <p className="text-gray-400 text-sm mb-4 line-clamp-2">
                  {item.description}
                </p>

                {/* Metadata */}
                <div className="flex items-center justify-between text-xs text-gray-400 mb-4">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>{item.duration}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Users className="w-3 h-3" />
                      <span>{item.players} player{item.players > 1 ? 's' : ''}</span>
                    </div>
                  </div>
                  <span className="capitalize text-purple-400">{item.genre}</span>
                </div>

                {/* Action Button */}
                <button className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-purple-500/20 border border-purple-500/30 rounded-lg text-purple-300 hover:bg-purple-500/30 transition-colors">
                  <Play className="w-4 h-4" />
                  <span className="font-medium">Experience</span>
                </button>
              </div>
            </div>
          ))}
        </div>

        {filteredContent.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <Sparkles className="w-10 h-10 text-white animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Creating New Experiences...</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Pollen is generating personalized entertainment content based on your preferences and interaction patterns.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
