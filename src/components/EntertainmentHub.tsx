
import React, { useState, useEffect } from 'react';
import { Play, Pause, Star, Clock, Users, Gamepad2, Film, Music, Sparkles, Download, ExternalLink } from 'lucide-react';
import { contentCurator, type WebContent } from '../services/contentCurator';

interface EntertainmentHubProps {
  isGenerating?: boolean;
}

export const EntertainmentHub = ({ isGenerating = true }: EntertainmentHubProps) => {
  const [content, setContent] = useState<WebContent[]>([]);
  const [generatingContent, setGeneratingContent] = useState(false);
  const [selectedType, setSelectedType] = useState('all');
  const [selectedGenre, setSelectedGenre] = useState('all');
  const [playingContent, setPlayingContent] = useState<string | null>(null);

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
        const curated = await contentCurator.scrapeAndCurateContent('entertainment', 15);
        setContent(prev => {
          const newItems = curated.filter(item => 
            !prev.some(existing => existing.title === item.title)
          );
          return [...newItems, ...prev].slice(0, 30);
        });
      } catch (error) {
        console.error('Failed to curate entertainment:', error);
      }
      setGeneratingContent(false);
    };

    generateContent();
    const interval = setInterval(generateContent, 50000);
    return () => clearInterval(interval);
  }, [isGenerating, generatingContent]);

  const handlePlay = (contentId: string, url: string) => {
    if (playingContent === contentId) {
      setPlayingContent(null);
    } else {
      setPlayingContent(contentId);
      // In production, this would actually launch/play the content
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  const filteredContent = content;

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Entertainment Discovery</h1>
            <p className="text-gray-400">AI-curated experiences • Interactive content • Ready to play</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingContent && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Discovering content...</span>
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
              <div className="h-48 bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center relative cursor-pointer"
                   onClick={() => handlePlay(item.id, item.url)}>
                {item.image ? (
                  <img src={item.image} alt={item.title} className="w-full h-full object-cover" />
                ) : (
                  <div className="absolute inset-0 bg-black/20" />
                )}
                
                <div className="absolute inset-0 flex items-center justify-center">
                  {playingContent === item.id ? (
                    <Pause className="w-12 h-12 text-white opacity-80 group-hover:opacity-100 group-hover:scale-110 transition-all" />
                  ) : (
                    <Play className="w-12 h-12 text-white opacity-80 group-hover:opacity-100 group-hover:scale-110 transition-all" />
                  )}
                </div>
                
                {/* Significance Badge */}
                <div className="absolute top-3 left-3 px-2 py-1 bg-black/50 backdrop-blur-sm rounded-full">
                  <div className="flex items-center space-x-1">
                    <Star className="w-3 h-3 text-yellow-400 fill-current" />
                    <span className="text-xs text-white font-medium">{item.significance.toFixed(1)}</span>
                  </div>
                </div>
                
                {/* Source */}
                <div className="absolute top-3 right-3 px-2 py-1 bg-black/50 backdrop-blur-sm rounded-full">
                  <span className="text-xs text-white font-medium">{item.source}</span>
                </div>
              </div>

              {/* Content Info */}
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
                      <span>Available now</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Users className="w-3 h-3" />
                      <span>Interactive</span>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center space-x-2">
                  <button 
                    onClick={() => handlePlay(item.id, item.url)}
                    className="flex-1 flex items-center justify-center space-x-2 px-4 py-2 bg-purple-500/20 border border-purple-500/30 rounded-lg text-purple-300 hover:bg-purple-500/30 transition-colors"
                  >
                    <Play className="w-4 h-4" />
                    <span className="font-medium">Experience</span>
                  </button>
                  
                  <button 
                    onClick={() => window.open(item.url, '_blank')}
                    className="p-2 bg-gray-700/50 border border-gray-600/50 rounded-lg text-gray-400 hover:text-white hover:bg-gray-600/50 transition-colors"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredContent.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <Sparkles className="w-10 h-10 text-white animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Curating Entertainment...</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Pollen is discovering the most engaging and interactive entertainment experiences from across the web.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
