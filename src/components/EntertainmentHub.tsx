
import React, { useState, useEffect } from 'react';
import { Play, Film, Gamepad2, Music, Image, Sparkles, Clock } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface EntertainmentItem {
  id: string;
  title: string;
  description: string;
  type: 'movie' | 'game' | 'music' | 'story' | 'video';
  thumbnail: string;
  duration?: string;
  genre: string;
  rating: number;
  timestamp: string;
}

export const EntertainmentHub = () => {
  const [contentItems, setContentItems] = useState<EntertainmentItem[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedType, setSelectedType] = useState('all');

  const contentTypes = [
    { id: 'all', name: 'All Content', icon: Sparkles },
    { id: 'movie', name: 'Movies', icon: Film },
    { id: 'game', name: 'Games', icon: Gamepad2 },
    { id: 'music', name: 'Music', icon: Music },
    { id: 'story', name: 'Stories', icon: Image }
  ];

  useEffect(() => {
    const generateContent = async () => {
      if (isGenerating) return;
      
      setIsGenerating(true);
      try {
        const contentPrompts = [
          "Create an engaging short film concept with unique storyline",
          "Design a creative puzzle game with innovative mechanics",
          "Generate an ambient music composition concept",
          "Write an immersive interactive story experience",
          "Create a documentary concept about emerging technologies"
        ];
        
        const randomPrompt = contentPrompts[Math.floor(Math.random() * contentPrompts.length)];
        const response = await pollenAI.generate(randomPrompt, "creative");
        
        const newItem: EntertainmentItem = {
          id: Date.now().toString(),
          title: generateTitle(),
          description: response.content,
          type: getContentType(randomPrompt),
          thumbnail: generateThumbnail(),
          duration: generateDuration(),
          genre: generateGenre(),
          rating: Math.random() * 5,
          timestamp: new Date().toLocaleTimeString()
        };
        
        setContentItems(prev => [newItem, ...prev.slice(0, 11)]);
      } catch (error) {
        console.error('Failed to generate entertainment:', error);
      }
      setIsGenerating(false);
    };

    generateContent();
    const interval = setInterval(generateContent, 60000); // Generate every minute
    return () => clearInterval(interval);
  }, []);

  const generateTitle = () => {
    const titles = [
      "Quantum Dreams: The Awakening",
      "Neon Horizons: Interactive Journey",
      "Echoes of Tomorrow",
      "The Last Algorithm",
      "Synthetic Memories",
      "Digital Odyssey",
      "Void Runners",
      "Crystal Frequencies"
    ];
    return titles[Math.floor(Math.random() * titles.length)];
  };

  const getContentType = (prompt: string): EntertainmentItem['type'] => {
    if (prompt.includes('film')) return 'movie';
    if (prompt.includes('game')) return 'game';
    if (prompt.includes('music')) return 'music';
    if (prompt.includes('story')) return 'story';
    return 'movie';
  };

  const generateThumbnail = () => {
    const gradients = [
      'bg-gradient-to-br from-purple-500 to-pink-500',
      'bg-gradient-to-br from-blue-500 to-cyan-500',
      'bg-gradient-to-br from-orange-500 to-red-500',
      'bg-gradient-to-br from-green-500 to-teal-500',
      'bg-gradient-to-br from-indigo-500 to-purple-500'
    ];
    return gradients[Math.floor(Math.random() * gradients.length)];
  };

  const generateDuration = () => {
    const durations = ['2:30', '5:45', '12:15', '8:20', '15:30', '3:45'];
    return durations[Math.floor(Math.random() * durations.length)];
  };

  const generateGenre = () => {
    const genres = ['Sci-Fi', 'Adventure', 'Mystery', 'Drama', 'Action', 'Comedy', 'Thriller'];
    return genres[Math.floor(Math.random() * genres.length)];
  };

  const filteredContent = selectedType === 'all' 
    ? contentItems 
    : contentItems.filter(item => item.type === selectedType);

  return (
    <div className="flex-1 overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Entertainment Hub</h1>
            <p className="text-white/60">On-demand content generated from your preferences</p>
          </div>
          <div className="flex items-center space-x-2">
            {isGenerating && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Creating...</span>
              </div>
            )}
          </div>
        </div>

        {/* Content Type Filter */}
        <div className="flex space-x-2 overflow-x-auto">
          {contentTypes.map((type) => (
            <button
              key={type.id}
              onClick={() => setSelectedType(type.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                selectedType === type.id
                  ? 'bg-purple-500 text-white'
                  : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
              }`}
            >
              <type.icon className="w-4 h-4" />
              <span>{type.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredContent.map((item) => (
            <div key={item.id} className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/10 overflow-hidden hover:bg-white/15 transition-colors group">
              {/* Thumbnail */}
              <div className={`h-48 ${item.thumbnail} relative flex items-center justify-center`}>
                <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors" />
                <button className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-white/30 transition-colors">
                  <Play className="w-6 h-6 ml-1" />
                </button>
                {item.duration && (
                  <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/50 text-white text-xs rounded">
                    {item.duration}
                  </div>
                )}
              </div>

              {/* Content Info */}
              <div className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="px-2 py-1 bg-purple-500/20 text-purple-300 text-xs font-medium rounded">
                    {item.genre}
                  </span>
                  <div className="flex items-center space-x-1">
                    {[...Array(5)].map((_, i) => (
                      <div
                        key={i}
                        className={`w-2 h-2 rounded-full ${
                          i < Math.floor(item.rating) ? 'bg-yellow-400' : 'bg-white/20'
                        }`}
                      />
                    ))}
                  </div>
                </div>
                
                <h3 className="font-semibold text-white mb-2 leading-tight">
                  {item.title}
                </h3>
                
                <p className="text-white/70 text-sm leading-relaxed mb-3 line-clamp-3">
                  {item.description.slice(0, 120)}...
                </p>
                
                <div className="flex items-center justify-between text-xs text-white/60">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-3 h-3" />
                    <span>{item.timestamp}</span>
                  </div>
                  <span className="capitalize">{item.type}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredContent.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full flex items-center justify-center mx-auto mb-4">
              <Film className="w-8 h-8 text-white animate-pulse" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Crafting Entertainment...</h3>
            <p className="text-white/60">Pollen is generating personalized content based on your preferences</p>
          </div>
        )}
      </div>
    </div>
  );
};
