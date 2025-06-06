
import React, { useState, useEffect } from 'react';
import { Play, Gamepad2, Music, Video, Sparkles, Send, Wand2, Star, Clock, Eye } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface EntertainmentContent {
  id: string;
  title: string;
  description: string;
  content: string;
  type: 'story' | 'game' | 'music' | 'video' | 'interactive';
  category: 'adventure' | 'puzzle' | 'creative' | 'educational' | 'social';
  duration: string;
  rating: number;
  views: number;
  isGenerated: boolean;
}

interface EntertainmentHubProps {
  isGenerating?: boolean;
}

export const EntertainmentHub = ({ isGenerating = true }: EntertainmentHubProps) => {
  const [content, setContent] = useState<EntertainmentContent[]>([]);
  const [userPrompt, setUserPrompt] = useState('');
  const [selectedType, setSelectedType] = useState('story');
  const [generatingContent, setGeneratingContent] = useState(false);
  const [selectedContent, setSelectedContent] = useState<EntertainmentContent | null>(null);

  const contentTypes = [
    { id: 'story', name: 'Interactive Stories', icon: Play },
    { id: 'game', name: 'Text Games', icon: Gamepad2 },
    { id: 'music', name: 'Music Ideas', icon: Music },
    { id: 'video', name: 'Video Concepts', icon: Video }
  ];

  useEffect(() => {
    if (!isGenerating) return;

    const generateStaticContent = async () => {
      if (generatingContent) return;
      
      setGeneratingContent(true);
      try {
        const types = ['story', 'game', 'music', 'video', 'interactive'] as const;
        const categories = ['adventure', 'puzzle', 'creative', 'educational', 'social'] as const;
        
        const randomType = types[Math.floor(Math.random() * types.length)];
        const randomCategory = categories[Math.floor(Math.random() * categories.length)];
        
        const response = await pollenAI.generate(
          `Create an engaging ${randomType} entertainment content for ${randomCategory} category. Make it interactive and immersive.`,
          "entertainment",
          true
        );
        
        const newContent: EntertainmentContent = {
          id: Date.now().toString(),
          title: generateContentTitle(randomType, randomCategory),
          description: response.content.slice(0, 150) + '...',
          content: response.content,
          type: randomType,
          category: randomCategory,
          duration: generateDuration(randomType),
          rating: Math.random() * 2 + 3.5,
          views: Math.floor(Math.random() * 10000) + 500,
          isGenerated: true
        };
        
        setContent(prev => [newContent, ...prev.slice(0, 11)]);
      } catch (error) {
        console.error('Failed to generate entertainment content:', error);
      }
      setGeneratingContent(false);
    };

    // Generate initial content and then periodically
    generateStaticContent();
    const interval = setInterval(generateStaticContent, Math.random() * 60000 + 90000);
    
    return () => clearInterval(interval);
  }, [isGenerating, generatingContent]);

  const generateContentTitle = (type: string, category: string) => {
    const titleTemplates = {
      story: [
        `The ${category.charAt(0).toUpperCase() + category.slice(1)} Chronicles`,
        `Journey Through the ${category.charAt(0).toUpperCase() + category.slice(1)} Realm`,
        `Echoes of ${category.charAt(0).toUpperCase() + category.slice(1)}`,
        `The Last ${category.charAt(0).toUpperCase() + category.slice(1)} Guardian`
      ],
      game: [
        `${category.charAt(0).toUpperCase() + category.slice(1)} Quest`,
        `The ${category.charAt(0).toUpperCase() + category.slice(1)} Challenge`,
        `Mind Maze: ${category.charAt(0).toUpperCase() + category.slice(1)}`,
        `${category.charAt(0).toUpperCase() + category.slice(1)} Master`
      ],
      music: [
        `Sounds of ${category.charAt(0).toUpperCase() + category.slice(1)}`,
        `${category.charAt(0).toUpperCase() + category.slice(1)} Symphony`,
        `Rhythms in ${category.charAt(0).toUpperCase() + category.slice(1)}`,
        `The ${category.charAt(0).toUpperCase() + category.slice(1)} Beat`
      ],
      video: [
        `${category.charAt(0).toUpperCase() + category.slice(1)} Vision`,
        `Through the ${category.charAt(0).toUpperCase() + category.slice(1)} Lens`,
        `${category.charAt(0).toUpperCase() + category.slice(1)} Stories`,
        `The ${category.charAt(0).toUpperCase() + category.slice(1)} Experience`
      ],
      interactive: [
        `Interactive ${category.charAt(0).toUpperCase() + category.slice(1)}`,
        `${category.charAt(0).toUpperCase() + category.slice(1)} Lab`,
        `Explore ${category.charAt(0).toUpperCase() + category.slice(1)}`,
        `${category.charAt(0).toUpperCase() + category.slice(1)} Playground`
      ]
    };

    const templates = titleTemplates[type as keyof typeof titleTemplates] || titleTemplates.story;
    return templates[Math.floor(Math.random() * templates.length)];
  };

  const generateDuration = (type: string) => {
    const durations = {
      story: ['15-30 min', '30-45 min', '45-60 min'],
      game: ['10-20 min', '20-40 min', '30-60 min'],
      music: ['3-5 min', '5-8 min', '8-12 min'],
      video: ['5-10 min', '10-15 min', '15-25 min'],
      interactive: ['10-30 min', '20-45 min', '30-60 min']
    };
    
    const typeDurations = durations[type as keyof typeof durations] || durations.story;
    return typeDurations[Math.floor(Math.random() * typeDurations.length)];
  };

  const handleGenerateFromPrompt = async () => {
    if (!userPrompt.trim() || generatingContent) return;
    
    setGeneratingContent(true);
    try {
      const response = await pollenAI.generate(
        `Create detailed ${selectedType} entertainment content based on this user request: "${userPrompt}". Make it engaging, interactive, and immersive.`,
        "entertainment",
        true
      );
      
      const newContent: EntertainmentContent = {
        id: Date.now().toString(),
        title: userPrompt.slice(0, 50) + (userPrompt.length > 50 ? '...' : ''),
        description: `Custom ${selectedType} generated from your prompt`,
        content: response.content,
        type: selectedType as any,
        category: 'creative',
        duration: generateDuration(selectedType),
        rating: 4.5,
        views: 1,
        isGenerated: true
      };
      
      setContent(prev => [newContent, ...prev]);
      setUserPrompt('');
    } catch (error) {
      console.error('Failed to generate custom content:', error);
    }
    setGeneratingContent(false);
  };

  const openContent = (contentItem: EntertainmentContent) => {
    setSelectedContent(contentItem);
  };

  const closeContent = () => {
    setSelectedContent(null);
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      adventure: 'text-orange-400 bg-orange-400/10 border-orange-400/20',
      puzzle: 'text-purple-400 bg-purple-400/10 border-purple-400/20',
      creative: 'text-pink-400 bg-pink-400/10 border-pink-400/20',
      educational: 'text-green-400 bg-green-400/10 border-green-400/20',
      social: 'text-blue-400 bg-blue-400/10 border-blue-400/20'
    };
    return colors[category as keyof typeof colors] || 'text-gray-400 bg-gray-400/10 border-gray-400/20';
  };

  const getTypeIcon = (type: string) => {
    const icons = {
      story: Play,
      game: Gamepad2,
      music: Music,
      video: Video,
      interactive: Sparkles
    };
    const IconComponent = icons[type as keyof typeof icons] || Play;
    return <IconComponent className="w-4 h-4" />;
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-800/50 bg-gray-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Entertainment Hub</h1>
            <p className="text-gray-400">AI-generated content • Interactive experiences • Custom creations</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingContent && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Creating content...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{content.length}</div>
              <div className="text-xs text-gray-400">Content pieces</div>
            </div>
          </div>
        </div>

        {/* Custom Prompt Input */}
        <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50 mb-4">
          <h3 className="text-lg font-semibold text-white mb-3">Create Custom Content</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="bg-gray-700/50 border border-gray-600/50 rounded-lg px-3 py-2 text-white"
            >
              {contentTypes.map(type => (
                <option key={type.id} value={type.id}>{type.name}</option>
              ))}
            </select>
            <div className="flex-1 flex items-center space-x-2 bg-gray-700/50 rounded-lg px-4 py-2 border border-gray-600/50 focus-within:border-cyan-500/50 transition-colors">
              <Wand2 className="w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Describe what you want to create..."
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleGenerateFromPrompt()}
                className="flex-1 bg-transparent text-white placeholder-gray-400 outline-none"
              />
              <button
                onClick={handleGenerateFromPrompt}
                disabled={!userPrompt.trim() || generatingContent}
                className="px-4 py-1 bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors"
              >
                Generate
              </button>
            </div>
          </div>
        </div>

        {/* Content Type Tabs */}
        <div className="flex space-x-2 overflow-x-auto">
          {contentTypes.map((type) => (
            <button
              key={type.id}
              className="flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap bg-gray-800/50 text-gray-300 hover:bg-gray-700/50 hover:text-white transition-colors"
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
          {content.map((item) => (
            <div key={item.id} 
                 className="bg-gray-900/80 rounded-xl border border-gray-800/50 overflow-hidden hover:bg-gray-900/90 transition-all duration-200 backdrop-blur-sm cursor-pointer"
                 onClick={() => openContent(item)}>
              
              {/* Content Preview */}
              <div className="h-32 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center relative">
                <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center">
                  {getTypeIcon(item.type)}
                </div>
                
                <div className="absolute top-3 left-3 flex items-center space-x-1">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getCategoryColor(item.category)}`}>
                    {item.category}
                  </span>
                </div>
                
                <div className="absolute top-3 right-3 px-2 py-1 bg-cyan-500/20 text-cyan-300 text-xs font-medium rounded-full">
                  AI Generated
                </div>
              </div>

              {/* Content Info */}
              <div className="p-5">
                <h3 className="font-bold text-white mb-2 line-clamp-2">
                  {item.title}
                </h3>
                
                <p className="text-gray-400 text-sm mb-4 line-clamp-2">
                  {item.description}
                </p>

                <div className="flex items-center justify-between text-sm text-gray-400 mb-3">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4" />
                    <span>{item.duration}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-400" />
                    <span className="text-yellow-400">{item.rating.toFixed(1)}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1 text-gray-400">
                    <Eye className="w-4 h-4" />
                    <span className="text-sm">{item.views.toLocaleString()} views</span>
                  </div>
                  
                  <div className="flex items-center space-x-1 text-cyan-400">
                    <Play className="w-4 h-4" />
                    <span className="text-sm font-medium">Experience</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {content.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-8">
              <Sparkles className="w-12 h-12 text-white animate-pulse" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">Entertainment Hub Loading...</h3>
            <p className="text-gray-400 max-w-lg mx-auto text-lg">
              Pollen AI is creating immersive entertainment experiences across stories, games, music, and interactive content.
            </p>
          </div>
        )}
      </div>

      {/* Content Modal */}
      {selectedContent && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 rounded-2xl border border-gray-800 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-800">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">{selectedContent.title}</h2>
                  <div className="flex items-center space-x-4 text-sm text-gray-400">
                    <span className={`px-2 py-1 rounded-full border ${getCategoryColor(selectedContent.category)}`}>
                      {selectedContent.category}
                    </span>
                    <span>{selectedContent.duration}</span>
                    <div className="flex items-center space-x-1">
                      <Star className="w-4 h-4 text-yellow-400" />
                      <span className="text-yellow-400">{selectedContent.rating.toFixed(1)}</span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={closeContent}
                  className="text-gray-400 hover:text-white transition-colors text-2xl"
                >
                  ×
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <div className="text-gray-200 leading-relaxed whitespace-pre-line">
                {selectedContent.content}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
