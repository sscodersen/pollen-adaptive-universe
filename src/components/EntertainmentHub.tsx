
import React, { useState, useEffect } from 'react';
import { Play, Music, Gamepad2, Sparkles, Volume2, Video, Headphones, Film, Zap, Download, Star } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface EntertainmentItem {
  id: string;
  title: string;
  description: string;
  type: 'video' | 'music' | 'game' | 'story' | 'podcast' | 'interactive';
  category: 'creative' | 'educational' | 'relaxation' | 'adventure' | 'social' | 'productivity';
  duration?: string;
  rating: number;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  tags: string[];
  downloadable: boolean;
}

interface EntertainmentHubProps {
  isGenerating?: boolean;
}

export const EntertainmentHub = ({ isGenerating = true }: EntertainmentHubProps) => {
  const [entertainmentItems, setEntertainmentItems] = useState<EntertainmentItem[]>([]);
  const [generatingContent, setGeneratingContent] = useState(false);
  const [selectedType, setSelectedType] = useState('all');

  const contentTypes = ['all', 'video', 'music', 'game', 'story', 'podcast', 'interactive'];

  const entertainmentTemplates = {
    video: [
      "immersive documentary exploring breakthrough innovations",
      "creative tutorial series on emerging technologies",
      "visual storytelling piece about global solutions",
      "animated explanation of complex scientific concepts"
    ],
    music: [
      "AI-generated ambient soundscape for focus",
      "adaptive musical composition based on productivity patterns",
      "collaborative symphony featuring global artists",
      "therapeutic sound design for relaxation and creativity"
    ],
    game: [
      "problem-solving puzzle game with real-world applications",
      "collaborative strategy game promoting teamwork",
      "educational simulation teaching sustainability concepts",
      "interactive decision-making game about future scenarios"
    ],
    story: [
      "interactive narrative about technological possibilities",
      "collaborative storytelling platform for communities",
      "adaptive fiction that responds to reader choices",
      "immersive world-building experience with AI characters"
    ],
    podcast: [
      "expert interview series on innovation and creativity",
      "community-driven discussion about future solutions",
      "guided meditation for productivity and well-being",
      "educational series breaking down complex topics"
    ],
    interactive: [
      "virtual reality experience showcasing sustainable futures",
      "augmented reality tool for creative collaboration",
      "interactive workshop on innovation methodologies",
      "immersive simulation of breakthrough technologies"
    ]
  };

  useEffect(() => {
    if (!isGenerating) return;

    const generateContent = async () => {
      if (generatingContent) return;
      
      setGeneratingContent(true);
      try {
        const types = Object.keys(entertainmentTemplates) as (keyof typeof entertainmentTemplates)[];
        const categories = ['creative', 'educational', 'relaxation', 'adventure', 'social', 'productivity'] as const;
        
        const randomType = types[Math.floor(Math.random() * types.length)];
        const randomCategory = categories[Math.floor(Math.random() * categories.length)];
        const templates = entertainmentTemplates[randomType];
        const randomTemplate = templates[Math.floor(Math.random() * templates.length)];
        
        const response = await pollenAI.generate(
          `Create ${randomType} entertainment content: ${randomTemplate}. Make it engaging, innovative, and valuable for ${randomCategory} purposes.`,
          "entertainment",
          true
        );
        
        const newItem: EntertainmentItem = {
          id: Date.now().toString() + Math.random(),
          title: generateTitle(randomType, randomTemplate),
          description: response.content,
          type: randomType,
          category: randomCategory,
          duration: generateDuration(randomType),
          rating: Math.round((4.0 + Math.random() * 1.0) * 10) / 10,
          difficulty: ['beginner', 'intermediate', 'advanced'][Math.floor(Math.random() * 3)] as any,
          tags: generateTags(randomType, randomCategory),
          downloadable: ['music', 'podcast', 'story'].includes(randomType)
        };
        
        setEntertainmentItems(prev => [newItem, ...prev.slice(0, 19)]);
      } catch (error) {
        console.error('Failed to generate entertainment content:', error);
      }
      setGeneratingContent(false);
    };

    const initialTimeout = setTimeout(generateContent, 2000);
    const interval = setInterval(generateContent, Math.random() * 30000 + 40000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingContent]);

  const generateTitle = (type: string, template: string) => {
    const words = template.split(' ');
    const keyWords = words.slice(0, 4).join(' ');
    return keyWords.charAt(0).toUpperCase() + keyWords.slice(1);
  };

  const generateDuration = (type: string) => {
    const durations = {
      video: ['5 min', '12 min', '25 min', '45 min'],
      music: ['3 min', '8 min', '15 min', '30 min'],
      game: ['10 min', '30 min', '1 hour', '2 hours'],
      story: ['15 min', '30 min', '1 hour', '2 hours'],
      podcast: ['20 min', '45 min', '1 hour', '1.5 hours'],
      interactive: ['15 min', '30 min', '1 hour', 'Variable']
    };
    const options = durations[type as keyof typeof durations] || ['Variable'];
    return options[Math.floor(Math.random() * options.length)];
  };

  const generateTags = (type: string, category: string) => {
    const baseTags = {
      video: ['Visual', 'Immersive', 'HD'],
      music: ['Audio', 'Rhythmic', 'Atmospheric'],
      game: ['Interactive', 'Strategic', 'Engaging'],
      story: ['Narrative', 'Character-driven', 'Immersive'],
      podcast: ['Audio', 'Discussion', 'Expert'],
      interactive: ['Hands-on', 'Collaborative', 'Dynamic']
    };
    
    const categoryTags = {
      creative: ['Artistic', 'Inspiring', 'Original'],
      educational: ['Learning', 'Informative', 'Practical'],
      relaxation: ['Calming', 'Peaceful', 'Meditative'],
      adventure: ['Exciting', 'Exploratory', 'Thrilling'],
      social: ['Community', 'Collaborative', 'Shared'],
      productivity: ['Efficient', 'Goal-oriented', 'Practical']
    };
    
    return [
      ...(baseTags[type as keyof typeof baseTags] || []),
      ...(categoryTags[category as keyof typeof categoryTags] || [])
    ].slice(0, 3);
  };

  const getTypeIcon = (type: string) => {
    const icons = {
      video: Video,
      music: Music,
      game: Gamepad2,
      story: Film,
      podcast: Headphones,
      interactive: Zap
    };
    const IconComponent = icons[type as keyof typeof icons] || Play;
    return <IconComponent className="w-5 h-5" />;
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      creative: 'text-purple-400 bg-purple-400/10',
      educational: 'text-blue-400 bg-blue-400/10',
      relaxation: 'text-green-400 bg-green-400/10',
      adventure: 'text-red-400 bg-red-400/10',
      social: 'text-yellow-400 bg-yellow-400/10',
      productivity: 'text-cyan-400 bg-cyan-400/10'
    };
    return colors[category as keyof typeof colors] || 'text-gray-400 bg-gray-400/10';
  };

  const filteredItems = selectedType === 'all' 
    ? entertainmentItems 
    : entertainmentItems.filter(item => item.type === selectedType);

  return (
    <div className="flex-1 flex flex-col bg-gray-900">
      {/* Enhanced Header */}
      <div className="p-6 border-b border-gray-700/50 bg-gradient-to-r from-gray-800/50 to-gray-900/50">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Pollen Entertainment Hub</h1>
            <p className="text-gray-400">AI-generated • Interactive content • Personalized experiences</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingContent && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Creating content...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{filteredItems.length}</div>
              <div className="text-xs text-gray-400">Available experiences</div>
            </div>
          </div>
        </div>

        {/* Content Type Selector */}
        <div className="flex space-x-3 overflow-x-auto pb-2">
          {contentTypes.map((type) => (
            <button
              key={type}
              onClick={() => setSelectedType(type)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg whitespace-nowrap transition-all ${
                selectedType === type
                  ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300'
                  : 'bg-gray-700/50 border border-gray-600/50 text-gray-300 hover:bg-gray-600/50'
              }`}
            >
              {type !== 'all' && getTypeIcon(type)}
              <span className="text-sm font-medium capitalize">{type}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredItems.map((item) => (
            <div key={item.id} className="bg-gradient-to-br from-gray-800/60 to-gray-900/60 rounded-xl border border-gray-700/50 p-6 hover:border-gray-600/50 transition-all duration-300 backdrop-blur-sm group cursor-pointer">
              {/* Item Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg flex items-center justify-center text-white">
                    {getTypeIcon(item.type)}
                  </div>
                  <div>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(item.category)}`}>
                      {item.category}
                    </div>
                  </div>
                </div>
                {item.downloadable && (
                  <button className="p-2 bg-gray-700/50 rounded-lg hover:bg-gray-600/50 transition-colors">
                    <Download className="w-4 h-4 text-white" />
                  </button>
                )}
              </div>

              {/* Item Content */}
              <h3 className="text-lg font-bold text-white mb-3 leading-tight group-hover:text-cyan-300 transition-colors">
                {item.title}
              </h3>
              
              <p className="text-gray-300 text-sm leading-relaxed mb-4 line-clamp-3">
                {item.description}
              </p>

              {/* Tags */}
              <div className="flex flex-wrap gap-2 mb-4">
                {item.tags.map((tag) => (
                  <span key={tag} className="px-2 py-1 bg-gray-700/50 text-gray-300 text-xs rounded">
                    {tag}
                  </span>
                ))}
              </div>

              {/* Item Footer */}
              <div className="flex items-center justify-between pt-4 border-t border-gray-700/50">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm font-medium text-white">{item.rating}</span>
                  </div>
                  {item.duration && (
                    <span className="text-sm text-gray-400">{item.duration}</span>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`text-xs px-2 py-1 rounded ${
                    item.difficulty === 'beginner' ? 'bg-green-500/20 text-green-400' :
                    item.difficulty === 'intermediate' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {item.difficulty}
                  </span>
                  <button className="flex items-center space-x-1 px-3 py-1 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg text-white text-sm font-medium hover:from-cyan-600 hover:to-purple-600 transition-all">
                    <Play className="w-3 h-3" />
                    <span>Experience</span>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredItems.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-8">
              <Sparkles className="w-12 h-12 text-white animate-pulse" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">Creating Entertainment Experiences...</h3>
            <p className="text-gray-400 max-w-lg mx-auto text-lg">
              Pollen is generating personalized entertainment content across multiple formats to enhance creativity, learning, and enjoyment.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
