
import React, { useState, useEffect, useCallback } from 'react';
import { BookOpen, Film, Music, Headphones, Gamepad2, Sparkles, Crown } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';
import { ContentCard } from './entertainment/ContentCard';
import { ContentViewer } from './entertainment/ContentViewer';
import { CustomContentGenerator } from './entertainment/CustomContentGenerator';
import { rankItems } from '../services/generalRanker';
import { SignificanceBadge } from "./entertainment/SignificanceBadge";
import { TrendingBadge } from "./entertainment/TrendingBadge";

interface EntertainmentHubProps {
  isGenerating?: boolean;
}

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

export const EntertainmentHub = ({ isGenerating = false }: EntertainmentHubProps) => {
  const [content, setContent] = useState<ContentItem[]>([]);
  const [selectedContent, setSelectedContent] = useState<ContentItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState("significance");
  const [userPrompt, setUserPrompt] = useState('');
  const [generatingCustom, setGeneratingCustom] = useState(false);

  const contentTypes = [
    { type: 'story', name: 'Interactive Stories', icon: BookOpen, color: 'text-blue-400' },
    { type: 'music', name: 'AI Music', icon: Music, color: 'text-purple-400' },
    { type: 'video', name: 'AI Videos', icon: Film, color: 'text-red-400' },
    { type: 'audio', name: 'Podcasts', icon: Headphones, color: 'text-green-400' },
    { type: 'game', name: 'Mini Games', icon: Gamepad2, color: 'text-yellow-400' },
    { type: 'interactive', name: 'Interactive', icon: Sparkles, color: 'text-cyan-400' }
  ];

  const generateContent = useCallback(async () => {
    const templates = [
      {
        type: 'story' as const,
        title: 'The Quantum Garden',
        description: 'An interactive narrative where your choices reshape reality across multiple dimensions.',
        content: `You stand at the edge of the Quantum Garden, where every decision creates a new timeline...\n\nChapter 1: The Threshold\nThe air shimmers with possibility as you approach the crystalline gates. Three paths stretch before you, each leading to a different version of reality. The garden responds to your thoughts, blooming with colors that shouldn't exist.\n\nChoice 1: Follow the silver path that whispers of technological wonders\nChoice 2: Take the golden route that hums with ancient wisdom  \nChoice 3: Step onto the prismatic walkway that shifts between both\n\nYour choice will determine not just your story, but the very nature of the reality you'll explore. The garden awaits your decision...\n\n[Interactive elements respond to your selections, creating a unique narrative experience]`,
        category: 'Sci-Fi Interactive',
        tags: ['interactive', 'sci-fi', 'choice-driven', 'trending'],
        duration: '15-30 min',
        difficulty: 'Medium'
      },
      {
        type: 'music' as const,
        title: 'Neural Symphony #47',
        description: 'AI-composed ambient music that adapts to your current mood and environment.',
        content: `🎵 Now Playing: Neural Symphony #47 - "Digital Dreams"\n\nThis piece was generated using advanced neural networks trained on thousands of classical and contemporary compositions. The AI analyzed patterns in harmony, rhythm, and emotional resonance to create something entirely new.\n\nKey Features:\n- Adaptive tempo based on time of day\n- Harmonies that respond to listener feedback\n- Seamless 30-minute loop with no repetition\n- Binaural beats for enhanced focus\n\nThe composition begins with ethereal pads, slowly building into a complex tapestry of synthesized orchestral elements. Each listen reveals new layers and subtle variations.\n\n[Audio player interface would be embedded here with visualization]`,
        category: 'AI Music',
        tags: ['ambient', 'ai-generated', 'adaptive', 'focus'],
        duration: '30 min'
      },
      {
        type: 'video' as const,
        title: 'Future Cities Visualization',
        description: 'AI-generated exploration of sustainable urban environments in 2050.',
        content: `🎬 AI-Generated Video: "Cities of Tomorrow"\n\nThis visualization was created using advanced AI models trained on architectural data, urban planning principles, and environmental science research.\n\nVideo Description:\nA journey through hypothetical cities of 2050, showcasing:\n- Vertical farms integrated into skyscrapers\n- Autonomous transportation networks\n- Carbon-negative building materials\n- Community spaces designed for human connection\n\nThe AI considered factors like population density, climate adaptation, renewable energy integration, and social equity to generate these urban visions.\n\n[Video player interface would be embedded here with 4K playback]\n\nTechnical Details:\n- Generated using neural rendering techniques\n- 4K resolution with realistic lighting\n- Physics-based environmental simulations\n- Culturally diverse architectural styles`,
        category: 'Future Concepts',
        tags: ['ai-video', 'futurism', 'sustainability', 'viral'],
        duration: '12 min'
      }
    ];

    const shuffledTemplates = [...templates].sort(() => Math.random() - 0.5);
    
    const contentItems = shuffledTemplates.map((template, index) => {
      const scored = significanceAlgorithm.scoreContent(template.content, 'entertainment', 'Pollen Entertainment');
      
      return {
        id: (Date.now() + index).toString(),
        title: template.title,
        description: template.description,
        type: template.type,
        content: template.content,
        duration: template.duration,
        category: template.category,
        tags: template.tags,
        significance: scored.significanceScore,
        trending: scored.significanceScore > 7.5 || template.tags.includes('trending') || template.tags.includes('viral'),
        views: Math.floor(Math.random() * 100000) + 5000,
        likes: Math.floor(Math.random() * 10000) + 50,
        shares: Math.floor(Math.random() * 5000) + 100,
        comments: Math.floor(Math.random() * 2000) + 50,
        rating: Math.round((Math.random() * 2 + 3) * 10) / 10,
        difficulty: template.difficulty
      };
    });

    return contentItems;
  }, []);

  const loadContent = useCallback(async () => {
    setLoading(true);
    try {
      const newContent = await generateContent();
      setContent(newContent.sort((a, b) => b.significance - a.significance));
    } catch (error) {
      console.error('Error loading content:', error);
    } finally {
      setLoading(false);
    }
  }, [generateContent]);

  useEffect(() => {
    loadContent();
    const interval = setInterval(loadContent, 120000);
    return () => clearInterval(interval);
  }, [loadContent]);

  const generateCustomContent = async () => {
    if (!userPrompt.trim() || generatingCustom) return;
    
    setGeneratingCustom(true);
    
    try {
      const response = await pollenAI.generate(userPrompt, 'entertainment', true);
      
      const scored = significanceAlgorithm.scoreContent(response.content, 'entertainment', 'Pollen AI');
      
      const customContent: ContentItem = {
        id: Date.now().toString(),
        title: `Custom: ${userPrompt.slice(0, 50)}${userPrompt.length > 50 ? '...' : ''}`,
        description: `AI-generated content based on your prompt: "${userPrompt}"`,
        type: 'interactive',
        content: response.content,
        duration: 'Variable',
        category: 'Custom Generated',
        tags: ['custom', 'ai-generated', 'prompt-based', 'new'],
        significance: scored.significanceScore,
        trending: true,
        views: 0,
        likes: 0,
        shares: 0,
        comments: 0,
        rating: 4.5
      };
      
      setContent(prev => [customContent, ...prev]);
      setUserPrompt('');
    } catch (error) {
      console.error('Error generating custom content:', error);
    } finally {
      setGeneratingCustom(false);
    }
  };

  // Filter + sort logic
  const filteredContent = content.filter(item => {
    if (filter === 'trending') return item.trending;
    if (filter === 'all') return true;
    return item.type === filter;
  });
  // Unified ranker logic
  const sortedContent = rankItems(filteredContent, { type: "entertainment", sortBy });

  if (selectedContent) {
    return (
      <ContentViewer 
        content={selectedContent} 
        onBack={() => setSelectedContent(null)} 
      />
    );
  }

  return (
    <div className="flex-1 bg-gradient-to-br from-slate-950 via-gray-950 to-blue-1000 min-h-screen transition-all">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2 font-playfair drop-shadow">Entertainment Hub</h1>
              <p className="text-gray-400">AI-generated content • Interactive experiences • Personalized entertainment</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Generating</span>
              </div>
              <div className="px-4 py-2 bg-purple-500/10 text-purple-300 rounded-full text-sm font-medium border border-purple-500/20 flex items-center space-x-2">
                <Crown className="w-4 h-4" />
                <span>Premium AI</span>
              </div>
            </div>
          </div>

          {/* Custom Content Generator */}
          <CustomContentGenerator
            userPrompt={userPrompt}
            setUserPrompt={setUserPrompt}
            onGenerate={generateCustomContent}
            isGenerating={generatingCustom}
          />

          {/* Filter Tabs */}
          <div className="flex items-center mt-6 overflow-x-auto gap-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                filter === 'all'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              All Content
            </button>
            <button
              onClick={() => setFilter('trending')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                filter === 'trending'
                  ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                  : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              Trending
            </button>
            {contentTypes.map(({ type, name, icon: Icon, color }) => (
              <button
                key={type}
                onClick={() => setFilter(type)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap flex items-center space-x-2 ${
                  filter === type
                    ? 'bg-gray-700/50 text-white border border-gray-600/50'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${color}`} />
                <span>{name}</span>
              </button>
            ))}
            {/* Sort Dropdown */}
            <select
              value={sortBy}
              onChange={e => setSortBy(e.target.value)}
              className="ml-auto bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:border-cyan-500/50 shadow"
              style={{ minWidth: 140 }}
            >
              <option value="significance">Sort by Significance</option>
              <option value="rating">Sort by Rating</option>
              <option value="views">Sort by Views</option>
              <option value="trending">Sort by Trending</option>
            </select>
          </div>
        </div>
      </div>

      {/* Content Grid */}
      <div className="p-6">
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-gray-900/50 rounded-xl p-6 animate-pulse shadow-lg">
                <div className="h-4 bg-gray-700/50 rounded mb-4"></div>
                <div className="h-3 bg-gray-800/50 rounded mb-2"></div>
                <div className="h-3 bg-gray-800/50 rounded w-2/3"></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sortedContent.map((item) => (
              <div
                key={item.id}
                className="bg-gradient-to-br from-gray-900/60 via-gray-900/50 to-blue-950/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/80 transition-all cursor-pointer group shadow-lg relative animate-fade-in"
                onClick={() => setSelectedContent(item)}
                role="button"
                tabIndex={0}
              >
                {/* NEW BADGES Top-right */}
                <div className="absolute top-4 right-4 flex gap-2 z-10">
                  <TrendingBadge isTrending={item.trending} />
                  <SignificanceBadge score={item.significance} />
                </div>
                {/* Info */}
                <div className="flex items-center mb-2 gap-2">
                  <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
                    {item.category}
                  </span>
                  <span className="px-2 py-1 bg-gray-700/50 text-gray-300 rounded text-xs">{item.duration}</span>
                  {item.difficulty && (
                    <span className="px-2 py-1 bg-blue-900/20 text-blue-300 rounded text-xs border border-blue-800/30">
                      {item.difficulty}
                    </span>
                  )}
                </div>
                <h3 className="text-lg font-bold text-white mb-2 group-hover:text-cyan-300 transition-colors truncate font-playfair">{item.title}</h3>
                <p className="text-gray-400 text-sm mb-3 line-clamp-2">{item.description}</p>
                {/* Meta */}
                <div className="mt-4 flex justify-between text-xs text-gray-500">
                  <span>👁 {item.views.toLocaleString()} views</span>
                  <span>⭐ {item.rating.toFixed(1)} rating</span>
                  <span>💬 {item.comments} comments</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && filteredContent.length === 0 && (
          <div className="text-center py-12 animate-fade-in">
            <Sparkles className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg">No content found for this filter</p>
            <p className="text-gray-500 text-sm mt-2">Try selecting a different category or generate custom content</p>
          </div>
        )}
      </div>
    </div>
  );
};
