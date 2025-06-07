
import React, { useState, useEffect, useCallback } from 'react';
import { Play, Headphones, BookOpen, Film, Music, Gamepad2, Sparkles, TrendingUp, Award, Send, Mic } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

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
  thumbnail?: string;
}

export const EntertainmentHub = ({ isGenerating = false }: EntertainmentHubProps) => {
  const [content, setContent] = useState<ContentItem[]>([]);
  const [selectedContent, setSelectedContent] = useState<ContentItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
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
        content: `You stand at the edge of the Quantum Garden, where every decision creates a new timeline...

Chapter 1: The Threshold
The air shimmers with possibility as you approach the crystalline gates. Three paths stretch before you, each leading to a different version of reality. The garden responds to your thoughts, blooming with colors that shouldn't exist.

Choice 1: Follow the silver path that whispers of technological wonders
Choice 2: Take the golden route that hums with ancient wisdom  
Choice 3: Step onto the prismatic walkway that shifts between both

Your choice will determine not just your story, but the very nature of the reality you'll explore. The garden awaits your decision...

[Interactive elements respond to your selections, creating a unique narrative experience]`,
        category: 'Sci-Fi Interactive',
        tags: ['interactive', 'sci-fi', 'choice-driven'],
        duration: '15-30 min'
      },
      {
        type: 'music' as const,
        title: 'Neural Symphony #47',
        description: 'AI-composed ambient music that adapts to your current mood and environment.',
        content: `🎵 Now Playing: Neural Symphony #47 - "Digital Dreams"

This piece was generated using advanced neural networks trained on thousands of classical and contemporary compositions. The AI analyzed patterns in harmony, rhythm, and emotional resonance to create something entirely new.

Key Features:
- Adaptive tempo based on time of day
- Harmonies that respond to listener feedback
- Seamless 30-minute loop with no repetition
- Binaural beats for enhanced focus

The composition begins with ethereal pads, slowly building into a complex tapestry of synthesized orchestral elements. Each listen reveals new layers and subtle variations.`,
        category: 'AI Music',
        tags: ['ambient', 'ai-generated', 'adaptive'],
        duration: '30 min'
      },
      {
        type: 'video' as const,
        title: 'Future Cities Visualization',
        description: 'AI-generated exploration of sustainable urban environments in 2050.',
        content: `🎬 AI-Generated Video: "Cities of Tomorrow"

This visualization was created using advanced AI models trained on architectural data, urban planning principles, and environmental science research.

Video Description:
A journey through hypothetical cities of 2050, showcasing:
- Vertical farms integrated into skyscrapers
- Autonomous transportation networks
- Carbon-negative building materials
- Community spaces designed for human connection

The AI considered factors like population density, climate adaptation, renewable energy integration, and social equity to generate these urban visions.

[Video would display here with AI-generated visuals and narration]

Technical Details:
- Generated using neural rendering techniques
- 4K resolution with realistic lighting
- Physics-based environmental simulations
- Culturally diverse architectural styles`,
        category: 'Future Concepts',
        tags: ['ai-video', 'futurism', 'sustainability'],
        duration: '12 min'
      },
      {
        type: 'audio' as const,
        title: 'The Innovation Podcast',
        description: 'AI-hosted discussion on breakthrough technologies and their implications.',
        content: `🎙️ AI Podcast: "Breakthrough Analysis Episode 23"

Host: Aria (AI Journalist)
Topic: Recent Advances in Quantum Computing and Biotechnology

Episode Summary:
Today's AI-generated podcast explores the intersection of quantum computing and biological systems. The AI host synthesizes recent research papers, expert interviews, and trend analysis to provide insights on:

- Quantum-enhanced drug discovery processes
- Biocomputing using living cells
- Ethical implications of quantum biology
- Timeline for practical applications

The discussion draws from 847 research papers published in the last 30 days, interviews with 23 leading scientists, and patent filings from major tech companies.

Key Insights:
1. Quantum algorithms are revolutionizing protein folding predictions
2. Bio-quantum hybrid systems show promise for sustainable computing
3. Regulatory frameworks are rapidly evolving to address new capabilities

[Audio would play here with natural AI-generated speech and discussion]`,
        category: 'Tech Analysis',
        tags: ['ai-podcast', 'technology', 'analysis'],
        duration: '18 min'
      },
      {
        type: 'game' as const,
        title: 'Pattern Recognition Challenge',
        description: 'AI-designed puzzle game that adapts to your cognitive patterns.',
        content: `🎮 AI Game: "Cognitive Resonance"

Welcome to a puzzle experience designed by artificial intelligence to challenge and enhance human cognitive abilities.

Game Mechanics:
The AI observes your problem-solving patterns and dynamically adjusts difficulty and puzzle types to maintain optimal challenge level.

Current Challenge: Level 1
Pattern Type: Spatial-Temporal Sequences

Instructions:
1. Observe the sequence of shapes and colors
2. Identify the underlying pattern
3. Predict the next element in the sequence
4. The AI will adapt based on your performance

[Interactive game interface would be displayed here]

Features:
- Personalized difficulty scaling
- Real-time performance analytics
- Cognitive skill development tracking
- Multiplayer pattern recognition competitions

The AI continuously learns from player interactions to create increasingly sophisticated and engaging challenges.`,
        category: 'Cognitive Games',
        tags: ['ai-game', 'puzzle', 'adaptive'],
        duration: '5-20 min'
      },
      {
        type: 'interactive' as const,
        title: 'Virtual AI Mentor',
        description: 'Interactive conversation with an AI that adapts to your learning style.',
        content: `🤖 Interactive AI Mentor: "Professor Synthesis"

Welcome to your personalized learning experience. I'm an AI mentor designed to adapt to your unique learning style and interests.

Current Session: Introduction to Emergent Technologies

Mentor: "Hello! I've analyzed your interaction patterns and believe you'd benefit from a visual-kinesthetic approach to learning. Shall we explore how artificial intelligence is transforming different industries?"

[Interactive conversation interface]

Available Topics:
- AI in Healthcare: Personalized medicine and diagnostics
- Sustainable Technology: Climate solutions and renewable energy
- Space Exploration: Mars colonization and asteroid mining
- Biotechnology: Gene editing and synthetic biology

The AI mentor adapts its:
- Communication style to match your preferences
- Examples to align with your interests
- Pacing based on your comprehension
- Teaching methods to optimize learning

You can ask questions, request deeper explanations, or explore tangential topics. The mentor maintains context throughout our conversation and builds on previous sessions.`,
        category: 'Educational',
        tags: ['ai-mentor', 'interactive', 'learning'],
        duration: 'Unlimited'
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
        trending: scored.significanceScore > 7.5,
        views: Math.floor(Math.random() * 100000) + 5000,
        likes: Math.floor(Math.random() * 10000) + 500
      };
    });

    return contentItems;
  }, []);

  const loadContent = useCallback(async () => {
    setLoading(true);
    const newContent = await generateContent();
    setContent(newContent.sort((a, b) => b.significance - a.significance));
    setLoading(false);
  }, [generateContent]);

  useEffect(() => {
    loadContent();
    const interval = setInterval(loadContent, 60000); // Refresh every minute
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
        title: `Custom: ${userPrompt.slice(0, 50)}...`,
        description: `AI-generated content based on your prompt: "${userPrompt}"`,
        type: 'interactive',
        content: response.content,
        duration: 'Variable',
        category: 'Custom Generated',
        tags: ['custom', 'ai-generated', 'prompt-based'],
        significance: scored.significanceScore,
        trending: true,
        views: 0,
        likes: 0
      };
      
      setContent(prev => [customContent, ...prev]);
      setUserPrompt('');
    } catch (error) {
      console.error('Error generating custom content:', error);
    } finally {
      setGeneratingCustom(false);
    }
  };

  const filteredContent = content.filter(item => {
    if (filter === 'trending') return item.trending;
    if (filter === 'all') return true;
    return item.type === filter;
  });

  if (selectedContent) {
    return (
      <div className="flex-1 bg-gray-950">
        {/* Content Header */}
        <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <button
                onClick={() => setSelectedContent(null)}
                className="flex items-center space-x-2 text-cyan-400 hover:text-cyan-300 transition-colors"
              >
                <span>← Back to Entertainment</span>
              </button>
              <div className="flex items-center space-x-3">
                <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                  selectedContent.significance > 8 
                    ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                    : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                }`}>
                  {selectedContent.significance.toFixed(1)} Quality
                </div>
                <div className="flex items-center space-x-2 text-gray-400 text-sm">
                  <span>{selectedContent.views.toLocaleString()} views</span>
                  <span>•</span>
                  <span>{selectedContent.likes.toLocaleString()} likes</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Content Display */}
        <div className="p-6 max-w-4xl mx-auto">
          <div className="mb-6">
            <div className="flex items-center space-x-4 mb-4">
              <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm border border-purple-500/30">
                {selectedContent.category}
              </span>
              <span className="text-gray-400 text-sm">{selectedContent.duration}</span>
            </div>
            <h1 className="text-4xl font-bold text-white mb-4">{selectedContent.title}</h1>
            <p className="text-xl text-gray-300 leading-relaxed">{selectedContent.description}</p>
          </div>

          <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-8">
            <div className="prose prose-invert max-w-none">
              <div className="text-gray-200 leading-relaxed whitespace-pre-line">
                {selectedContent.content}
              </div>
            </div>
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-2 mt-8 pt-6 border-t border-gray-800/50">
            {selectedContent.tags.map((tag, index) => (
              <span key={index} className="px-3 py-1 bg-gray-700/50 text-gray-300 rounded-full text-sm border border-gray-600/50">
                #{tag}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-gray-950">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Entertainment Hub</h1>
              <p className="text-gray-400">AI-generated content • Interactive experiences • Personalized entertainment</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Generating</span>
              </div>
            </div>
          </div>

          {/* Custom Content Generator */}
          <div className="bg-gray-800/50 rounded-xl border border-gray-700/50 p-4 mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">Create Custom Content</h3>
            <div className="flex space-x-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={userPrompt}
                  onChange={(e) => setUserPrompt(e.target.value)}
                  placeholder="Describe what you want to create... (e.g., 'a story about time travel' or 'relaxing music for studying')"
                  className="w-full bg-gray-700/50 border border-gray-600/50 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:border-cyan-500/50 focus:outline-none"
                  onKeyPress={(e) => e.key === 'Enter' && generateCustomContent()}
                />
              </div>
              <button
                onClick={generateCustomContent}
                disabled={!userPrompt.trim() || generatingCustom}
                className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 px-6 py-3 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {generatingCustom ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Generate</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Filters */}
          <div className="flex space-x-2 overflow-x-auto">
            <button
              onClick={() => setFilter('all')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                filter === 'all'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
              }`}
            >
              <Sparkles className="w-4 h-4" />
              <span className="text-sm font-medium">All Content</span>
            </button>
            <button
              onClick={() => setFilter('trending')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                filter === 'trending'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm font-medium">Trending</span>
            </button>
            {contentTypes.map((type) => (
              <button
                key={type.type}
                onClick={() => setFilter(type.type)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                  filter === type.type
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
                }`}
              >
                <type.icon className={`w-4 h-4 ${type.color}`} />
                <span className="text-sm font-medium">{type.name}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content Grid */}
      <div className="p-6">
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-gray-900/50 rounded-xl p-6 border border-gray-800/50 animate-pulse">
                <div className="w-full h-40 bg-gray-700 rounded-lg mb-4"></div>
                <div className="w-3/4 h-4 bg-gray-700 rounded mb-2"></div>
                <div className="w-full h-3 bg-gray-700 rounded mb-4"></div>
                <div className="flex space-x-2">
                  <div className="w-16 h-6 bg-gray-700 rounded"></div>
                  <div className="w-16 h-6 bg-gray-700 rounded"></div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredContent.map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedContent(item)}
                className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-all cursor-pointer group"
              >
                {/* Content Preview */}
                <div className="w-full h-40 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg mb-4 flex items-center justify-center group-hover:from-cyan-500/20 group-hover:to-purple-500/20 transition-all">
                  {contentTypes.find(t => t.type === item.type) && (
                    <contentTypes.find(t => t.type === item.type)!.icon className={`w-12 h-12 ${contentTypes.find(t => t.type === item.type)!.color}`} />
                  )}
                </div>

                {/* Content Info */}
                <div className="flex items-center justify-between mb-2">
                  <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
                    {item.category}
                  </span>
                  <div className={`px-2 py-1 rounded text-xs font-medium ${
                    item.significance > 8 
                      ? 'bg-red-500/20 text-red-300'
                      : 'bg-cyan-500/20 text-cyan-300'
                  }`}>
                    {item.significance.toFixed(1)}
                  </div>
                </div>

                <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-cyan-300 transition-colors">
                  {item.title}
                </h3>

                <p className="text-gray-400 text-sm mb-4 line-clamp-2">
                  {item.description}
                </p>

                {/* Meta Info */}
                <div className="flex items-center justify-between text-xs text-gray-500 mb-3">
                  <span>{item.duration}</span>
                  <div className="flex items-center space-x-3">
                    <span>{(item.views / 1000).toFixed(1)}k views</span>
                    <span>{(item.likes / 1000).toFixed(1)}k likes</span>
                  </div>
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-2">
                  {item.tags.slice(0, 2).map((tag, index) => (
                    <span key={index} className="px-2 py-1 bg-gray-600/20 text-gray-400 rounded text-xs border border-gray-600/30">
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
