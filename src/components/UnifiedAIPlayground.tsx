import React, { useState, useEffect, useCallback } from 'react';
import { 
  Brain, 
  Gamepad2, 
  Music, 
  Megaphone, 
  Building, 
  Wand2, 
  Zap, 
  Sparkles,
  Play,
  Download,
  Share,
  ExternalLink,
  Loader2,
  Film,
  GraduationCap,
  Settings,
  Code,
  Copy,
  Star
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { contentOrchestrator } from '../services/contentOrchestrator';
import { pollenAI } from '../services/pollenAI';

// TypeScript interfaces for prompt cards
interface PromptCard {
  id: string;
  title: string;
  description: string;
  prompt: string;
  category: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  type: 'text' | 'code' | 'creative' | 'analysis';
  tags: string[];
  estimatedTime?: string;
}

interface AICategory {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  color: string;
  description: string;
  tools: string[];
  promptCards: PromptCard[];
}

// Enhanced AI Playground Categories with Entertainment, Learning, and Task Automation
const aiCategories: AICategory[] = [
  {
    id: 'content',
    name: 'Content Generation',
    icon: Wand2,
    color: 'from-purple-500 to-pink-500',
    description: 'Generate articles, posts, stories, and creative writing',
    tools: ['Text Generation', 'Story Writing', 'Blog Posts', 'Social Media'],
    promptCards: [
      {
        id: 'blog-post',
        title: 'Tech Blog Post',
        description: 'Write a comprehensive blog post about emerging technology',
        prompt: 'Write a 1000-word blog post about the impact of AI on sustainable technology, including current trends, future possibilities, and practical applications.',
        category: 'content',
        difficulty: 'intermediate',
        type: 'text',
        tags: ['blog', 'technology', 'AI'],
        estimatedTime: '5-10 min'
      },
      {
        id: 'social-campaign',
        title: 'Social Media Campaign',
        description: 'Create engaging social media content series',
        prompt: 'Create a 7-day social media campaign for a eco-friendly startup, including posts, hashtags, and engagement strategies for Instagram, Twitter, and LinkedIn.',
        category: 'content',
        difficulty: 'advanced',
        type: 'creative',
        tags: ['social media', 'marketing', 'campaign'],
        estimatedTime: '10-15 min'
      }
    ]
  },
  {
    id: 'entertainment',
    name: 'Entertainment Studio',
    icon: Film,
    color: 'from-red-500 to-pink-500',
    description: 'Create movies, series, documentaries, and entertainment content',
    tools: ['Movie Scripts', 'Series Concepts', 'Character Development', 'Plot Generation'],
    promptCards: [
      {
        id: 'movie-concept',
        title: 'Sci-Fi Movie Concept',
        description: 'Generate a complete movie concept with plot and characters',
        prompt: 'Create a detailed concept for a sci-fi thriller movie set in 2045, including main characters, plot outline, key scenes, and target audience. Focus on AI ethics and human consciousness themes.',
        category: 'entertainment',
        difficulty: 'advanced',
        type: 'creative',
        tags: ['movie', 'sci-fi', 'plot'],
        estimatedTime: '15-20 min'
      },
      {
        id: 'series-pilot',
        title: 'Series Pilot Episode',
        description: 'Write a pilot episode for a new TV series',
        prompt: 'Write a pilot episode script for a cyberpunk drama series about hackers fighting corporate surveillance. Include dialogue, scene descriptions, and character introductions.',
        category: 'entertainment',
        difficulty: 'advanced',
        type: 'text',
        tags: ['series', 'script', 'cyberpunk'],
        estimatedTime: '20-30 min'
      }
    ]
  },
  {
    id: 'learning',
    name: 'Learning Center',
    icon: GraduationCap,
    color: 'from-blue-500 to-indigo-500',
    description: 'Create educational content, courses, and learning materials',
    tools: ['Course Creation', 'Tutorial Writing', 'Lesson Plans', 'Educational Content'],
    promptCards: [
      {
        id: 'tutorial-guide',
        title: 'Programming Tutorial',
        description: 'Create a comprehensive programming tutorial',
        prompt: 'Create a beginner-friendly tutorial on React hooks, including code examples, common pitfalls, best practices, and hands-on exercises. Structure it as a step-by-step guide.',
        category: 'learning',
        difficulty: 'intermediate',
        type: 'code',
        tags: ['programming', 'React', 'tutorial'],
        estimatedTime: '15-25 min'
      },
      {
        id: 'course-outline',
        title: 'Online Course Structure',
        description: 'Design a complete course curriculum',
        prompt: 'Design a 8-week online course on "Digital Marketing for Startups" including weekly modules, learning objectives, assignments, and assessment methods.',
        category: 'learning',
        difficulty: 'advanced',
        type: 'analysis',
        tags: ['course', 'marketing', 'curriculum'],
        estimatedTime: '20-30 min'
      }
    ]
  },
  {
    id: 'automation',
    name: 'Task Automation',
    icon: Zap,
    color: 'from-orange-500 to-yellow-500',
    description: 'Automate workflows, create scripts, and optimize processes',
    tools: ['Workflow Design', 'Script Generation', 'Process Optimization', 'Integration Setup'],
    promptCards: [
      {
        id: 'python-automation',
        title: 'Python Automation Script',
        description: 'Generate Python scripts for task automation',
        prompt: 'Create a Python script that automatically organizes files in a directory by date and file type, includes error handling, logging, and can be scheduled to run daily.',
        category: 'automation',
        difficulty: 'intermediate',
        type: 'code',
        tags: ['python', 'automation', 'files'],
        estimatedTime: '10-15 min'
      },
      {
        id: 'workflow-design',
        title: 'Business Workflow',
        description: 'Design automated business processes',
        prompt: 'Design an automated customer onboarding workflow for a SaaS company, including email sequences, task assignments, data collection, and integration touchpoints.',
        category: 'automation',
        difficulty: 'advanced',
        type: 'analysis',
        tags: ['workflow', 'business', 'automation'],
        estimatedTime: '15-20 min'
      }
    ]
  },
  {
    id: 'music',
    name: 'Music Creation',
    icon: Music,
    color: 'from-blue-500 to-cyan-500',
    description: 'Compose melodies, create beats, and generate audio',
    tools: ['Beat Generation', 'Melody Composition', 'Audio Effects', 'Sound Design'],
    promptCards: [
      {
        id: 'electronic-track',
        title: 'Electronic Music Track',
        description: 'Create structure for an electronic music composition',
        prompt: 'Design a 4-minute electronic music track with detailed structure including intro, build-up, drop, breakdown, and outro. Include BPM, key signature, and instrument suggestions.',
        category: 'music',
        difficulty: 'intermediate',
        type: 'creative',
        tags: ['electronic', 'composition', 'structure'],
        estimatedTime: '10-15 min'
      }
    ]
  },
  {
    id: 'games',
    name: 'Game Development',
    icon: Gamepad2,
    color: 'from-green-500 to-emerald-500',
    description: 'Create game concepts, mechanics, and interactive experiences',
    tools: ['Game Mechanics', 'Character Creation', 'World Building', 'Storylines'],
    promptCards: [
      {
        id: 'game-concept',
        title: 'Indie Game Concept',
        description: 'Develop a complete indie game concept',
        prompt: 'Create a detailed concept for an indie puzzle-platformer game, including core mechanics, art style, target audience, level design principles, and monetization strategy.',
        category: 'games',
        difficulty: 'advanced',
        type: 'creative',
        tags: ['game design', 'indie', 'puzzle'],
        estimatedTime: '20-30 min'
      }
    ]
  },
  {
    id: 'ads',
    name: 'Marketing & Ads',
    icon: Megaphone,
    color: 'from-orange-500 to-red-500',
    description: 'Design campaigns, create copy, and generate marketing materials',
    tools: ['Ad Copy', 'Campaign Strategy', 'Visual Concepts', 'Brand Messaging'],
    promptCards: [
      {
        id: 'ad-campaign',
        title: 'Digital Ad Campaign',
        description: 'Create a comprehensive advertising campaign',
        prompt: 'Develop a multi-platform digital advertising campaign for a new fitness app, including ad copy for Facebook, Google Ads, and Instagram, with audience targeting and budget allocation.',
        category: 'ads',
        difficulty: 'advanced',
        type: 'creative',
        tags: ['advertising', 'campaign', 'fitness'],
        estimatedTime: '15-25 min'
      }
    ]
  },
  {
    id: 'industries',
    name: 'Industry Solutions',
    icon: Building,
    color: 'from-indigo-500 to-purple-500',
    description: 'Professional tools for various business sectors',
    tools: ['Business Analysis', 'Industry Reports', 'Process Automation', 'Data Insights'],
    promptCards: [
      {
        id: 'market-analysis',
        title: 'Market Analysis Report',
        description: 'Generate comprehensive market analysis',
        prompt: 'Create a detailed market analysis report for the electric vehicle charging infrastructure industry, including market size, key players, trends, opportunities, and 5-year forecast.',
        category: 'industries',
        difficulty: 'advanced',
        type: 'analysis',
        tags: ['market analysis', 'EV', 'business'],
        estimatedTime: '25-35 min'
      }
    ]
  },
  {
    id: 'creative',
    name: 'Creative Studio',
    icon: Sparkles,
    color: 'from-pink-500 to-rose-500',
    description: 'Unleash creativity with AI-powered artistic tools',
    tools: ['Image Concepts', 'Design Ideas', 'Creative Writing', 'Art Direction'],
    promptCards: [
      {
        id: 'brand-identity',
        title: 'Brand Identity Design',
        description: 'Create complete brand identity concepts',
        prompt: 'Design a complete brand identity for a sustainable fashion startup, including logo concepts, color palette, typography, brand voice, and visual style guidelines.',
        category: 'creative',
        difficulty: 'advanced',
        type: 'creative',
        tags: ['branding', 'design', 'sustainable'],
        estimatedTime: '20-30 min'
      }
    ]
  }
];

// Difficulty color mapping
const difficultyColors = {
  beginner: 'bg-green-500/20 text-green-300 border-green-500/30',
  intermediate: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  advanced: 'bg-red-500/20 text-red-300 border-red-500/30'
};

// Type icon mapping
const typeIcons = {
  text: Wand2,
  code: Code,
  creative: Sparkles,
  analysis: Brain
};

export function UnifiedAIPlayground() {
  const [activeCategory, setActiveCategory] = useState('content');
  const [prompt, setPrompt] = useState('');
  const [selectedCard, setSelectedCard] = useState<PromptCard | null>(null);
  const [generatedContent, setGeneratedContent] = useState<any[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<any[]>([]);
  const [customPrompt, setCustomPrompt] = useState('');

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);
    try {
      // Use Pollen AI directly for better quality and reasoning
      const pollenResponse = await pollenAI.generate(prompt, activeCategory);
      
      // Also get enhanced content through orchestrator for diversity
      const { content } = await contentOrchestrator.generateContent({
        type: activeCategory as any,
        count: 2,
        query: prompt,
        strategy: {
          diversity: 0.95,
          freshness: 1.0,
          personalization: 0.8,
          qualityThreshold: 8.0,
          trendingBoost: 1.5
        },
        realtime: true
      });

      // Create social-style content entries
      const pollenContent = {
        id: `pollen-${Date.now()}`,
        category: activeCategory,
        prompt,
        content: {
          title: `Pollen AI: ${activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)}`,
          description: pollenResponse.content,
          reasoning: pollenResponse.reasoning,
          confidence: pollenResponse.confidence,
          source: 'pollen-ai'
        },
        timestamp: new Date().toISOString(),
        type: 'pollen-ai',
        likes: Math.floor(Math.random() * 50) + 10,
        shares: Math.floor(Math.random() * 20) + 5,
        user: {
          name: 'Pollen AI',
          username: 'pollen_ai',
          avatar: 'bg-gradient-to-r from-cyan-500 to-blue-500',
          verified: true
        }
      };

      const enhancedContent = content.map((item: any, index: number) => ({
        id: `enhanced-${Date.now()}-${index}`,
        category: activeCategory,
        prompt,
        content: item,
        timestamp: new Date().toISOString(),
        type: 'enhanced',
        likes: Math.floor(Math.random() * 30) + 5,
        shares: Math.floor(Math.random() * 10) + 2,
        user: {
          name: 'AI Assistant',
          username: 'ai_assistant',
          avatar: 'bg-gradient-to-r from-purple-500 to-pink-500',
          verified: true
        }
      }));

      const newContent = [pollenContent, ...enhancedContent];
      setGeneratedContent(newContent);
      setHistory(prev => [...newContent, ...prev].slice(0, 50));
    } catch (error) {
      console.error('Generation failed:', error);
    }
    setIsGenerating(false);
  }, [activeCategory, prompt]);

  const usePromptCard = (card: PromptCard) => {
    setSelectedCard(card);
    setPrompt(card.prompt);
  };

  const useCustomPrompt = () => {
    setSelectedCard(null);
    setPrompt(customPrompt);
  };

  const getCurrentCategory = () => aiCategories.find(cat => cat.id === activeCategory);

  return (
    <div className="flex-1 bg-black min-h-0 flex flex-col">
      {/* Header - Pure black design with social feel */}
      <div className="bg-black border-b border-gray-900/50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-2xl border border-cyan-500/30">
                <Brain className="w-8 h-8 text-cyan-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                  AI Playground 
                  <span className="text-sm bg-cyan-500/10 text-cyan-400 px-3 py-1 rounded-full border border-cyan-500/20">
                    Live
                  </span>
                </h1>
                <p className="text-gray-400">Create, explore, and share with Pollen AI • Real-time generation • Social creativity</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-cyan-400 bg-cyan-500/10 px-3 py-2 rounded-lg border border-cyan-500/30">
                <Zap className="w-4 h-4 animate-pulse" />
                <span>Pollen AI • Live</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-green-400 bg-green-500/10 px-3 py-2 rounded-lg border border-green-500/30">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span>{generatedContent.length} creations</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 max-w-7xl mx-auto p-6 w-full">
        <Tabs value={activeCategory} onValueChange={setActiveCategory} className="w-full">
          {/* Category Navigation - Pure black with social feel */}
          <TabsList className="grid w-full grid-cols-3 lg:grid-cols-9 mb-8 bg-black border border-gray-900/50">
            {aiCategories.map((category) => (
              <TabsTrigger
                key={category.id}
                value={category.id}
                className="flex flex-col items-center space-y-1 p-3 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500/20 data-[state=active]:to-pink-500/20 min-h-[80px]"
              >
                <category.icon className="w-5 h-5" />
                <span className="text-xs font-medium text-center leading-tight">{category.name}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          {aiCategories.map((category) => (
            <TabsContent key={category.id} value={category.id} className="space-y-6">
              {/* Category Header - Black design */}
              <div className="bg-gray-900/90 rounded-xl p-6 border border-gray-800/50">
                <div className="flex items-center space-x-4 mb-4">
                  <div className={`p-3 bg-gradient-to-br ${category.color} bg-opacity-20 rounded-lg border border-gray-700/30`}>
                    <category.icon className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-white">{category.name}</h2>
                    <p className="text-gray-400">{category.description}</p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {category.tools.map((tool, index) => (
                    <Badge
                      key={index}
                      variant="outline"
                      className="bg-gray-800/50 text-gray-300 border-gray-700/50"
                    >
                      {tool}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* TypeScript Prompt Cards */}
              <div className="space-y-4">
                <h3 className="text-xl font-bold text-white">Featured Prompt Cards</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {category.promptCards.map((card) => {
                    const TypeIcon = typeIcons[card.type];
                    return (
                      <div
                        key={card.id}
                        className={`bg-gray-900/90 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/95 transition-all cursor-pointer group ${
                          selectedCard?.id === card.id ? 'ring-2 ring-purple-500/50' : ''
                        }`}
                        onClick={() => usePromptCard(card)}
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <div className="p-2 bg-gray-800/50 rounded-lg border border-gray-700/30">
                              <TypeIcon className="w-4 h-4 text-gray-300" />
                            </div>
                            <div>
                              <h4 className="text-lg font-semibold text-white group-hover:text-purple-300 transition-colors">
                                {card.title}
                              </h4>
                              <div className="flex items-center space-x-2 mt-1">
                                <Badge className={difficultyColors[card.difficulty]}>
                                  {card.difficulty}
                                </Badge>
                                {card.estimatedTime && (
                                  <span className="text-xs text-gray-500">{card.estimatedTime}</span>
                                )}
                              </div>
                            </div>
                          </div>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={(e) => {
                              e.stopPropagation();
                              navigator.clipboard.writeText(card.prompt);
                            }}
                          >
                            <Copy className="w-4 h-4" />
                          </Button>
                        </div>
                        
                        <p className="text-gray-400 text-sm mb-4 line-clamp-2">{card.description}</p>
                        
                        <div className="flex flex-wrap gap-1">
                          {card.tags.map((tag, tagIndex) => (
                            <span
                              key={tagIndex}
                              className="px-2 py-1 bg-gray-800/30 text-gray-500 rounded text-xs border border-gray-700/30"
                            >
                              #{tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Generation Interface - Enhanced */}
              <div className="bg-gray-900/90 rounded-xl border border-gray-800/50 p-6 space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-lg font-semibold text-white">Create with AI</label>
                    {selectedCard && (
                      <div className="flex items-center space-x-2 text-sm text-purple-400">
                        <Star className="w-4 h-4" />
                        <span>Using: {selectedCard.title}</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="space-y-3">
                    <Textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder={selectedCard ? selectedCard.prompt : `Describe your ${category.name.toLowerCase()} idea...`}
                      className="w-full min-h-[120px] bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 focus:border-purple-500 resize-y"
                      rows={6}
                    />
                    
                    <div className="flex items-center justify-between">
                      <div className="flex space-x-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setSelectedCard(null);
                            setPrompt('');
                          }}
                          className="border-gray-700 text-gray-300"
                        >
                          Clear
                        </Button>
                        {selectedCard && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setPrompt(selectedCard.prompt)}
                            className="border-gray-700 text-gray-300"
                          >
                            Reset to Original
                          </Button>
                        )}
                      </div>
                      <span className="text-xs text-gray-500">
                        {prompt.length} characters
                      </span>
                    </div>
                  </div>

                  <Button
                    onClick={handleGenerate}
                    disabled={isGenerating || !prompt.trim()}
                    className="w-full h-12 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold text-lg"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Wand2 className="w-5 h-5 mr-2" />
                        Generate with AI
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* Generated Content - Enhanced */}
              {generatedContent.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                    <Sparkles className="w-5 h-5 text-purple-400" />
                    <span>Generated Content</span>
                  </h3>
                  <div className="grid gap-4">
                    {generatedContent.map((item, index) => (
                      <div key={item.id} className="bg-gray-900/90 rounded-xl border border-gray-800/50 p-6">
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 bg-gradient-to-br ${category.color} bg-opacity-20 rounded-lg border border-gray-700/30`}>
                              <category.icon className="w-4 h-4 text-white" />
                            </div>
                            <div>
                              <span className="text-sm font-medium text-white">
                                Result {index + 1}
                              </span>
                              {selectedCard && (
                                <span className="text-xs text-gray-400 block">
                                  From: {selectedCard.title}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex space-x-2">
                            <Button size="sm" variant="outline" className="border-gray-700">
                              <Share className="w-4 h-4 mr-2" />
                              Share
                            </Button>
                            <Button size="sm" variant="outline" className="border-gray-700">
                              <Download className="w-4 h-4 mr-2" />
                              Export
                            </Button>
                          </div>
                        </div>
                        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                          <pre className="whitespace-pre-wrap text-gray-300 text-sm font-mono">
                            {typeof item.content === 'string' 
                              ? item.content 
                              : JSON.stringify(item.content, null, 2)
                            }
                          </pre>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>
          ))}
        </Tabs>

        {/* History Section */}
        {history.length > 0 && (
          <div className="mt-8 space-y-4">
            <h3 className="text-xl font-bold text-white flex items-center space-x-2">
              <Brain className="w-5 h-5 text-purple-400" />
              <span>Recent Generations</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {history.slice(0, 6).map((item, index) => {
                const category = aiCategories.find(cat => cat.id === item.category);
                return (
                  <div key={item.id} className="bg-gray-900/90 rounded-lg border border-gray-800/50 p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      {category && <category.icon className="w-4 h-4 text-gray-400" />}
                      <span className="text-sm text-gray-400">{category?.name}</span>
                    </div>
                    <p className="text-white text-sm font-medium mb-2 line-clamp-2">
                      {item.prompt.slice(0, 100)}...
                    </p>
                    <p className="text-gray-500 text-xs">
                      {new Date(item.timestamp).toLocaleString()}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}