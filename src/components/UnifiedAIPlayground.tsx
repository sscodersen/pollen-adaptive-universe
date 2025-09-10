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
  Loader2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { contentOrchestrator } from '../services/contentOrchestrator';

// AI Playground Categories
const aiCategories = [
  {
    id: 'content',
    name: 'Content Generation',
    icon: Wand2,
    color: 'from-purple-500 to-pink-500',
    description: 'Generate articles, posts, stories, and creative writing',
    tools: ['Text Generation', 'Story Writing', 'Blog Posts', 'Social Media']
  },
  {
    id: 'music',
    name: 'Music Creation',
    icon: Music,
    color: 'from-blue-500 to-cyan-500',
    description: 'Compose melodies, create beats, and generate audio',
    tools: ['Beat Generation', 'Melody Composition', 'Audio Effects', 'Sound Design']
  },
  {
    id: 'games',
    name: 'Game Development',
    icon: Gamepad2,
    color: 'from-green-500 to-emerald-500',
    description: 'Create game concepts, mechanics, and interactive experiences',
    tools: ['Game Mechanics', 'Character Creation', 'World Building', 'Storylines']
  },
  {
    id: 'ads',
    name: 'Marketing & Ads',
    icon: Megaphone,
    color: 'from-orange-500 to-red-500',
    description: 'Design campaigns, create copy, and generate marketing materials',
    tools: ['Ad Copy', 'Campaign Strategy', 'Visual Concepts', 'Brand Messaging']
  },
  {
    id: 'industries',
    name: 'Industry Solutions',
    icon: Building,
    color: 'from-indigo-500 to-purple-500',
    description: 'Professional tools for various business sectors',
    tools: ['Business Analysis', 'Industry Reports', 'Process Automation', 'Data Insights']
  },
  {
    id: 'creative',
    name: 'Creative Studio',
    icon: Sparkles,
    color: 'from-pink-500 to-rose-500',
    description: 'Unleash creativity with AI-powered artistic tools',
    tools: ['Image Concepts', 'Design Ideas', 'Creative Writing', 'Art Direction']
  }
];

const generatePrompts = {
  content: [
    "Write a compelling blog post about sustainable technology",
    "Create a social media campaign for eco-friendly products",
    "Generate an engaging newsletter for tech enthusiasts",
    "Write product descriptions for innovative gadgets"
  ],
  music: [
    "Create an upbeat electronic track with ambient elements",
    "Generate a lo-fi hip-hop beat for studying",
    "Compose a cinematic orchestral piece for a film scene",
    "Design sound effects for a futuristic video game"
  ],
  games: [
    "Design a puzzle game mechanic using physics",
    "Create characters for a cyberpunk adventure game",
    "Generate a storyline for an indie platformer",
    "Develop game balance for a strategy RPG"
  ],
  ads: [
    "Create a video ad campaign for a tech startup",
    "Design social media ads for seasonal promotions",
    "Generate billboard copy for urban advertising",
    "Develop influencer marketing strategies"
  ],
  industries: [
    "Analyze market trends in renewable energy",
    "Generate automation workflows for manufacturing",
    "Create financial forecasting models",
    "Design customer service optimization strategies"
  ],
  creative: [
    "Generate concepts for an art installation",
    "Create character designs for animated series",
    "Design logo concepts for creative agencies",
    "Generate mood boards for interior design"
  ]
};

export function UnifiedAIPlayground() {
  const [activeCategory, setActiveCategory] = useState('content');
  const [prompt, setPrompt] = useState('');
  const [generatedContent, setGeneratedContent] = useState<any[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<any[]>([]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);
    try {
      const { content } = await contentOrchestrator.generateContent({
        type: activeCategory as any,
        count: 3,
        query: prompt,
        strategy: {
          diversity: 0.9,
          freshness: 0.8,
          personalization: 0.7,
          qualityThreshold: 7.5,
          trendingBoost: 1.2
        }
      });

      const newContent = content.map((item: any, index: number) => ({
        id: `${Date.now()}-${index}`,
        category: activeCategory,
        prompt,
        content: item,
        timestamp: new Date().toISOString(),
        type: activeCategory
      }));

      setGeneratedContent(newContent);
      setHistory(prev => [...newContent, ...prev].slice(0, 50));
    } catch (error) {
      console.error('Generation failed:', error);
    }
    setIsGenerating(false);
  }, [activeCategory, prompt]);

  const useSamplePrompt = (samplePrompt: string) => {
    setPrompt(samplePrompt);
  };

  return (
    <div className="flex-1 bg-gray-950 min-h-0 flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 border-b border-gray-800/50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl border border-purple-500/30">
                <Brain className="w-8 h-8 text-purple-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">AI Playground</h1>
                <p className="text-gray-400">Unified creative workspace powered by Pollen AI</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-purple-400 bg-purple-500/10 px-3 py-2 rounded-lg border border-purple-500/30">
              <Zap className="w-4 h-4" />
              <span>Powered by Pollen LLMX</span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 max-w-7xl mx-auto p-6 w-full">
        <Tabs value={activeCategory} onValueChange={setActiveCategory} className="w-full">
          {/* Category Navigation */}
          <TabsList className="grid w-full grid-cols-6 mb-8 bg-gray-900/50 border border-gray-800/50">
            {aiCategories.map((category) => (
              <TabsTrigger
                key={category.id}
                value={category.id}
                className="flex flex-col items-center space-y-2 p-4 data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500/20 data-[state=active]:to-pink-500/20"
              >
                <category.icon className="w-5 h-5" />
                <span className="text-sm font-medium">{category.name}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          {aiCategories.map((category) => (
            <TabsContent key={category.id} value={category.id} className="space-y-6">
              {/* Category Header */}
              <div className={`bg-gradient-to-r ${category.color} bg-opacity-10 rounded-xl p-6 border border-gray-800/50`}>
                <div className="flex items-center space-x-4 mb-4">
                  <div className={`p-3 bg-gradient-to-br ${category.color} bg-opacity-20 rounded-lg`}>
                    <category.icon className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-white">{category.name}</h2>
                    <p className="text-gray-300">{category.description}</p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {category.tools.map((tool, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-gray-900/50 text-gray-300 rounded-lg text-sm border border-gray-700/50"
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              </div>

              {/* Generation Interface */}
              <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 space-y-6">
                <div className="space-y-4">
                  <label className="text-lg font-semibold text-white">What would you like to create?</label>
                  <div className="space-y-3">
                    <Input
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder={`Describe your ${category.name.toLowerCase()} idea...`}
                      className="w-full h-12 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 focus:border-purple-500 text-lg"
                      onKeyPress={(e) => e.key === 'Enter' && !isGenerating && handleGenerate()}
                    />
                    
                    {/* Sample Prompts */}
                    <div className="space-y-2">
                      <p className="text-sm text-gray-400">Try these examples:</p>
                      <div className="flex flex-wrap gap-2">
                        {generatePrompts[category.id as keyof typeof generatePrompts]?.map((samplePrompt, index) => (
                          <button
                            key={index}
                            onClick={() => useSamplePrompt(samplePrompt)}
                            className="px-3 py-1 bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 rounded-lg text-sm border border-gray-700/30 transition-colors"
                          >
                            {samplePrompt}
                          </button>
                        ))}
                      </div>
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

              {/* Generated Content */}
              {generatedContent.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-white">Generated Content</h3>
                  <div className="grid gap-4">
                    {generatedContent.map((item, index) => (
                      <div key={item.id} className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6">
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center space-x-2">
                            <div className={`p-2 bg-gradient-to-br ${category.color} bg-opacity-20 rounded-lg`}>
                              <category.icon className="w-4 h-4 text-white" />
                            </div>
                            <span className="text-sm font-medium text-gray-300">
                              Result {index + 1}
                            </span>
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
                          <pre className="whitespace-pre-wrap text-gray-300 text-sm">
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
      </div>
    </div>
  );
}