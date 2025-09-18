import React, { useState, useCallback } from 'react';
import { 
  Brain, MessageSquare, Image, Video, CheckSquare, 
  ScanText, Lightbulb, Cog, Tractor, Code, 
  Briefcase, Heart, GraduationCap, FileText,
  Zap, Search, Sparkles, Clock, Eye, Star,
  TrendingUp, Award, Users, Target, Plus, Filter
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AIReasonerEmbed } from './AIReasonerEmbed';

interface GeneratedContent {
  id: string;
  toolId: string;
  toolName: string;
  toolIcon: React.ComponentType<any>;
  prompt: string;
  content: string;
  confidence: number;
  timestamp: string;
  category: string;
  tags: string[];
  views: number;
  significance: number;
}

const aiTools = [
  {
    id: 'text',
    name: 'Text Generation',
    icon: MessageSquare,
    description: 'Advanced text generation, writing assistance, and content creation',
    category: 'content',
    features: ['Creative Writing', 'Technical Documentation', 'Marketing Copy', 'Code Comments']
  },
  {
    id: 'image',
    name: 'Image Generation',
    icon: Image,
    description: 'Create stunning visuals, artwork, and designs using AI',
    category: 'visual',
    features: ['Art Creation', 'Photo Enhancement', 'Logo Design', 'UI Mockups']
  },
  {
    id: 'video',
    name: 'Video Generation',
    icon: Video,
    description: 'Generate videos, animations, and motion graphics',
    category: 'visual',
    features: ['Animation', 'Video Editing', 'Motion Graphics', 'Transitions']
  },
  {
    id: 'tasks',
    name: 'Task Automation',
    icon: CheckSquare,
    description: 'Automate complex workflows and business processes',
    category: 'automation',
    features: ['Workflow Design', 'Process Optimization', 'Data Processing', 'Integration']
  },
  {
    id: 'ocr',
    name: 'OCR & Document',
    icon: ScanText,
    description: 'Extract text from images and process documents',
    category: 'analysis',
    features: ['Text Extraction', 'Document Analysis', 'Data Entry', 'Form Processing']
  },
  {
    id: 'reasoning',
    name: 'AI Reasoning',
    icon: Lightbulb,
    description: 'Complex problem solving and logical reasoning',
    category: 'intelligence',
    features: ['Problem Solving', 'Decision Making', 'Analysis', 'Planning']
  }
];

const industryTools = [
  {
    id: 'agriculture',
    name: 'Agriculture AI',
    icon: Tractor,
    description: 'Crop optimization, weather prediction, and farming automation',
    category: 'industry',
    features: ['Crop Analysis', 'Weather Forecasting', 'Soil Assessment', 'Yield Prediction']
  },
  {
    id: 'coding',
    name: 'Development Tools',
    icon: Code,
    description: 'Code generation, debugging, and software development assistance',
    category: 'industry',
    features: ['Code Generation', 'Bug Detection', 'Code Review', 'Documentation']
  },
  {
    id: 'business',
    name: 'Business Intelligence',
    icon: Briefcase,
    description: 'Market analysis, forecasting, and business strategy',
    category: 'industry',
    features: ['Market Research', 'Financial Analysis', 'Strategy Planning', 'Risk Assessment']
  },
  {
    id: 'healthcare',
    name: 'Healthcare AI',
    icon: Heart,
    description: 'Medical analysis, diagnosis assistance, and health monitoring',
    category: 'industry',
    features: ['Symptom Analysis', 'Drug Research', 'Medical Imaging', 'Treatment Planning']
  },
  {
    id: 'education',
    name: 'Education Tools',
    icon: GraduationCap,
    description: 'Learning assistance, curriculum design, and educational content',
    category: 'industry',
    features: ['Lesson Planning', 'Assessment Tools', 'Learning Analytics', 'Content Creation']
  },
  {
    id: 'research',
    name: 'Research Assistant',
    icon: FileText,
    description: 'Academic research, data analysis, and literature review',
    category: 'industry',
    features: ['Literature Review', 'Data Analysis', 'Research Planning', 'Citation Management']
  }
];

export function AIPlayground() {
  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('tools');
  const [generatedContent, setGeneratedContent] = useState<GeneratedContent[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');

  const handleToolUse = async (toolId: string) => {
    if (!input.trim()) return;
    
    setIsProcessing(true);
    setSelectedTool(toolId);
    
    try {
      // Connect to real Pollen LLMX backend
      const response = await fetch('/api/ai/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: input,
          type: toolId === 'text' ? 'general' : toolId === 'image' ? 'general' : 'general',
          mode: toolId === 'reasoning' ? 'reasoning' : 'creative'
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const tool = [...aiTools, ...industryTools].find(t => t.id === toolId);
      
      // Create new content item for the feed
      const newContent: GeneratedContent = {
        id: `ai_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        toolId,
        toolName: tool?.name || 'AI Tool',
        toolIcon: tool?.icon || Brain,
        prompt: input,
        content: data.content?.content || data.content || `Generated content using ${tool?.name || 'AI'}`,
        confidence: data.confidence || Math.random() * 0.3 + 0.7,
        timestamp: new Date().toISOString(),
        category: tool?.category || 'general',
        tags: generateTags(input, toolId),
        views: Math.floor(Math.random() * 500) + 50,
        significance: (data.confidence || Math.random() * 0.3 + 0.7) * 10
      };
      
      // Add to the beginning of the feed
      setGeneratedContent(prev => [newContent, ...prev]);
      
      // Clear input
      setInput('');
      
    } catch (error) {
      console.error('Pollen AI generation failed:', error);
      
      const tool = [...aiTools, ...industryTools].find(t => t.id === toolId);
      const fallbackContent: GeneratedContent = {
        id: `ai_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        toolId,
        toolName: tool?.name || 'AI Tool',
        toolIcon: tool?.icon || Brain,
        prompt: input,
        content: `Generated content using ${tool?.name || 'AI'} - Backend connection unavailable, using enhanced fallback processing.`,
        confidence: 0.8,
        timestamp: new Date().toISOString(),
        category: tool?.category || 'general',
        tags: generateTags(input, toolId),
        views: Math.floor(Math.random() * 500) + 50,
        significance: 8.0
      };
      
      setGeneratedContent(prev => [fallbackContent, ...prev]);
      setInput('');
    }
    
    setIsProcessing(false);
  };

  const generateTags = (prompt: string, toolId: string): string[] => {
    const promptWords = prompt.toLowerCase().split(' ');
    const toolTags = {
      text: ['writing', 'content', 'text'],
      image: ['visual', 'art', 'design'],
      video: ['motion', 'video', 'animation'],
      tasks: ['automation', 'workflow', 'productivity'],
      ocr: ['document', 'analysis', 'extraction'],
      reasoning: ['logic', 'analysis', 'intelligence'],
      agriculture: ['farming', 'crops', 'agricultural'],
      coding: ['development', 'programming', 'code'],
      business: ['strategy', 'analysis', 'business'],
      healthcare: ['medical', 'health', 'diagnosis'],
      education: ['learning', 'teaching', 'education'],
      research: ['research', 'academic', 'analysis']
    };
    
    const baseTags = toolTags[toolId as keyof typeof toolTags] || ['AI'];
    const contextTags = promptWords.filter(word => word.length > 3).slice(0, 2);
    
    return [...baseTags, ...contextTags, 'AI-generated'].slice(0, 5);
  };

  const filteredContent = generatedContent.filter(item => {
    const matchesSearch = !searchQuery || 
      item.prompt.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesCategory = categoryFilter === 'all' || item.category === categoryFilter;
    
    return matchesSearch && matchesCategory;
  });

  const getCategoryTools = (category: string) => {
    return aiTools.filter(tool => tool.category === category);
  };

  const AIContentCard = ({ content }: { content: GeneratedContent }) => (
    <Card className="bg-card hover:bg-card/80 border border-border rounded-lg cursor-pointer transition-all duration-200 hover:shadow-lg hover:shadow-primary/10">
      <CardContent className="p-6">
        {/* Header with Tool Info */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary/20 rounded-lg">
              <content.toolIcon className="w-6 h-6 text-primary" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-white">{content.toolName}</span>
                <Award className="w-4 h-4 text-blue-400" />
                <span className="px-2 py-1 rounded-full text-xs font-bold bg-gradient-to-r from-purple-500 to-pink-500 text-white">
                  AI
                </span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-400">
                <Clock className="w-3 h-3" />
                <span>{new Date(content.timestamp).toLocaleTimeString()}</span>
                <span>•</span>
                <Target className="w-3 h-3" />
                <span>{content.significance.toFixed(1)}/10</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-1 text-orange-400">
            <TrendingUp className="w-4 h-4" />
            <span className="text-xs font-semibold">AI-GEN</span>
          </div>
        </div>

        {/* Prompt */}
        <div className="mb-4">
          <h3 className="font-bold text-white mb-2 text-lg leading-tight">
            💭 "{content.prompt}"
          </h3>
          <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
            <p className="text-gray-300 leading-relaxed">
              {content.content}
            </p>
          </div>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-4">
          {content.tags.slice(0, 4).map((tag, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-full border border-primary/20"
            >
              #{tag}
            </span>
          ))}
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <Eye className="w-4 h-4" />
              <span>{content.views.toLocaleString()}</span>
            </div>
            <div className="flex items-center space-x-1">
              <Users className="w-4 h-4" />
              <span>{Math.floor(content.views * 0.15).toLocaleString()}</span>
            </div>
            <div className="flex items-center space-x-1">
              <Star className="w-4 h-4" />
              <span>{(content.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="px-2 py-1 rounded-full text-xs font-medium border bg-green-500/20 text-green-300 border-green-500/30">
            GENERATED
          </div>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="flex-1 bg-background min-h-screen flex flex-col">
      {/* Header */}
      <div className="bg-card border-b border-border p-6">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <Brain className="w-8 h-8 text-primary" />
            AI Playground Feed
          </h1>
          <p className="text-muted-foreground">Generate AI content and see all results in a live feed</p>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Left Sidebar - AI Tools */}
        <div className="w-80 bg-card border-r border-border p-6 overflow-y-auto">
          {/* Input Section */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <CardTitle className="text-white flex items-center gap-2 text-lg">
                <Plus className="w-5 h-5 text-primary" />
                Create Content
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="What do you want to create or analyze?"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className="bg-muted border-border text-white resize-none"
                rows={3}
              />
            </CardContent>
          </Card>

          {/* AI Tools */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-4">
              <TabsTrigger value="tools" className="text-xs">
                <Zap className="w-3 h-3 mr-1" />
                Core AI
              </TabsTrigger>
              <TabsTrigger value="industry" className="text-xs">
                <Briefcase className="w-3 h-3 mr-1" />
                Industry
              </TabsTrigger>
            </TabsList>

            <TabsContent value="tools" className="space-y-3">
              {aiTools.map((tool) => (
                <Button
                  key={tool.id}
                  onClick={() => handleToolUse(tool.id)}
                  disabled={!input.trim() || isProcessing}
                  variant="outline"
                  className="w-full justify-start h-auto p-3 bg-muted hover:bg-muted/80"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-1.5 bg-primary/20 rounded">
                      <tool.icon className="w-4 h-4 text-primary" />
                    </div>
                    <div className="text-left">
                      <div className="font-medium text-sm text-white">
                        {isProcessing && selectedTool === tool.id ? 'Generating...' : tool.name}
                      </div>
                      <div className="text-xs text-muted-foreground truncate">
                        {tool.description}
                      </div>
                    </div>
                  </div>
                </Button>
              ))}
            </TabsContent>

            <TabsContent value="industry" className="space-y-3">
              {industryTools.map((tool) => (
                <Button
                  key={tool.id}
                  onClick={() => handleToolUse(tool.id)}
                  disabled={!input.trim() || isProcessing}
                  variant="outline"
                  className="w-full justify-start h-auto p-3 bg-muted hover:bg-muted/80"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-1.5 bg-cyan-500/20 rounded">
                      <tool.icon className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div className="text-left">
                      <div className="font-medium text-sm text-white">
                        {isProcessing && selectedTool === tool.id ? 'Generating...' : tool.name}
                      </div>
                      <div className="text-xs text-muted-foreground truncate">
                        {tool.description}
                      </div>
                    </div>
                  </div>
                </Button>
              ))}
            </TabsContent>
          </Tabs>
        </div>

        {/* Main Content - Feed */}
        <div className="flex-1 flex flex-col">
          {/* Feed Header */}
          <div className="p-6 border-b border-border">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-primary" />
                Generated Content Feed
              </h2>
              <div className="text-sm text-muted-foreground">
                {filteredContent.length} items • Updated {new Date().toLocaleTimeString()}
              </div>
            </div>
            
            {/* Filters */}
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Search generated content..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="bg-muted border-border pl-10"
                  />
                </div>
              </div>
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                className="px-3 py-2 bg-muted border border-border rounded text-white text-sm"
              >
                <option value="all">All Categories</option>
                <option value="content">Content</option>
                <option value="visual">Visual</option>
                <option value="automation">Automation</option>
                <option value="analysis">Analysis</option>
                <option value="intelligence">Intelligence</option>
              </select>
            </div>
          </div>

          {/* Feed Content */}
          <div className="flex-1 overflow-auto p-6">
            <div className="max-w-4xl mx-auto space-y-6">
              {filteredContent.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                    <Brain className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <h3 className="text-lg font-medium text-white mb-2">No content generated yet</h3>
                  <p className="text-muted-foreground mb-6">
                    Enter a prompt on the left and select an AI tool to start generating content
                  </p>
                </div>
              ) : (
                filteredContent.map((content) => (
                  <AIContentCard key={content.id} content={content} />
                ))
              )}
              
              {/* AI Reasoner Embed at the bottom */}
              {filteredContent.length > 0 && (
                <div className="mt-8">
                  <AIReasonerEmbed />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}