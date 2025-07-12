import React, { useState } from 'react';
import { 
  Brain, MessageSquare, Image, Video, CheckSquare, 
  ScanText, Lightbulb, Cog, Tractor, Code, 
  Briefcase, Heart, GraduationCap, FileText,
  Zap, Search, Sparkles
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

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
    features: ['Crop Analysis', 'Weather Forecasting', 'Soil Assessment', 'Yield Prediction']
  },
  {
    id: 'coding',
    name: 'Development Tools',
    icon: Code,
    description: 'Code generation, debugging, and software development assistance',
    features: ['Code Generation', 'Bug Detection', 'Code Review', 'Documentation']
  },
  {
    id: 'business',
    name: 'Business Intelligence',
    icon: Briefcase,
    description: 'Market analysis, forecasting, and business strategy',
    features: ['Market Research', 'Financial Analysis', 'Strategy Planning', 'Risk Assessment']
  },
  {
    id: 'healthcare',
    name: 'Healthcare AI',
    icon: Heart,
    description: 'Medical analysis, diagnosis assistance, and health monitoring',
    features: ['Symptom Analysis', 'Drug Research', 'Medical Imaging', 'Treatment Planning']
  },
  {
    id: 'education',
    name: 'Education Tools',
    icon: GraduationCap,
    description: 'Learning assistance, curriculum design, and educational content',
    features: ['Lesson Planning', 'Assessment Tools', 'Learning Analytics', 'Content Creation']
  },
  {
    id: 'research',
    name: 'Research Assistant',
    icon: FileText,
    description: 'Academic research, data analysis, and literature review',
    features: ['Literature Review', 'Data Analysis', 'Research Planning', 'Citation Management']
  }
];

export function AIPlayground() {
  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [result, setResult] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('tools');

  const handleToolUse = async (toolId: string) => {
    if (!input.trim()) return;
    
    setIsProcessing(true);
    setSelectedTool(toolId);
    
    // Simulate AI processing
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const responses = {
      text: `Generated creative content based on: "${input}"\n\nHere's a compelling piece of content that captures the essence of your request with engaging storytelling and clear messaging...`,
      image: `Image generation parameters processed for: "${input}"\n\nCreating a high-quality visual with optimal composition, lighting, and artistic style...`,
      video: `Video generation pipeline initiated for: "${input}"\n\nProducing dynamic video content with smooth transitions and professional quality...`,
      tasks: `Workflow automation designed for: "${input}"\n\nCreated an efficient process flow with error handling and optimization checkpoints...`,
      ocr: `Document analysis completed for uploaded content.\n\nExtracted text, identified key information, and structured data for further processing...`,
      reasoning: `Complex analysis performed on: "${input}"\n\nApplied logical reasoning, considered multiple perspectives, and provided actionable insights...`
    };
    
    setResult(responses[toolId as keyof typeof responses] || 'Processing completed successfully.');
    setIsProcessing(false);
  };

  const getCategoryTools = (category: string) => {
    return aiTools.filter(tool => tool.category === category);
  };

  return (
    <div className="flex-1 bg-gray-950 min-h-screen">
      {/* Header */}
      <div className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50 p-6">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <Brain className="w-8 h-8 text-purple-400" />
            AI Playground
          </h1>
          <p className="text-gray-400">Explore powerful AI tools and industry-specific solutions</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2 bg-gray-900/50 border border-gray-800/50">
            <TabsTrigger value="tools" className="flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Core AI Tools
            </TabsTrigger>
            <TabsTrigger value="industry" className="flex items-center gap-2">
              <Briefcase className="w-4 h-4" />
              Industry Solutions
            </TabsTrigger>
          </TabsList>

          <TabsContent value="tools" className="mt-6">
            {/* Input Section */}
            <Card className="mb-8 bg-gray-900/50 border-gray-800/50">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-purple-400" />
                  AI Input Interface
                </CardTitle>
                <CardDescription>
                  Enter your request and select an AI tool to process it
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Describe what you want to create, analyze, or automate..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  className="bg-gray-800 border-gray-700 text-white"
                  rows={3}
                />
                {result && (
                  <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                    <h4 className="text-white font-medium mb-2">AI Response:</h4>
                    <p className="text-gray-300 whitespace-pre-wrap">{result}</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* AI Tools Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {aiTools.map((tool) => (
                <Card key={tool.id} className="bg-gray-900/50 border-gray-800/50 hover:bg-gray-900/70 transition-colors">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-3">
                      <div className="p-2 bg-purple-500/20 rounded-lg">
                        <tool.icon className="w-6 h-6 text-purple-400" />
                      </div>
                      {tool.name}
                    </CardTitle>
                    <CardDescription>{tool.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex flex-wrap gap-2">
                        {tool.features.map((feature, index) => (
                          <Badge key={index} variant="outline" className="text-xs border-gray-600 text-gray-300">
                            {feature}
                          </Badge>
                        ))}
                      </div>
                      <Button 
                        onClick={() => handleToolUse(tool.id)}
                        disabled={!input.trim() || isProcessing}
                        className="w-full bg-purple-600 hover:bg-purple-700"
                      >
                        {isProcessing && selectedTool === tool.id ? 'Processing...' : `Use ${tool.name}`}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="industry" className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {industryTools.map((tool) => (
                <Card key={tool.id} className="bg-gray-900/50 border-gray-800/50 hover:bg-gray-900/70 transition-colors">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-3">
                      <div className="p-2 bg-cyan-500/20 rounded-lg">
                        <tool.icon className="w-6 h-6 text-cyan-400" />
                      </div>
                      {tool.name}
                    </CardTitle>
                    <CardDescription>{tool.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex flex-wrap gap-2">
                        {tool.features.map((feature, index) => (
                          <Badge key={index} variant="outline" className="text-xs border-gray-600 text-gray-300">
                            {feature}
                          </Badge>
                        ))}
                      </div>
                      <Button 
                        onClick={() => handleToolUse(tool.id)}
                        disabled={!input.trim() || isProcessing}
                        className="w-full bg-cyan-600 hover:bg-cyan-700"
                      >
                        {isProcessing && selectedTool === tool.id ? 'Processing...' : `Launch ${tool.name}`}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}