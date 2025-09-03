import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { 
  Bot, Brain, Code, GraduationCap, Heart, DollarSign, 
  Plane, Home, Sprout, Palette, Music, ShoppingCart,
  Settings, TrendingUp, Zap, Globe, Database, Activity
} from 'lucide-react';
import { APIKeyManager } from '@/services/industryServices';
import { smartHomeService, agricultureService, developmentService, educationService } from '@/services/industryServices';
import { healthcareService } from '@/services/healthcareService';
import { financeService } from '@/services/financeService';
import { travelService } from '@/services/travelService';
import { pollenAdaptiveService } from '@/services/pollenAdaptiveService';
import { PollenStatus } from '@/components/PollenStatus';
import { ContentManagementDashboard } from '@/components/ContentManagementDashboard';
import { PlatformOptimizer } from '@/components/PlatformOptimizer';

interface IndustrySection {
  id: string;
  name: string;
  icon: React.ElementType;
  description: string;
  color: string;
  features: string[];
}

const INDUSTRIES: IndustrySection[] = [
  {
    id: 'ai-playground',
    name: 'AI Playground',
    icon: Bot,
    description: 'Interactive content generation for ads, music, social posts, and more',
    color: 'from-purple-500 to-pink-500',
    features: ['Content Generation', 'Task Automation', 'Custom Templates']
  },
  {
    id: 'entertainment',
    name: 'Entertainment',
    icon: Palette,
    description: 'Real-time trending content curation and creative generation',
    color: 'from-red-500 to-orange-500',
    features: ['Script Writing', 'Game Design', 'Music Creation']
  },
  {
    id: 'smart-home',
    name: 'Smart Home',
    icon: Home,
    description: 'Automated routines, voice commands, and security analysis',
    color: 'from-blue-500 to-cyan-500',
    features: ['Device Automation', 'Voice Commands', 'Security Analysis']
  },
  {
    id: 'agriculture',
    name: 'Agriculture',
    icon: Sprout,
    description: 'Crop management, weather analysis, and market trends',
    color: 'from-green-500 to-emerald-500',
    features: ['Crop Recommendations', 'Weather Insights', 'Market Analysis']
  },
  {
    id: 'development',
    name: 'Coding & Development',
    icon: Code,
    description: 'Code generation, bug fixing, and documentation',
    color: 'from-indigo-500 to-purple-500',
    features: ['Code Generation', 'Bug Fixing', 'Documentation']
  },
  {
    id: 'education',
    name: 'Education',
    icon: GraduationCap,
    description: 'Study guides, practice problems, and essay assistance',
    color: 'from-yellow-500 to-orange-500',
    features: ['Study Guides', 'Practice Problems', 'Essay Writing']
  },
  {
    id: 'healthcare',
    name: 'Healthcare',
    icon: Heart,
    description: 'Patient reports, treatment plans, and research curation',
    color: 'from-pink-500 to-rose-500',
    features: ['Patient Reports', 'Treatment Plans', 'Research Curation']
  },
  {
    id: 'finance',
    name: 'Finance',
    icon: DollarSign,
    description: 'Market analysis, investment strategies, and budgeting',
    color: 'from-emerald-500 to-teal-500',
    features: ['Market Analysis', 'Investment Planning', 'Budget Management']
  },
  {
    id: 'travel',
    name: 'Travel & Hospitality',
    icon: Plane,
    description: 'Personalized itineraries, recommendations, and local guides',
    color: 'from-sky-500 to-blue-500',
    features: ['Travel Planning', 'Accommodations', 'Local Guides']
  }
];

const IndustryDashboard: React.FC = () => {
  const [selectedIndustry, setSelectedIndustry] = useState<string>('ai-playground');
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedContent, setGeneratedContent] = useState<any>(null);
  const [inputText, setInputText] = useState('');

  useEffect(() => {
    loadApiKeys();
  }, []);

  const loadApiKeys = async () => {
    const keys = await APIKeyManager.getAllKeys();
    setApiKeys(keys);
  };

  const handleApiKeyUpdate = async (service: string, key: string) => {
    if (key.trim()) {
      await APIKeyManager.setKey(service, key);
    } else {
      await APIKeyManager.removeKey(service);
    }
    await loadApiKeys();
  };

  const generateContent = async () => {
    if (!inputText.trim()) return;
    
    setIsGenerating(true);
    setGeneratedContent(null);

    try {
      let result;
      
      switch (selectedIndustry) {
        case 'ai-playground':
          result = await pollenAdaptiveService.proposeTask(inputText);
          break;
        case 'smart-home':
          result = await smartHomeService.generateRoutine(inputText);
          break;
        case 'agriculture':
          result = await agricultureService.generateCropRecommendation(inputText, 'spring');
          break;
        case 'development':
          result = await developmentService.generateCode(inputText, 'javascript');
          break;
        case 'education':
          result = await educationService.generateStudyGuide(inputText, 'intermediate');
          break;
        case 'healthcare':
          result = await healthcareService.generatePatientReport([inputText], 'General assessment');
          break;
        case 'finance':
          result = await financeService.generateMarketAnalysis(inputText);
          break;
        case 'travel':
          result = await travelService.generateItinerary({
            destination: inputText,
            duration: 5,
            budget: 2000,
            interests: ['culture', 'food'],
            travelStyle: 'moderate'
          });
          break;
        default:
          result = await pollenAdaptiveService.proposeTask(inputText);
      }
      
      setGeneratedContent(result);
    } catch (error) {
      console.error('Generation failed:', error);
      setGeneratedContent({ error: 'Generation failed. Please try again.' });
    } finally {
      setIsGenerating(false);
    }
  };

  const renderApiKeySettings = () => (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          API Configuration
        </CardTitle>
        <CardDescription>
          Configure your API keys for enhanced AI capabilities. Keys are stored locally and securely.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {['openai', 'claude', 'gemini', 'perplexity'].map(service => (
            <div key={service} className="space-y-2">
              <label className="text-sm font-medium capitalize">{service} API Key</label>
              <Input
                type="password"
                placeholder={`Enter ${service} API key`}
                defaultValue={apiKeys[service] || ''}
                onBlur={(e) => handleApiKeyUpdate(service, e.target.value)}
              />
            </div>
          ))}
        </div>
        <div className="text-xs text-muted-foreground">
          🔒 API keys are encrypted and stored locally in your browser. They never leave your device.
        </div>
      </CardContent>
    </Card>
  );

  const renderIndustryGrid = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
      {INDUSTRIES.map((industry) => {
        const Icon = industry.icon;
        const isSelected = selectedIndustry === industry.id;
        
        return (
          <Card 
            key={industry.id}
            className={`cursor-pointer transition-all duration-200 hover:shadow-lg ${
              isSelected ? 'ring-2 ring-primary shadow-lg' : 'hover:shadow-md'
            }`}
            onClick={() => setSelectedIndustry(industry.id)}
          >
            <CardContent className="p-6">
              <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${industry.color} flex items-center justify-center mb-4`}>
                <Icon className="h-6 w-6 text-white" />
              </div>
              <h3 className="font-semibold text-lg mb-2">{industry.name}</h3>
              <p className="text-sm text-muted-foreground mb-4">{industry.description}</p>
              <div className="flex flex-wrap gap-1">
                {industry.features.map((feature, idx) => (
                  <Badge key={idx} variant="secondary" className="text-xs">
                    {feature}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );

  const renderContentGenerator = () => {
    const selectedIndustryData = INDUSTRIES.find(i => i.id === selectedIndustry);
    if (!selectedIndustryData) return null;

    const Icon = selectedIndustryData.icon;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Icon className="h-5 w-5" />
            {selectedIndustryData.name} Generator
          </CardTitle>
          <CardDescription>
            Generate AI-powered content for {selectedIndustryData.name.toLowerCase()}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Input Prompt</label>
            <Input
              placeholder={getPlaceholderText(selectedIndustry)}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && generateContent()}
            />
          </div>
          
          <Button 
            onClick={generateContent} 
            disabled={isGenerating || !inputText.trim()}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Generating...
              </>
            ) : (
              <>
                <Zap className="h-4 w-4 mr-2" />
                Generate Content
              </>
            )}
          </Button>

          {generatedContent && (
            <div className="mt-6 p-4 bg-muted rounded-lg">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Generated Result
              </h4>
              <pre className="text-sm overflow-auto whitespace-pre-wrap">
                {JSON.stringify(generatedContent, null, 2)}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const getPlaceholderText = (industryId: string): string => {
    const placeholders = {
      'ai-playground': 'Create a social media campaign for a new product...',
      'smart-home': 'Turn on lights when motion is detected in the evening...',
      'agriculture': 'Best crops for Mediterranean climate...',
      'development': 'Create a function to validate email addresses...',
      'education': 'Introduction to Machine Learning...',
      'healthcare': 'Patient experiencing headaches and fatigue...',
      'finance': 'AAPL stock analysis...',
      'travel': 'Tokyo'
    };
    return placeholders[industryId as keyof typeof placeholders] || 'Enter your request...';
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
          Pollen AI Platform
        </h1>
        <p className="text-xl text-muted-foreground">
          Comprehensive AI-driven solutions across industries
        </p>
      </div>

      <Tabs defaultValue="industries" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="industries" className="flex items-center gap-2">
            <Globe className="h-4 w-4" />
            Industries
          </TabsTrigger>
          <TabsTrigger value="content-mgmt" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Content
          </TabsTrigger>
          <TabsTrigger value="optimizer" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Optimizer
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="industries" className="mt-8">
          {renderIndustryGrid()}
          {renderContentGenerator()}
        </TabsContent>

        <TabsContent value="content-mgmt" className="mt-8">
          <ContentManagementDashboard />
        </TabsContent>

        <TabsContent value="optimizer" className="mt-8">
          <PlatformOptimizer />
        </TabsContent>

        <TabsContent value="settings" className="mt-8">
          <div className="grid gap-8">
            <PollenStatus />
            {renderApiKeySettings()}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default IndustryDashboard;