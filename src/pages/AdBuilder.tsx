import React, { useState, useEffect } from 'react';
import { Layout } from '../components/Layout';
import { UnifiedHeader } from '../components/UnifiedHeader';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Textarea } from '../components/ui/textarea';
import { Badge } from '../components/ui/badge';
import { globalSearch } from '../services/globalSearch';
import { 
  Palette, 
  Target, 
  TrendingUp, 
  Users, 
  Eye, 
  Zap,
  Brain,
  Sparkles,
  BarChart3,
  Lightbulb
} from 'lucide-react';

export default function AdBuilder() {
  const [adType, setAdType] = useState('display');
  const [aiInsights, setAiInsights] = useState<any>(null);
  const [generatingAd, setGeneratingAd] = useState(false);
  const [adData, setAdData] = useState({
    headline: '',
    description: '',
    cta: ''
  });

  useEffect(() => {
    loadAIInsights();
  }, []);

  const loadAIInsights = async () => {
    const insights = await globalSearch.getInsights();
    setAiInsights(insights);
  };

  const generateAIAd = async () => {
    setGeneratingAd(true);
    
    // Simulate AI ad generation
    setTimeout(() => {
      setAdData({
        headline: 'Revolutionary AI-Powered Solution',
        description: 'Experience the future of technology with our breakthrough AI platform that adapts to your needs in real-time.',
        cta: 'Discover the Future'
      });
      setGeneratingAd(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-gray-950 to-blue-1000">
      <UnifiedHeader 
        title="AI Ad Intelligence Studio"
        subtitle="Create significance-driven advertisements with advanced AI targeting"
        activeFeatures={['ai', 'optimized']}
      />

      <div className="p-6 space-y-6">
        <Tabs value={adType} onValueChange={setAdType} className="space-y-6">
          <TabsList className="bg-slate-800/50 border border-slate-700/50">
            <TabsTrigger value="display" className="data-[state=active]:bg-cyan-500/20">
              <Palette className="w-4 h-4 mr-2" />
              Display Ads
            </TabsTrigger>
            <TabsTrigger value="video" className="data-[state=active]:bg-purple-500/20">
              <Eye className="w-4 h-4 mr-2" />
              Video Ads
            </TabsTrigger>
            <TabsTrigger value="interactive" className="data-[state=active]:bg-green-500/20">
              <Zap className="w-4 h-4 mr-2" />
              Interactive
            </TabsTrigger>
          </TabsList>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <TabsContent value="display" className="space-y-6">
                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-cyan-300 flex items-center">
                      <Lightbulb className="w-5 h-5 mr-2" />
                      AI-Enhanced Creative Content
                    </CardTitle>
                    <CardDescription>Let AI help craft compelling, high-significance advertisements</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <label className="text-sm font-medium text-slate-300 mb-2 block">Headline</label>
                      <Input 
                        value={adData.headline}
                        onChange={(e) => setAdData({...adData, headline: e.target.value})}
                        placeholder="Enter compelling headline..."
                        className="bg-slate-700/50 border-slate-600/50"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-300 mb-2 block">Description</label>
                      <Textarea 
                        value={adData.description}
                        onChange={(e) => setAdData({...adData, description: e.target.value})}
                        placeholder="Describe your product or service..."
                        className="bg-slate-700/50 border-slate-600/50"
                        rows={3}
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-300 mb-2 block">Call to Action</label>
                      <Input 
                        value={adData.cta}
                        onChange={(e) => setAdData({...adData, cta: e.target.value})}
                        placeholder="Learn More, Shop Now, Get Started..."
                        className="bg-slate-700/50 border-slate-600/50"
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-purple-900/30 to-cyan-900/30 border-purple-500/30">
                  <CardHeader>
                    <CardTitle className="text-purple-300 flex items-center">
                      <Brain className="w-5 h-5 mr-2" />
                      AI Optimization Engine
                    </CardTitle>
                    <CardDescription>Advanced AI tools for maximum ad performance</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Button 
                      onClick={generateAIAd}
                      disabled={generatingAd}
                      className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600"
                    >
                      {generatingAd ? (
                        <>
                          <Sparkles className="w-4 h-4 mr-2 animate-spin" />
                          Generating AI Content...
                        </>
                      ) : (
                        <>
                          <Brain className="w-4 h-4 mr-2" />
                          Generate AI-Optimized Ad
                        </>
                      )}
                    </Button>
                    <div className="grid grid-cols-2 gap-3">
                      <Button variant="outline" className="border-slate-600/50 hover:bg-slate-700/50">
                        <TrendingUp className="w-4 h-4 mr-2" />
                        Predict Performance
                      </Button>
                      <Button variant="outline" className="border-slate-600/50 hover:bg-slate-700/50">
                        <BarChart3 className="w-4 h-4 mr-2" />
                        A/B Test Variants
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="video" className="space-y-6">
                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-purple-300">Video Content</CardTitle>
                    <CardDescription>Create engaging video advertisements</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="border-2 border-dashed border-slate-600/50 rounded-lg p-8 text-center">
                      <Eye className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                      <p className="text-slate-400">Upload video or generate with AI</p>
                      <Button className="mt-4">Upload Video</Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="interactive" className="space-y-6">
                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-green-300">Interactive Elements</CardTitle>
                    <CardDescription>Create engaging, interactive ad experiences</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <Button variant="outline" className="h-20 flex-col">
                        <Zap className="w-6 h-6 mb-2" />
                        Quiz/Poll
                      </Button>
                      <Button variant="outline" className="h-20 flex-col">
                        <Target className="w-6 h-6 mb-2" />
                        Product Demo
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </div>

            <div className="space-y-6">
              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-cyan-300 flex items-center">
                    <Target className="w-5 h-5 mr-2" />
                    Smart Targeting Engine
                  </CardTitle>
                  <CardDescription>AI-powered audience intelligence</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-slate-300 mb-2 block">AI-Suggested Audiences</label>
                    <div className="space-y-2">
                      <Badge variant="secondary" className="bg-cyan-500/20 text-cyan-300 border-cyan-500/30">
                        Tech Early Adopters (94% match)
                      </Badge>
                      <Badge variant="secondary" className="bg-purple-500/20 text-purple-300 border-purple-500/30">
                        Innovation Enthusiasts (89% match)
                      </Badge>
                      <Badge variant="secondary" className="bg-green-500/20 text-green-300 border-green-500/30">
                        High-Intent Buyers (87% match)
                      </Badge>
                    </div>
                  </div>
                  <Button variant="outline" className="w-full border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10">
                    <Users className="w-4 h-4 mr-2" />
                    Refine AI Targeting
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-green-900/30 to-blue-900/30 border-green-500/30">
                <CardHeader>
                  <CardTitle className="text-green-300 flex items-center">
                    <Sparkles className="w-5 h-5 mr-2" />
                    AI Performance Prediction
                  </CardTitle>
                  <CardDescription>Real-time significance & impact analysis</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">Significance Score</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-16 h-2 bg-slate-700 rounded-full">
                          <div className="w-14 h-2 bg-gradient-to-r from-green-500 to-cyan-500 rounded-full"></div>
                        </div>
                        <span className="text-green-400 font-bold">8.7/10</span>
                      </div>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Predicted CTR</span>
                      <span className="text-cyan-400 font-semibold">3.2%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Est. Reach</span>
                      <span className="text-purple-400 font-semibold">124K</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">ROI Prediction</span>
                      <span className="text-green-400 font-semibold">+340%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {aiInsights && (
                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-purple-300 flex items-center">
                      <TrendingUp className="w-5 h-5 mr-2" />
                      Trending Insights
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {aiInsights.trendingTopics?.slice(0, 3).map((topic: string, index: number) => (
                        <div key={index} className="p-2 bg-slate-700/30 rounded text-sm text-slate-300">
                          {topic}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </Tabs>
      </div>
    </div>
  );
}
