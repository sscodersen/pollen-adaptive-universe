
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Textarea } from '../components/ui/textarea';
import { Badge } from '../components/ui/badge';
import { 
  Palette, 
  Target, 
  TrendingUp, 
  Users, 
  Eye, 
  Zap,
  Brain,
  Sparkles
} from 'lucide-react';

export default function AdBuilder() {
  const [adType, setAdType] = useState('display');

  return (
    <Layout>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              AI Ad Builder
            </h1>
            <p className="text-slate-400 mt-2">Create intelligent, significance-driven advertisements</p>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="border-purple-500/30 text-purple-300">
              <Brain className="w-3 h-3 mr-1" />
              AI Powered
            </Badge>
            <Badge variant="outline" className="border-cyan-500/30 text-cyan-300">
              <Sparkles className="w-3 h-3 mr-1" />
              Smart Targeting
            </Badge>
          </div>
        </div>

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
                    <CardTitle className="text-cyan-300">Creative Content</CardTitle>
                    <CardDescription>Define your ad's visual and textual elements</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <label className="text-sm font-medium text-slate-300 mb-2 block">Headline</label>
                      <Input 
                        placeholder="Enter compelling headline..."
                        className="bg-slate-700/50 border-slate-600/50"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-300 mb-2 block">Description</label>
                      <Textarea 
                        placeholder="Describe your product or service..."
                        className="bg-slate-700/50 border-slate-600/50"
                        rows={3}
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-300 mb-2 block">Call to Action</label>
                      <Input 
                        placeholder="Learn More, Shop Now, Get Started..."
                        className="bg-slate-700/50 border-slate-600/50"
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-purple-300">AI Optimization</CardTitle>
                    <CardDescription>Let AI enhance your ad performance</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Button className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600">
                      <Brain className="w-4 h-4 mr-2" />
                      Generate AI Variants
                    </Button>
                    <Button variant="outline" className="w-full border-slate-600/50 hover:bg-slate-700/50">
                      <TrendingUp className="w-4 h-4 mr-2" />
                      Predict Performance
                    </Button>
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
                  <CardTitle className="text-cyan-300">Smart Targeting</CardTitle>
                  <CardDescription>AI-powered audience selection</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-slate-300 mb-2 block">Target Audience</label>
                    <div className="space-y-2">
                      <Badge variant="secondary">Tech Enthusiasts</Badge>
                      <Badge variant="secondary">Age 25-40</Badge>
                      <Badge variant="secondary">High Intent</Badge>
                    </div>
                  </div>
                  <Button variant="outline" className="w-full">
                    <Users className="w-4 h-4 mr-2" />
                    Refine Targeting
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-green-300">Performance Preview</CardTitle>
                  <CardDescription>Estimated ad performance</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Significance Score</span>
                      <span className="text-green-400 font-bold">8.7/10</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Predicted CTR</span>
                      <span className="text-cyan-400">3.2%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Est. Reach</span>
                      <span className="text-purple-400">124K</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </Tabs>
      </div>
    </Layout>
  );
}
