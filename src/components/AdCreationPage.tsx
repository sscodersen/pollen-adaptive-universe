
import React, { useState } from 'react';
import { Megaphone, Target, Eye, Zap, TrendingUp, Users, Globe, Sparkles } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const adTemplates = [
  {
    id: 1,
    name: 'Tech Product Launch',
    category: 'Technology',
    thumbnail: 'bg-gradient-to-br from-blue-500 to-cyan-500',
    description: 'Perfect for launching tech products with modern, sleek designs',
    ctr: 4.2,
    conversions: 12.8
  },
  {
    id: 2,
    name: 'Social Campaign',
    category: 'Social Media',
    thumbnail: 'bg-gradient-to-br from-purple-500 to-pink-500',
    description: 'Engaging social media ads with high viral potential',
    ctr: 6.7,
    conversions: 9.3
  },
  {
    id: 3,
    name: 'E-commerce Promo',
    category: 'Retail',
    thumbnail: 'bg-gradient-to-br from-emerald-500 to-teal-500',
    description: 'Drive sales with compelling product showcases',
    ctr: 5.1,
    conversions: 15.2
  },
  {
    id: 4,
    name: 'Brand Awareness',
    category: 'Branding',
    thumbnail: 'bg-gradient-to-br from-orange-500 to-red-500',
    description: 'Build brand recognition with memorable creative assets',
    ctr: 3.8,
    conversions: 8.9
  }
];

const targetingOptions = [
  { name: 'Demographics', icon: Users, active: true },
  { name: 'Interests', icon: Target, active: false },
  { name: 'Behaviors', icon: Zap, active: true },
  { name: 'Locations', icon: Globe, active: false }
];

const campaignStats = [
  { label: 'Total Campaigns', value: '1,234', change: '+12%' },
  { label: 'Active Ads', value: '456', change: '+8%' },
  { label: 'Avg. CTR', value: '4.8%', change: '+15%' },
  { label: 'Conversions', value: '2,890', change: '+23%' }
];

export function AdCreationPage() {
  const [selectedTemplate, setSelectedTemplate] = useState<number | null>(null);
  const [adTitle, setAdTitle] = useState('');
  const [adDescription, setAdDescription] = useState('');

  return (
    <div className="flex-1 bg-gray-950 min-h-screen">
      {/* Header */}
      <div className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50 p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <Megaphone className="w-8 h-8 text-orange-400" />
            Ad Creation Studio
          </h1>
          <p className="text-gray-400">Create high-converting ads with AI-powered insights and templates</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6">
        {/* Campaign Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {campaignStats.map((stat, index) => (
            <Card key={index} className="bg-gray-900/50 border-gray-800/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-gray-400">{stat.label}</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="text-2xl font-bold text-white">{stat.value}</div>
                <p className="text-xs text-green-400 flex items-center gap-1 mt-1">
                  <TrendingUp className="w-3 h-3" />
                  {stat.change}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Creation Area */}
          <div className="lg:col-span-2 space-y-8">
            {/* Ad Templates */}
            <div>
              <h2 className="text-2xl font-bold text-white mb-4">Choose Template</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {adTemplates.map((template) => (
                  <div
                    key={template.id}
                    onClick={() => setSelectedTemplate(template.id)}
                    className={`bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 cursor-pointer transition-all hover:bg-gray-900/70 ${
                      selectedTemplate === template.id ? 'border-orange-500/50 bg-orange-500/10' : ''
                    }`}
                  >
                    <div className={`${template.thumbnail} h-32 rounded-lg mb-4 flex items-center justify-center`}>
                      <Megaphone className="w-12 h-12 text-white/80" />
                    </div>
                    
                    <div className="mb-3">
                      <Badge className="bg-orange-500/20 text-orange-300 hover:bg-orange-500/20 mb-2">
                        {template.category}
                      </Badge>
                      <h3 className="font-semibold text-white mb-1">{template.name}</h3>
                      <p className="text-sm text-gray-400">{template.description}</p>
                    </div>
                    
                    <div className="flex justify-between text-sm">
                      <div>
                        <span className="text-gray-400">CTR: </span>
                        <span className="text-green-400 font-medium">{template.ctr}%</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Conv: </span>
                        <span className="text-blue-400 font-medium">{template.conversions}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Ad Content Creation */}
            <div>
              <h2 className="text-2xl font-bold text-white mb-4">Create Ad Content</h2>
              <Card className="bg-gray-900/50 border-gray-800/50">
                <CardContent className="p-6 space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Ad Title
                    </label>
                    <Input
                      value={adTitle}
                      onChange={(e) => setAdTitle(e.target.value)}
                      placeholder="Enter compelling ad title..."
                      className="bg-gray-800/50 border-gray-700 text-white placeholder-gray-400"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Ad Description
                    </label>
                    <textarea
                      value={adDescription}
                      onChange={(e) => setAdDescription(e.target.value)}
                      placeholder="Write your ad description here..."
                      rows={4}
                      className="w-full p-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:border-orange-500 outline-none"
                    />
                  </div>
                  
                  <div className="flex gap-3">
                    <Button className="bg-orange-600 hover:bg-orange-700 flex-1">
                      <Sparkles className="w-4 h-4 mr-2" />
                      Generate with AI
                    </Button>
                    <Button variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                      Preview
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-8">
            {/* Targeting Options */}
            <div>
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-orange-400" />
                Targeting
              </h2>
              <Card className="bg-gray-900/50 border-gray-800/50">
                <CardContent className="p-4">
                  <div className="space-y-4">
                    {targetingOptions.map((option, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <option.icon className="w-5 h-5 text-gray-400" />
                          <span className="text-white">{option.name}</span>
                        </div>
                        <div className={`w-10 h-6 rounded-full transition-colors ${
                          option.active ? 'bg-orange-500' : 'bg-gray-600'
                        }`}>
                          <div className={`w-4 h-4 bg-white rounded-full mt-1 transition-transform ${
                            option.active ? 'translate-x-5' : 'translate-x-1'
                          }`} />
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <Button className="w-full mt-6 bg-orange-600 hover:bg-orange-700">
                    Configure Targeting
                  </Button>
                </CardContent>
              </Card>
            </div>

            {/* Performance Predictions */}
            <div>
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5 text-blue-400" />
                Performance Prediction
              </h2>
              <Card className="bg-gray-900/50 border-gray-800/50">
                <CardContent className="p-4 space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Expected Reach</span>
                    <span className="text-white font-semibold">45K - 78K</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Estimated CTR</span>
                    <span className="text-green-400 font-semibold">4.2% - 6.8%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Cost per Click</span>
                    <span className="text-blue-400 font-semibold">$0.45 - $0.89</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Conversion Rate</span>
                    <span className="text-purple-400 font-semibold">8.5% - 12.3%</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Quick Actions */}
            <div>
              <h2 className="text-xl font-bold text-white mb-4">Quick Actions</h2>
              <div className="space-y-3">
                <Button className="w-full justify-start bg-green-600 hover:bg-green-700">
                  <Zap className="w-4 h-4 mr-2" />
                  Launch Campaign
                </Button>
                <Button variant="outline" className="w-full justify-start border-gray-700 text-gray-300 hover:bg-gray-800">
                  <Eye className="w-4 h-4 mr-2" />
                  A/B Testing
                </Button>
                <Button variant="outline" className="w-full justify-start border-gray-700 text-gray-300 hover:bg-gray-800">
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Analytics
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
