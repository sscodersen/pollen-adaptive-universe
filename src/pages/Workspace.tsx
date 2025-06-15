
import React, { useState, useEffect } from 'react';
import { UnifiedHeader } from '../components/UnifiedHeader';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { globalSearch } from '../services/globalSearch';
import { 
  Users, 
  Brain, 
  Zap, 
  Target, 
  Globe, 
  BarChart3,
  Calendar,
  MessageSquare,
  FileText,
  Lightbulb,
  TrendingUp,
  Activity,
  Clock,
  CheckCircle,
  ArrowRight
} from 'lucide-react';

export default function Workspace() {
  const [insights, setInsights] = useState<any>(null);
  const [activeProjects, setActiveProjects] = useState([
    {
      id: '1',
      title: 'Quantum-AI Integration Research',
      status: 'Active',
      progress: 78,
      team: ['Alice', 'Bob', 'Charlie'],
      priority: 'High',
      crossDomainConnections: ['news', 'analytics', 'automation'],
      significance: 9.2
    },
    {
      id: '2',
      title: 'Sustainable Tech Product Development',
      status: 'Planning',
      progress: 23,
      team: ['Diana', 'Eve'],
      priority: 'Medium',
      crossDomainConnections: ['product', 'social', 'ads'],
      significance: 8.8
    },
    {
      id: '3',
      title: 'Social Intelligence Analytics Platform',
      status: 'Active',
      progress: 56,
      team: ['Frank', 'Grace', 'Henry', 'Iris'],
      priority: 'High',
      crossDomainConnections: ['social', 'analytics', 'entertainment'],
      significance: 8.5
    }
  ]);

  const [aiSuggestions, setAiSuggestions] = useState([
    {
      type: 'optimization',
      title: 'Workflow Optimization Opportunity',
      description: 'AI detected a 34% efficiency gain possible by connecting your quantum research with automation tools',
      action: 'Optimize Now',
      significance: 8.7
    },
    {
      type: 'collaboration',
      title: 'Cross-Team Synergy Detected',
      description: 'Teams working on social analytics and entertainment could benefit from shared AI models',
      action: 'Connect Teams',
      significance: 8.3
    },
    {
      type: 'insight',
      title: 'Market Trend Alignment',
      description: 'Your sustainable tech project aligns with 3 trending market opportunities worth $2.4B',
      action: 'View Analysis',
      significance: 9.0
    }
  ]);

  useEffect(() => {
    loadInsights();
  }, []);

  const loadInsights = async () => {
    const data = await globalSearch.getInsights();
    setInsights(data);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-gray-950 to-blue-1000">
      <UnifiedHeader 
        title="Intelligent Workspace Hub"
        subtitle="Cross-domain collaboration • AI-mediated productivity • Real-time intelligence synthesis"
        activeFeatures={['ai', 'learning', 'optimized']}
      />

      <div className="p-6 space-y-6">
        {/* Intelligence Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card className="bg-gradient-to-br from-cyan-900/30 to-blue-900/30 border-cyan-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-cyan-300 flex items-center">
                <Brain className="w-4 h-4 mr-2" />
                AI Insights
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-cyan-400">127</div>
              <div className="text-xs text-cyan-300">Cross-domain connections active</div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-900/30 to-emerald-900/30 border-green-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-green-300 flex items-center">
                <TrendingUp className="w-4 h-4 mr-2" />
                Productivity
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">+340%</div>
              <div className="text-xs text-green-300">Above baseline with AI assistance</div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border-purple-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-purple-300 flex items-center">
                <Users className="w-4 h-4 mr-2" />
                Active Teams
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-400">12</div>
              <div className="text-xs text-purple-300">Collaborating globally</div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-orange-900/30 to-red-900/30 border-orange-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-orange-300 flex items-center">
                <Target className="w-4 h-4 mr-2" />
                Significance Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-400">8.9/10</div>
              <div className="text-xs text-orange-300">High-impact work focus</div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="projects" className="space-y-6">
          <TabsList className="bg-slate-800/50 border border-slate-700/50">
            <TabsTrigger value="projects" className="data-[state=active]:bg-cyan-500/20">
              <FileText className="w-4 h-4 mr-2" />
              Active Projects
            </TabsTrigger>
            <TabsTrigger value="ai-insights" className="data-[state=active]:bg-purple-500/20">
              <Brain className="w-4 h-4 mr-2" />
              AI Insights
            </TabsTrigger>
            <TabsTrigger value="collaboration" className="data-[state=active]:bg-green-500/20">
              <Users className="w-4 h-4 mr-2" />
              Team Intelligence
            </TabsTrigger>
          </TabsList>

          <TabsContent value="projects" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {activeProjects.map((project) => (
                <Card key={project.id} className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-slate-200">{project.title}</CardTitle>
                      <div className="flex items-center space-x-2">
                        <Badge variant="secondary" className={
                          project.priority === 'High' 
                            ? 'bg-red-500/20 text-red-300 border-red-500/30'
                            : 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
                        }>
                          {project.priority}
                        </Badge>
                        <Badge variant="secondary" className="bg-purple-500/20 text-purple-300 border-purple-500/30">
                          {project.significance.toFixed(1)}
                        </Badge>
                      </div>
                    </div>
                    <CardDescription>
                      Cross-domain connections: {project.crossDomainConnections.join(', ')}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm text-slate-400 mb-2">
                        <span>Progress</span>
                        <span>{project.progress}%</span>
                      </div>
                      <div className="w-full h-2 bg-slate-700 rounded-full">
                        <div 
                          className="h-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full transition-all duration-500"
                          style={{ width: `${project.progress}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Users className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">
                          {project.team.length} team members
                        </span>
                      </div>
                      <Button variant="outline" size="sm" className="border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10">
                        <ArrowRight className="w-4 h-4 mr-1" />
                        Open Project
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="ai-insights" className="space-y-6">
            <div className="space-y-4">
              {aiSuggestions.map((suggestion, index) => (
                <Card key={index} className="bg-gradient-to-r from-slate-800/50 to-purple-900/20 border-purple-500/30">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <Lightbulb className="w-5 h-5 text-purple-400" />
                          <h3 className="text-lg font-semibold text-purple-300">{suggestion.title}</h3>
                          <Badge variant="secondary" className="bg-purple-500/20 text-purple-300 border-purple-500/30">
                            {suggestion.significance.toFixed(1)}
                          </Badge>
                        </div>
                        <p className="text-slate-300 mb-3">{suggestion.description}</p>
                      </div>
                      <Button className="bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 ml-4">
                        {suggestion.action}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="collaboration" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-green-300 flex items-center">
                    <Globe className="w-5 h-5 mr-2" />
                    Global Team Activity
                  </CardTitle>
                  <CardDescription>Real-time collaboration across time zones</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {[
                      { name: 'Alice Chen', location: 'San Francisco', status: 'Active', task: 'Quantum algorithm optimization' },
                      { name: 'Bob Kumar', location: 'Mumbai', status: 'Active', task: 'AI model training' },
                      { name: 'Diana López', location: 'Barcelona', status: 'Break', task: 'Sustainability research' },
                      { name: 'Frank Wilson', location: 'New York', status: 'Active', task: 'Social platform integration' }
                    ].map((member, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            member.status === 'Active' ? 'bg-green-400' : 'bg-yellow-400'
                          }`}></div>
                          <div>
                            <div className="font-medium text-slate-200">{member.name}</div>
                            <div className="text-xs text-slate-400">{member.location} • {member.task}</div>
                          </div>
                        </div>
                        <Badge variant="secondary" className={
                          member.status === 'Active' 
                            ? 'bg-green-500/20 text-green-300 border-green-500/30'
                            : 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
                        }>
                          {member.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-cyan-300 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2" />
                    Cross-Domain Intelligence
                  </CardTitle>
                  <CardDescription>AI-detected collaboration opportunities</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {insights?.crossDomainInsights?.slice(0, 4).map((insight: any, index: number) => (
                      <div key={index} className="p-3 bg-slate-700/30 rounded-lg border border-slate-600/30">
                        <div className="flex items-center justify-between mb-2">
                          <div className="text-sm font-medium text-slate-200">
                            {insight.sourceType} → {insight.targetType}
                          </div>
                          <Badge variant="secondary" className="bg-cyan-500/20 text-cyan-300 border-cyan-500/30">
                            {insight.significance.toFixed(1)}
                          </Badge>
                        </div>
                        <div className="text-xs text-slate-400">{insight.connection}</div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
