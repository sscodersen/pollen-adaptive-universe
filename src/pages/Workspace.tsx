
import React from 'react';
import { Layout } from '../components/Layout';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { 
  Briefcase, 
  Users, 
  FileText, 
  Calendar,
  CheckCircle,
  Clock,
  Brain,
  Zap,
  Target,
  TrendingUp
} from 'lucide-react';

export default function Workspace() {
  return (
    <Layout>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Intelligent Workspace
            </h1>
            <p className="text-slate-400 mt-2">AI-powered productivity and collaboration hub</p>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="border-cyan-500/30 text-cyan-300">
              <Brain className="w-3 h-3 mr-1" />
              Smart Assist
            </Badge>
            <Badge variant="outline" className="border-green-500/30 text-green-300">
              <Zap className="w-3 h-3 mr-1" />
              Auto-sync
            </Badge>
          </div>
        </div>

        <Tabs defaultValue="personal" className="space-y-6">
          <TabsList className="bg-slate-800/50 border border-slate-700/50">
            <TabsTrigger value="personal" className="data-[state=active]:bg-cyan-500/20">
              <Briefcase className="w-4 h-4 mr-2" />
              Personal
            </TabsTrigger>
            <TabsTrigger value="team" className="data-[state=active]:bg-purple-500/20">
              <Users className="w-4 h-4 mr-2" />
              Team
            </TabsTrigger>
            <TabsTrigger value="projects" className="data-[state=active]:bg-green-500/20">
              <Target className="w-4 h-4 mr-2" />
              Projects
            </TabsTrigger>
          </TabsList>

          <TabsContent value="personal" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-6">
                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-cyan-300">Today's Focus</CardTitle>
                    <CardDescription>AI-curated priority tasks</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-green-400" />
                      <span className="flex-1 text-slate-300">Review quarterly analytics report</span>
                      <Badge variant="secondary" className="bg-red-500/20 text-red-300">High</Badge>
                    </div>
                    <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
                      <Clock className="w-5 h-5 text-yellow-400" />
                      <span className="flex-1 text-slate-300">Prepare presentation for team meeting</span>
                      <Badge variant="secondary" className="bg-yellow-500/20 text-yellow-300">Medium</Badge>
                    </div>
                    <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
                      <Clock className="w-5 h-5 text-blue-400" />
                      <span className="flex-1 text-slate-300">Update project documentation</span>
                      <Badge variant="secondary" className="bg-blue-500/20 text-blue-300">Low</Badge>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-purple-300">Smart Insights</CardTitle>
                    <CardDescription>AI-generated productivity recommendations</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="p-4 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 rounded-lg border border-purple-500/20">
                      <div className="flex items-center space-x-2 mb-2">
                        <Brain className="w-5 h-5 text-purple-400" />
                        <span className="font-medium text-purple-300">Productivity Pattern</span>
                      </div>
                      <p className="text-slate-300 text-sm">
                        Your focus is 40% higher during 9-11 AM. Consider scheduling complex tasks during this window.
                      </p>
                    </div>
                    <div className="p-4 bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-lg border border-green-500/20">
                      <div className="flex items-center space-x-2 mb-2">
                        <TrendingUp className="w-5 h-5 text-green-400" />
                        <span className="font-medium text-green-300">Workflow Optimization</span>
                      </div>
                      <p className="text-slate-300 text-sm">
                        Batching similar tasks could save you 2.5 hours per week based on your activity patterns.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-6">
                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-green-300">Quick Actions</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button className="w-full justify-start bg-slate-700/50 hover:bg-slate-700">
                      <FileText className="w-4 h-4 mr-2" />
                      New Document
                    </Button>
                    <Button className="w-full justify-start bg-slate-700/50 hover:bg-slate-700">
                      <Calendar className="w-4 h-4 mr-2" />
                      Schedule Meeting
                    </Button>
                    <Button className="w-full justify-start bg-slate-700/50 hover:bg-slate-700">
                      <Target className="w-4 h-4 mr-2" />
                      Create Task
                    </Button>
                    <Button className="w-full justify-start bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600">
                      <Brain className="w-4 h-4 mr-2" />
                      AI Assistant
                    </Button>
                  </CardContent>
                </Card>

                <Card className="bg-slate-800/50 border-slate-700/50">
                  <CardHeader>
                    <CardTitle className="text-orange-300">This Week</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-cyan-400">73%</div>
                      <div className="text-sm text-slate-400">Tasks Completed</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-400">28h</div>
                      <div className="text-sm text-slate-400">Focused Time</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400">9.2</div>
                      <div className="text-sm text-slate-400">Productivity Score</div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="team" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-purple-300">Team Activity</CardTitle>
                  <CardDescription>Recent team collaboration highlights</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
                    <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center">
                      <span className="text-white text-sm font-bold">A</span>
                    </div>
                    <div className="flex-1">
                      <div className="text-slate-300">Alice completed design review</div>
                      <div className="text-xs text-slate-400">2 hours ago</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
                    <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-400 rounded-full flex items-center justify-center">
                      <span className="text-white text-sm font-bold">B</span>
                    </div>
                    <div className="flex-1">
                      <div className="text-slate-300">Bob updated project timeline</div>
                      <div className="text-xs text-slate-400">4 hours ago</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-cyan-300">Shared Goals</CardTitle>
                  <CardDescription>Team objectives and progress</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-slate-300">Q4 Product Launch</span>
                      <span className="text-green-400">85%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-700 rounded-full">
                      <div className="w-4/5 h-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full"></div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-slate-300">User Research</span>
                      <span className="text-yellow-400">60%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-700 rounded-full">
                      <div className="w-3/5 h-2 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full"></div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="projects" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-cyan-300">Platform Redesign</CardTitle>
                  <CardDescription>UI/UX overhaul project</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Progress</span>
                      <span className="text-cyan-400">72%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-700 rounded-full">
                      <div className="w-3/4 h-2 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"></div>
                    </div>
                    <Badge variant="secondary" className="bg-green-500/20 text-green-300">On Track</Badge>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-purple-300">AI Integration</CardTitle>
                  <CardDescription>Smart features implementation</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Progress</span>
                      <span className="text-purple-400">45%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-700 rounded-full">
                      <div className="w-2/5 h-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"></div>
                    </div>
                    <Badge variant="secondary" className="bg-yellow-500/20 text-yellow-300">In Progress</Badge>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-green-300">Analytics Dashboard</CardTitle>
                  <CardDescription>Data visualization tools</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Progress</span>
                      <span className="text-green-400">90%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-700 rounded-full">
                      <div className="w-11/12 h-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full"></div>
                    </div>
                    <Badge variant="secondary" className="bg-blue-500/20 text-blue-300">Review</Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
