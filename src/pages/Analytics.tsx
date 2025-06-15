
import React from 'react';
import { Layout } from '../components/Layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { 
  TrendingUp, 
  Users, 
  Activity, 
  Brain,
  BarChart3,
  PieChart,
  LineChart,
  Target
} from 'lucide-react';

export default function Analytics() {
  return (
    <Layout>
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Intelligence Analytics
            </h1>
            <p className="text-slate-400 mt-2">Platform-wide insights and performance metrics</p>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="border-green-500/30 text-green-300">
              <Activity className="w-3 h-3 mr-1" />
              Real-time
            </Badge>
            <Badge variant="outline" className="border-cyan-500/30 text-cyan-300">
              <Brain className="w-3 h-3 mr-1" />
              AI Insights
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-300">Active Users</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-cyan-400">24,891</div>
              <div className="flex items-center text-sm text-green-400 mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                +12.5% from last week
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-300">Significance Score</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-400">8.7/10</div>
              <div className="flex items-center text-sm text-green-400 mt-1">
                <Brain className="w-4 h-4 mr-1" />
                Optimal performance
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-300">Content Interactions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">156K</div>
              <div className="flex items-center text-sm text-cyan-400 mt-1">
                <Activity className="w-4 h-4 mr-1" />
                +8.2% daily growth
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-300">AI Learning Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-400">94.3%</div>
              <div className="flex items-center text-sm text-purple-400 mt-1">
                <Target className="w-4 h-4 mr-1" />
                High accuracy
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardHeader>
              <CardTitle className="text-cyan-300">Domain Performance</CardTitle>
              <CardDescription>Engagement across platform domains</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Entertainment</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-24 h-2 bg-slate-700 rounded-full">
                      <div className="w-20 h-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"></div>
                    </div>
                    <span className="text-purple-400">83%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">News & Search</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-24 h-2 bg-slate-700 rounded-full">
                      <div className="w-18 h-2 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"></div>
                    </div>
                    <span className="text-cyan-400">75%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Shopping</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-24 h-2 bg-slate-700 rounded-full">
                      <div className="w-16 h-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full"></div>
                    </div>
                    <span className="text-green-400">67%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Social</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-24 h-2 bg-slate-700 rounded-full">
                      <div className="w-14 h-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-full"></div>
                    </div>
                    <span className="text-orange-400">58%</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardHeader>
              <CardTitle className="text-purple-300">AI Learning Insights</CardTitle>
              <CardDescription>Platform intelligence evolution</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Brain className="w-5 h-5 text-cyan-400" />
                    <span className="text-slate-300">Pattern Recognition</span>
                  </div>
                  <Badge variant="secondary" className="bg-cyan-500/20 text-cyan-300">
                    Improving
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Target className="w-5 h-5 text-purple-400" />
                    <span className="text-slate-300">Content Relevance</span>
                  </div>
                  <Badge variant="secondary" className="bg-green-500/20 text-green-300">
                    Optimal
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Users className="w-5 h-5 text-green-400" />
                    <span className="text-slate-300">User Understanding</span>
                  </div>
                  <Badge variant="secondary" className="bg-purple-500/20 text-purple-300">
                    Excellent
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
