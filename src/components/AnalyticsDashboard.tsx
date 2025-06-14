import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Users, Eye, Heart, MessageCircle, Zap, Globe, Cpu, Sparkles, History } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

interface AnalyticsData {
  engagement: Array<{ time: string; likes: number; comments: number; views: number }>;
  contentTypes: Array<{ name: string; value: number; color: string }>;
  userActivity: Array<{ day: string; active: number; new: number }>;
  performance: {
    totalViews: number;
    totalEngagement: number;
    activeUsers: number;
    contentGenerated: number;
  };
  modelInsights: {
    modelConfidence: number;
    learningRate: number;
    dailyAdaptations: number;
    reasoningTasks: number;
  };
}

export const AnalyticsDashboard = () => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [timeRange, setTimeRange] = useState('7d');

  useEffect(() => {
    generateAnalyticsData();
  }, [timeRange]);

  const generateAnalyticsData = () => {
    // Simulate analytics data
    const engagement = Array.from({ length: 24 }, (_, i) => ({
      time: `${i}:00`,
      likes: Math.floor(Math.random() * 1000) + 500,
      comments: Math.floor(Math.random() * 200) + 50,
      views: Math.floor(Math.random() * 5000) + 2000
    }));

    const contentTypes = [
      { name: 'Social Posts', value: 45, color: '#8B5CF6' },
      { name: 'News Articles', value: 25, color: '#06B6D4' },
      { name: 'Entertainment', value: 20, color: '#F59E0B' },
      { name: 'Tasks/Automation', value: 10, color: '#10B981' }
    ];

    const userActivity = Array.from({ length: 7 }, (_, i) => ({
      day: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i],
      active: Math.floor(Math.random() * 1000) + 800,
      new: Math.floor(Math.random() * 100) + 50
    }));

    const performance = {
      totalViews: 1247532,
      totalEngagement: 89234,
      activeUsers: 12847,
      contentGenerated: 5672
    };

    const modelInsights = {
      modelConfidence: 0.964,
      learningRate: 0.0015,
      dailyAdaptations: 42,
      reasoningTasks: 1823,
    };

    setAnalyticsData({ engagement, contentTypes, userActivity, performance, modelInsights });
  };

  if (!analyticsData) {
    return (
      <div className="p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Analytics Dashboard</h1>
          <p className="text-gray-400">Real-time insights into platform performance</p>
        </div>
        <div className="flex space-x-2">
          {['24h', '7d', '30d', '90d'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                timeRange === range
                  ? 'bg-cyan-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-3">
            <Eye className="w-8 h-8 text-blue-400" />
            <div>
              <p className="text-sm text-gray-400">Total Views</p>
              <p className="text-2xl font-bold text-white">{analyticsData.performance.totalViews.toLocaleString()}</p>
              <p className="text-xs text-green-400">↗ +12.5%</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-3">
            <Heart className="w-8 h-8 text-red-400" />
            <div>
              <p className="text-sm text-gray-400">Engagement</p>
              <p className="text-2xl font-bold text-white">{analyticsData.performance.totalEngagement.toLocaleString()}</p>
              <p className="text-xs text-green-400">↗ +8.3%</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-3">
            <Users className="w-8 h-8 text-green-400" />
            <div>
              <p className="text-sm text-gray-400">Active Users</p>
              <p className="text-2xl font-bold text-white">{analyticsData.performance.activeUsers.toLocaleString()}</p>
              <p className="text-xs text-green-400">↗ +15.7%</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-3">
            <Zap className="w-8 h-8 text-yellow-400" />
            <div>
              <p className="text-sm text-gray-400">Content Generated</p>
              <p className="text-2xl font-bold text-white">{analyticsData.performance.contentGenerated.toLocaleString()}</p>
              <p className="text-xs text-green-400">↗ +24.1%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Engagement Trends */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            <span>Engagement Trends</span>
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={analyticsData.engagement}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#F3F4F6'
                }}
              />
              <Line type="monotone" dataKey="views" stroke="#06B6D4" strokeWidth={2} />
              <Line type="monotone" dataKey="likes" stroke="#EF4444" strokeWidth={2} />
              <Line type="monotone" dataKey="comments" stroke="#10B981" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Content Distribution */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            <span>Content Distribution</span>
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={analyticsData.contentTypes}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {analyticsData.contentTypes.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#F3F4F6'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 mt-4">
            {analyticsData.contentTypes.map((item, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: item.color }}
                ></div>
                <span className="text-sm text-gray-300">{item.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* User Activity */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
          <Globe className="w-5 h-5 text-green-400" />
          <span>User Activity</span>
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={analyticsData.userActivity}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="day" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F3F4F6'
              }}
            />
            <Bar dataKey="active" fill="#06B6D4" radius={[4, 4, 0, 0]} />
            <Bar dataKey="new" fill="#10B981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* AI Model Insights */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-purple-400" />
          <span>AI Model Insights</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
            <div className="flex items-center space-x-3">
              <TrendingUp className="w-7 h-7 text-cyan-400" />
              <div>
                <p className="text-sm text-gray-400">Model Confidence</p>
                <p className="text-2xl font-bold text-white">{(analyticsData.modelInsights.modelConfidence * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
            <div className="flex items-center space-x-3">
              <Sparkles className="w-7 h-7 text-yellow-400" />
              <div>
                <p className="text-sm text-gray-400">Learning Rate</p>
                <p className="text-2xl font-bold text-white">{analyticsData.modelInsights.learningRate}</p>
              </div>
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
            <div className="flex items-center space-x-3">
              <History className="w-7 h-7 text-green-400" />
              <div>
                <p className="text-sm text-gray-400">Daily Adaptations</p>
                <p className="text-2xl font-bold text-white">{analyticsData.modelInsights.dailyAdaptations}</p>
              </div>
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
            <div className="flex items-center space-x-3">
              <Zap className="w-7 h-7 text-red-400" />
              <div>
                <p className="text-sm text-gray-400">Reasoning Tasks</p>
                <p className="text-2xl font-bold text-white">{analyticsData.modelInsights.reasoningTasks.toLocaleString()}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
