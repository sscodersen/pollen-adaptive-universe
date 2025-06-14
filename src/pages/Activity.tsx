
import React from 'react';
import { Layout } from '../components/Layout';
import { AIModelManager } from '../components/AIModelManager';
import { useApp } from '../contexts/AppContext';
import { Sparkles, Clock, TrendingUp, Users, FileText, Image, Video, Code } from 'lucide-react';

const Activity = () => {
  const { state } = useApp();

  const getContentIcon = (type: string) => {
    switch (type) {
      case 'blog': return FileText;
      case 'email': return FileText;
      case 'marketing': return TrendingUp;
      case 'image': return Image;
      case 'video': return Video;
      case 'code': return Code;
      default: return Sparkles;
    }
  };

  const recentActivity = [
    {
      id: '1',
      type: 'generation',
      title: 'Blog post generated',
      description: 'AI Marketing Strategies for 2024',
      timestamp: '2 minutes ago',
      icon: FileText
    },
    {
      id: '2',
      type: 'analysis',
      title: 'File analyzed',
      description: 'quarterly-report.xlsx processed',
      timestamp: '5 minutes ago',
      icon: TrendingUp
    },
    {
      id: '3',
      type: 'evolution',
      title: 'Model evolved',
      description: 'Pollen adapted to your writing style',
      timestamp: '12 minutes ago',
      icon: Sparkles
    }
  ];

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Activity Dashboard</h1>
          <p className="text-slate-400">Monitor Pollen's evolution and your creative progress</p>
        </div>

        {/* AI Model Manager */}
        <div className="mb-8">
          <AIModelManager />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-8 h-8 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-cyan-400" />
              </div>
              <span className="text-slate-400 text-sm">Generated Content</span>
            </div>
            <div className="text-2xl font-bold">{state.generatedContent.length}</div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                <FileText className="w-4 h-4 text-purple-400" />
              </div>
              <span className="text-slate-400 text-sm">Files Processed</span>
            </div>
            <div className="text-2xl font-bold">{state.uploadedFiles.length}</div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <Users className="w-4 h-4 text-green-400" />
              </div>
              <span className="text-slate-400 text-sm">Conversations</span>
            </div>
            <div className="text-2xl font-bold">{state.conversations.length}</div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
            <div className="flex items-center space-x-3 mb-2">
              <div className="w-8 h-8 bg-yellow-500/20 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-yellow-400" />
              </div>
              <span className="text-slate-400 text-sm">Model Evolution</span>
            </div>
            <div className="text-2xl font-bold">Active</div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Activity */}
          <div className="lg:col-span-2">
            <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
            <div className="space-y-4">
              {recentActivity.map((activity) => (
                <div key={activity.id} className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-4">
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-lg flex items-center justify-center">
                      <activity.icon className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-white">{activity.title}</h3>
                      <p className="text-slate-400 text-sm">{activity.description}</p>
                      <div className="flex items-center space-x-2 mt-2">
                        <Clock className="w-3 h-3 text-slate-500" />
                        <span className="text-xs text-slate-500">{activity.timestamp}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Generated Content */}
              {state.generatedContent.slice(-3).map((content) => {
                const IconComponent = getContentIcon(content.type);
                return (
                  <div key={content.id} className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-4">
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-lg flex items-center justify-center">
                        <IconComponent className="w-4 h-4 text-cyan-400" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-medium text-white">Content Generated</h3>
                        <p className="text-slate-400 text-sm">{content.title}</p>
                        <div className="flex items-center space-x-2 mt-2">
                          <Clock className="w-3 h-3 text-slate-500" />
                          <span className="text-xs text-slate-500">
                            {new Date(content.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Pollen Status */}
          <div className="lg:col-span-1">
            <h2 className="text-xl font-semibold mb-4">Pollen Status</h2>
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
              <div className="text-center mb-4">
                <div className="w-16 h-16 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-semibold">Pollen LLMX</h3>
                <p className="text-sm text-slate-400">Self-Evolving AI</p>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Learning Progress</span>
                  <span className="text-sm text-cyan-400">Active</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2 rounded-full animate-pulse-glow" style={{width: '78%'}}></div>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Adaptation Level</span>
                  <span className="text-sm text-green-400">High</span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-400">Privacy Mode</span>
                  <span className="text-sm text-green-400">Enabled</span>
                </div>
              </div>

              <div className="mt-6 p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/20">
                <p className="text-xs text-cyan-300">
                  Pollen is continuously learning from your interactions while keeping all data private and secure.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Activity;
