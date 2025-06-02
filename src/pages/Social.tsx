
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { Users, MessageCircle, Mic, Clock, UserCheck, Eye, EyeOff } from 'lucide-react';

const Social = () => {
  const [activeTab, setActiveTab] = useState('matches');
  const [isRecording, setIsRecording] = useState(false);

  const socialTabs = [
    { id: 'matches', name: 'Smart Matches', icon: Users },
    { id: 'conversations', name: 'Conversations', icon: MessageCircle },
    { id: 'voice', name: 'Voice Notes', icon: Mic }
  ];

  const matches = [
    { id: '1', compatibility: 94, interests: ['AI Research', 'Philosophy', 'Music'], lastSeen: '2h ago' },
    { id: '2', compatibility: 87, interests: ['Tech Innovation', 'Creative Writing'], lastSeen: '1d ago' }
  ];

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Anonymous Social Layer</h1>
          <p className="text-slate-400">AI-powered connections with privacy-first design</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="space-y-2 mb-6">
              {socialTabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-all ${
                    activeTab === tab.id
                      ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300'
                      : 'bg-slate-800/50 border border-slate-700/50 text-slate-300 hover:bg-slate-700/50'
                  }`}
                >
                  <tab.icon className="w-5 h-5" />
                  <span className="text-sm font-medium">{tab.name}</span>
                </button>
              ))}
            </div>

            {/* Privacy Status */}
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center space-x-2 mb-2">
                <EyeOff className="w-4 h-4 text-green-400" />
                <span className="text-sm font-medium text-green-400">Anonymous Mode</span>
              </div>
              <p className="text-xs text-slate-400">Your identity is protected with zero-knowledge architecture</p>
            </div>
          </div>

          {/* Content */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
              {activeTab === 'matches' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">AI-Curated Matches</h2>
                  <div className="space-y-4">
                    {matches.map((match) => (
                      <div key={match.id} className="bg-slate-700/30 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <div className="w-12 h-12 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center">
                              <Users className="w-6 h-6 text-white" />
                            </div>
                            <div>
                              <h3 className="font-medium">Anonymous User</h3>
                              <p className="text-sm text-slate-400">Last seen {match.lastSeen}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold text-cyan-400">{match.compatibility}%</div>
                            <div className="text-xs text-slate-400">Compatibility</div>
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {match.interests.map((interest) => (
                            <span key={interest} className="px-2 py-1 bg-slate-600/50 rounded text-xs">
                              {interest}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === 'conversations' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Active Conversations</h2>
                  <div className="space-y-4">
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <div className="flex items-center space-x-3 mb-2">
                        <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                        <span className="font-medium">Ephemeral Chat #1</span>
                        <Clock className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Expires in 2h</span>
                      </div>
                      <p className="text-sm text-slate-300">Discussing AI ethics and philosophy...</p>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'voice' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Voice Messages</h2>
                  <div className="text-center py-8">
                    <button
                      onClick={() => setIsRecording(!isRecording)}
                      className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                        isRecording 
                          ? 'bg-red-500 hover:bg-red-600 animate-pulse' 
                          : 'bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600'
                      }`}
                    >
                      <Mic className="w-8 h-8 text-white" />
                    </button>
                    <p className="mt-4 text-slate-300">
                      {isRecording ? 'Recording...' : 'Tap to record anonymous voice note'}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Social;
