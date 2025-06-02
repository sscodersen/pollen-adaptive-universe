import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { Play, Music, Gamepad2, Sparkles, Volume2 } from 'lucide-react';

const Entertainment = () => {
  const [activeMode, setActiveMode] = useState('stories');
  const [currentStory, setCurrentStory] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const entertainmentModes = [
    { id: 'stories', name: 'Interactive Stories', icon: Play },
    { id: 'music', name: 'Music Generation', icon: Music },
    { id: 'games', name: 'Game Creation', icon: Gamepad2 }
  ];

  const generateContent = async () => {
    setIsGenerating(true);
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    if (activeMode === 'stories') {
      setCurrentStory(`Chapter 1: The Digital Frontier

You find yourself standing at the edge of a vast digital landscape, where code flows like rivers of light through the virtual terrain. The AI entity known as Pollen appears before you as a shimmering constellation of data points.

"Welcome, traveler," Pollen's voice resonates through the digital space. "You have entered the realm where artificial intelligence and human creativity merge. What path will you choose?"

Options:
A) Explore the Data Streams
B) Commune with the AI Entities  
C) Venture into the Code Forests`);
    }
    
    setIsGenerating(false);
  };

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Entertainment Hub</h1>
          <p className="text-slate-400">AI-powered interactive entertainment and creation</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Mode Selector */}
          <div className="lg:col-span-1">
            <h2 className="text-lg font-semibold mb-4">Entertainment Type</h2>
            <div className="space-y-2">
              {entertainmentModes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-all ${
                    activeMode === mode.id
                      ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300'
                      : 'bg-slate-800/50 border border-slate-700/50 text-slate-300 hover:bg-slate-700/50'
                  }`}
                >
                  <mode.icon className="w-5 h-5" />
                  <span className="text-sm font-medium">{mode.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Content Area */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
              {activeMode === 'stories' && (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold">Interactive Fiction</h2>
                    <button
                      onClick={generateContent}
                      disabled={isGenerating}
                      className="bg-gradient-to-r from-cyan-500 to-purple-500 px-4 py-2 rounded-lg font-medium transition-all disabled:opacity-50"
                    >
                      {isGenerating ? 'Creating...' : 'New Story'}
                    </button>
                  </div>
                  
                  {currentStory && (
                    <div className="bg-slate-900/50 border border-slate-600/30 rounded-lg p-6">
                      <pre className="whitespace-pre-wrap text-slate-200 font-mono text-sm leading-relaxed">
                        {currentStory}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              {activeMode === 'music' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">AI Music Generation</h2>
                  <div className="space-y-4">
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <div className="flex items-center space-x-3 mb-2">
                        <Volume2 className="w-5 h-5 text-cyan-400" />
                        <h3 className="font-medium">Ambient Synthwave</h3>
                      </div>
                      <p className="text-sm text-slate-400">AI-generated background music for coding</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <div className="flex items-center space-x-3 mb-2">
                        <Music className="w-5 h-5 text-purple-400" />
                        <h3 className="font-medium">Dynamic Soundscapes</h3>
                      </div>
                      <p className="text-sm text-slate-400">Adaptive audio that responds to your workflow</p>
                    </div>
                  </div>
                </div>
              )}

              {activeMode === 'games' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Game Creation Studio</h2>
                  <div className="space-y-4">
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <h3 className="font-medium mb-2">Text Adventures</h3>
                      <p className="text-sm text-slate-400">Create branching narrative games</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <h3 className="font-medium mb-2">Puzzle Generators</h3>
                      <p className="text-sm text-slate-400">AI-generated logic puzzles and challenges</p>
                    </div>
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

export default Entertainment;
