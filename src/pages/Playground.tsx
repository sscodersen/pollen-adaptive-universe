
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { Sparkles, Upload, FileText, Code, Brain, Zap } from 'lucide-react';

const Playground = () => {
  const [activeMode, setActiveMode] = useState('chat');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m Pollen LLMX. I evolve and adapt based on our interactions. What would you like to create, analyze, or explore today?'
    }
  ]);

  const modes = [
    { id: 'chat', name: 'Chat & Reasoning', icon: Brain },
    { id: 'code', name: 'Code Assistant', icon: Code },
    { id: 'analysis', name: 'File Analysis', icon: FileText },
    { id: 'creative', name: 'Creative Studio', icon: Sparkles }
  ];

  const handleSend = () => {
    if (!input.trim()) return;
    
    const newMessages = [
      ...messages,
      { role: 'user', content: input },
      { 
        role: 'assistant', 
        content: `Based on your input "${input}", I'm analyzing and adapting my response patterns. This interaction helps me evolve my understanding of your preferences. Here's my adaptive response...`
      }
    ];
    
    setMessages(newMessages);
    setInput('');
  };

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">AI Playground</h1>
          <p className="text-slate-400">Universal AI workspace powered by adaptive Pollen LLMX</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Mode Selector */}
          <div className="lg:col-span-1">
            <h2 className="text-lg font-semibold mb-4">Modes</h2>
            <div className="space-y-2">
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-all duration-200 ${
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
            
            <div className="mt-6 bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <h3 className="font-semibold mb-2 flex items-center space-x-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span>Quick Actions</span>
              </h3>
              <div className="space-y-2">
                <button className="w-full text-left text-sm text-slate-300 hover:text-white transition-colors">
                  Upload & Analyze File
                </button>
                <button className="w-full text-left text-sm text-slate-300 hover:text-white transition-colors">
                  Generate Code
                </button>
                <button className="w-full text-left text-sm text-slate-300 hover:text-white transition-colors">
                  Create Content
                </button>
                <button className="w-full text-left text-sm text-slate-300 hover:text-white transition-colors">
                  Debug & Fix
                </button>
              </div>
            </div>
          </div>

          {/* Main Chat Area */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 h-[600px] flex flex-col">
              {/* Chat Header */}
              <div className="p-4 border-b border-slate-700/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-lg flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Pollen LLMX</h3>
                      <p className="text-xs text-slate-400">Adaptive AI • Evolving from interactions</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-xs text-green-400">Learning</span>
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white'
                          : 'bg-slate-700/50 text-slate-200'
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                ))}
              </div>

              {/* Input Area */}
              <div className="p-4 border-t border-slate-700/50">
                <div className="flex space-x-3">
                  <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
                    <Upload className="w-5 h-5 text-slate-400" />
                  </button>
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                      placeholder="Ask Pollen anything or upload a file to analyze..."
                      className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg px-4 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                    />
                  </div>
                  <button
                    onClick={handleSend}
                    className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 px-6 py-2 rounded-lg font-medium transition-all duration-200"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Playground;
