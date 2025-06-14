
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { Code as CodeIcon, Terminal, GitBranch, Bot, Play, Save } from 'lucide-react';

const Code = () => {
  const [activeTab, setActiveTab] = useState('editor');
  const [code, setCode] = useState(`// Welcome to Pollen Code Studio
// AI-powered development environment

function generateAI() {
  console.log("Pollen AI is thinking...");
  return "Hello from the future!";
}

generateAI();`);

  const codeTabs = [
    { id: 'editor', name: 'AI Code Editor', icon: CodeIcon },
    { id: 'terminal', name: 'Smart Terminal', icon: Terminal },
    { id: 'version', name: 'Version Control', icon: GitBranch }
  ];

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Code Studio</h1>
          <p className="text-slate-400">AI-powered development environment with intelligent assistance</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="space-y-2">
              {codeTabs.map((tab) => (
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

            {/* AI Assistant */}
            <div className="mt-6 bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center space-x-2 mb-2">
                <Bot className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-medium text-cyan-400">AI Assistant</span>
              </div>
              <p className="text-xs text-slate-400">Ready to help with code suggestions, debugging, and optimization</p>
            </div>
          </div>

          {/* Content */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
              {activeTab === 'editor' && (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold">AI Code Editor</h2>
                    <div className="flex space-x-2">
                      <button className="bg-slate-700 hover:bg-slate-600 px-3 py-1 rounded-lg text-sm transition-colors flex items-center space-x-1">
                        <Play className="w-4 h-4" />
                        <span>Run</span>
                      </button>
                      <button className="bg-slate-700 hover:bg-slate-600 px-3 py-1 rounded-lg text-sm transition-colors flex items-center space-x-1">
                        <Save className="w-4 h-4" />
                        <span>Save</span>
                      </button>
                    </div>
                  </div>
                  <textarea
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    className="w-full h-96 bg-slate-900/50 border border-slate-600/30 rounded-lg p-4 text-white font-mono text-sm resize-none"
                    spellCheck={false}
                  />
                </div>
              )}

              {activeTab === 'terminal' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Smart Terminal</h2>
                  <div className="bg-slate-900/50 border border-slate-600/30 rounded-lg p-4 h-96 font-mono text-sm">
                    <div className="text-green-400">$ pollen --status</div>
                    <div className="text-slate-300">Pollen AI Model: Active</div>
                    <div className="text-slate-300">Docker Container: Running</div>
                    <div className="text-slate-300">Memory Usage: 2.1GB / 8GB</div>
                    <div className="text-slate-300">Processing Power: 87%</div>
                    <div className="text-green-400 mt-2">$ _</div>
                  </div>
                </div>
              )}

              {activeTab === 'version' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Version Control</h2>
                  <div className="space-y-4">
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <h3 className="font-medium mb-2">Latest Commits</h3>
                      <div className="space-y-2">
                        <div className="text-sm">
                          <span className="text-cyan-400">feat:</span> Add AI-powered code suggestions
                        </div>
                        <div className="text-sm">
                          <span className="text-green-400">fix:</span> Improve performance optimization
                        </div>
                      </div>
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

export default Code;
