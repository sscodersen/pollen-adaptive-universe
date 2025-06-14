
import React from 'react';
import { Brain, Zap, TrendingUp, Database } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

export const PollenMemoryPanel = () => {
  const memoryStats = pollenAI.getMemoryStats();

  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-xl border border-white/10 p-6">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
          <Brain className="w-4 h-4 text-white" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Memory Core</h3>
          <p className="text-sm text-white/60">Learning & adaptation engine</p>
        </div>
      </div>

      {/* Memory Stats */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-white/60">Short-term</span>
          </div>
          <div className="text-lg font-bold text-white">{memoryStats.shortTermSize}</div>
          <div className="text-xs text-white/40">Active memories</div>
        </div>

        <div className="bg-white/5 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Database className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-white/60">Patterns</span>
          </div>
          <div className="text-lg font-bold text-white">{memoryStats.longTermPatterns}</div>
          <div className="text-xs text-white/40">Learned patterns</div>
        </div>
      </div>

      {/* Learning Status */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-white/60">Learning Status</span>
          <div className={`px-2 py-1 rounded-full text-xs font-medium ${
            memoryStats.isLearning 
              ? 'bg-green-500/20 text-green-400' 
              : 'bg-red-500/20 text-red-400'
          }`}>
            {memoryStats.isLearning ? 'Active' : 'Paused'}
          </div>
        </div>
        <div className="w-full bg-white/10 rounded-full h-2">
          <div 
            className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(100, (memoryStats.shortTermSize / 100) * 100)}%` }}
          />
        </div>
      </div>

      {/* Top Patterns */}
      <div>
        <h4 className="text-sm font-medium text-white/80 mb-3">Top Learning Patterns</h4>
        <div className="space-y-2">
          {memoryStats.topPatterns.slice(0, 5).map((pattern, index) => (
            <div key={index} className="flex items-center justify-between py-2 px-3 bg-white/5 rounded-lg">
              <span className="text-sm text-white/70 font-mono">{pattern.pattern}</span>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-white/40">{pattern.category}</span>
                <div className="w-12 bg-white/10 rounded-full h-1">
                  <div 
                    className="bg-gradient-to-r from-cyan-400 to-purple-400 h-1 rounded-full"
                    style={{ width: `${Math.min(100, (pattern.weight / 5) * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
