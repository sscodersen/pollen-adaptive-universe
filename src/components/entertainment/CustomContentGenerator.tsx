
import React, { useState } from 'react';
import { Send, Mic, Sparkles } from 'lucide-react';

interface CustomContentGeneratorProps {
  userPrompt: string;
  setUserPrompt: (prompt: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
}

export const CustomContentGenerator = ({ 
  userPrompt, 
  setUserPrompt, 
  onGenerate, 
  isGenerating 
}: CustomContentGeneratorProps) => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onGenerate();
  };

  return (
    <div className="bg-gray-900/30 rounded-xl border border-gray-700/50 p-6">
      <div className="flex items-center space-x-3 mb-4">
        <Sparkles className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold text-white">Generate Custom Content</h3>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="flex space-x-3">
          <div className="flex-1 relative">
            <input
              type="text"
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              placeholder="Describe the entertainment content you want..."
              className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20"
              disabled={isGenerating}
            />
          </div>
          <button
            type="button"
            className="px-4 py-3 bg-gray-700/50 hover:bg-gray-600/50 text-gray-300 rounded-lg transition-colors border border-gray-600/50"
            disabled={isGenerating}
          >
            <Mic className="w-5 h-5" />
          </button>
          <button
            type="submit"
            disabled={!userPrompt.trim() || isGenerating}
            className="px-6 py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            {isGenerating ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Generate</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
