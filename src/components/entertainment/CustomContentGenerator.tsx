
import React, { useState, useRef } from 'react';
import { Send, Mic, Sparkles, Zap, AlertCircle } from 'lucide-react';
import { useContentGeneration } from '../../hooks/useContentGeneration';

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
  const [isListening, setIsListening] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { error, clearError } = useContentGeneration();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (userPrompt.trim() && !isGenerating) {
      clearError();
      onGenerate();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      console.log('Speech recognition not supported');
      return;
    }

    setIsListening(!isListening);
    console.log('Voice input triggered');
  };

  const suggestionPrompts = [
    "Create an interactive story about time travel",
    "Generate ambient music for focus and productivity",
    "Design a mini puzzle game with unique mechanics",
    "Write a sci-fi adventure with player choices"
  ];

  const handleSuggestionClick = (suggestion: string) => {
    setUserPrompt(suggestion);
    inputRef.current?.focus();
  };

  return (
    <div className="bg-gray-900/30 rounded-xl border border-gray-700/50 p-6">
      <div className="flex items-center space-x-3 mb-4">
        <Sparkles className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold text-white">Generate Custom Content</h3>
        <div className="px-2 py-1 bg-cyan-500/10 text-cyan-400 rounded text-xs font-medium">
          AI-Powered
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-red-400" />
          <p className="text-red-300 text-sm">{error}</p>
          <button 
            onClick={clearError}
            className="ml-auto text-red-400 hover:text-red-300"
          >
            ×
          </button>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="flex space-x-3">
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Describe the entertainment content you want to create..."
              className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 transition-all"
              disabled={isGenerating}
              maxLength={200}
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-xs text-gray-500">
              {userPrompt.length}/200
            </div>
          </div>
          <button
            type="button"
            onClick={handleVoiceInput}
            className={`px-4 py-3 rounded-lg transition-all border border-gray-600/50 ${
              isListening 
                ? 'bg-red-500/20 text-red-400 border-red-500/30' 
                : 'bg-gray-700/50 hover:bg-gray-600/50 text-gray-300'
            }`}
            disabled={isGenerating}
            title="Voice input (experimental)"
          >
            <Mic className={`w-5 h-5 ${isListening ? 'animate-pulse' : ''}`} />
          </button>
          <button
            type="submit"
            disabled={!userPrompt.trim() || isGenerating}
            className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-all flex items-center space-x-2 font-medium"
          >
            {isGenerating ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                <span>Create</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Quick Suggestions */}
      <div className="space-y-2">
        <div className="flex items-center space-x-2 text-sm text-gray-400">
          <Zap className="w-4 h-4" />
          <span>Quick suggestions:</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {suggestionPrompts.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => handleSuggestionClick(suggestion)}
              disabled={isGenerating}
              className="px-3 py-1.5 bg-gray-800/40 hover:bg-gray-700/40 text-gray-300 text-sm rounded-lg transition-all border border-gray-700/30 hover:border-gray-600/50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
