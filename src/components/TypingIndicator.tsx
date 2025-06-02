
import React from 'react';

export const TypingIndicator = () => {
  return (
    <div className="flex items-center space-x-2 p-3 bg-slate-700/50 rounded-lg">
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
      </div>
      <span className="text-sm text-slate-400">Pollen is thinking...</span>
    </div>
  );
};
