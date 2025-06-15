
import React from 'react';
import { User, MessageSquare, Heart } from 'lucide-react';

interface ActivityPostProps {
  author: {
    name: string;
    avatar: React.ReactNode;
  };
  timestamp: string;
  children: React.ReactNode;
}

export const ActivityPost: React.FC<ActivityPostProps> = ({ author, timestamp, children }) => {
  return (
    <div className="bg-slate-800/50 rounded-xl p-4 md:p-5 border border-slate-700/50 animate-fade-in">
      <div className="flex items-center space-x-3 mb-4">
        {author.avatar}
        <div>
          <h3 className="font-semibold text-white">{author.name}</h3>
          <p className="text-xs text-slate-400">{timestamp}</p>
        </div>
      </div>
      <div className="text-slate-300 leading-relaxed space-y-3">
        {children}
      </div>
      <div className="mt-4 flex items-center space-x-4">
        <button className="flex items-center space-x-1.5 text-slate-400 hover:text-white transition-colors">
          <Heart className="w-4 h-4" />
          <span className="text-sm">Like</span>
        </button>
        <button className="flex items-center space-x-1.5 text-slate-400 hover:text-white transition-colors">
          <MessageSquare className="w-4 h-4" />
          <span className="text-sm">Comment</span>
        </button>
      </div>
    </div>
  )
}
