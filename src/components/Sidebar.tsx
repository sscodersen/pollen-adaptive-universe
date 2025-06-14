
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Sparkles, 
  Image, 
  FileText, 
  Calendar, 
  Play, 
  Search, 
  Users, 
  Code 
} from 'lucide-react';

const navigation = [
  { name: 'Activity', href: '/', icon: Home },
  { name: 'AI Playground', href: '/playground', icon: Sparkles },
  { name: 'Visual Studio', href: '/visual', icon: Image },
  { name: 'Text Engine', href: '/text', icon: FileText },
  { name: 'Task Executor', href: '/tasks', icon: Calendar },
  { name: 'Entertainment', href: '/entertainment', icon: Play },
  { name: 'Search & News', href: '/search', icon: Search },
  { name: 'Social Layer', href: '/social', icon: Users },
  { name: 'Code Studio', href: '/code', icon: Code },
];

export const Sidebar = () => {
  const location = useLocation();

  return (
    <div className="w-64 bg-slate-800/50 backdrop-blur-xl border-r border-slate-700/50">
      <div className="flex flex-col h-full">
        <div className="p-6 border-b border-slate-700/50">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Pollen</h1>
              <p className="text-sm text-slate-400">LLMX Platform</p>
            </div>
          </div>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-cyan-300 border border-cyan-500/30'
                    : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
                }`}
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium">{item.name}</span>
              </Link>
            );
          })}
        </nav>
        
        <div className="p-4 border-t border-slate-700/50">
          <div className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 p-3 rounded-lg border border-cyan-500/20">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-400">Evolving</span>
            </div>
            <p className="text-xs text-slate-300">
              Pollen LLMX is learning from your interactions and optimizing in real-time.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
