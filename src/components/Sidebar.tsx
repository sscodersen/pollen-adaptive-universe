
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
  Code,
  Activity,
  TrendingUp,
  BarChart3,
  Briefcase,
  Target,
  Zap
} from 'lucide-react';

const navigation = [
  { name: 'Activity', href: '/', icon: Home, badge: null },
  { name: 'AI Playground', href: '/playground', icon: Sparkles, badge: 'AI' },
  { name: 'Visual Studio', href: '/visual', icon: Image, badge: null },
  { name: 'Text Engine', href: '/text', icon: FileText, badge: null },
  { name: 'Task Executor', href: '/tasks', icon: Calendar, badge: null },
  { name: 'Entertainment', href: '/entertainment', icon: Play, badge: 'NEW' },
  { name: 'Search & News', href: '/search', icon: Search, badge: null },
  { name: 'Social Layer', href: '/social', icon: Users, badge: null },
  { name: 'Code Studio', href: '/code', icon: Code, badge: null },
  { name: 'Analytics', href: '/analytics', icon: BarChart3, badge: 'PRO' },
  { name: 'Workspace', href: '/workspace', icon: Briefcase, badge: null },
  { name: 'Ad Builder', href: '/ads', icon: Target, badge: 'NEW' },
];

export const Sidebar = () => {
  const location = useLocation();

  return (
    <div className="w-64 bg-slate-800/60 backdrop-blur-xl border-r border-slate-700/50">
      <div className="flex flex-col h-full">
        <div className="p-6 border-b border-slate-700/50">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-xl flex items-center justify-center shadow-lg">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Pollen
              </h1>
              <p className="text-sm text-slate-400 font-medium">LLMX Platform</p>
            </div>
          </div>
        </div>
        
        <nav className="flex-1 p-4 space-y-1">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`group flex items-center justify-between px-3 py-2.5 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-cyan-300 border border-cyan-500/30 shadow-lg'
                    : 'text-slate-300 hover:bg-slate-700/50 hover:text-white border border-transparent'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <item.icon className={`w-5 h-5 transition-colors ${
                    isActive ? 'text-cyan-400' : 'group-hover:text-white'
                  }`} />
                  <span className="font-medium">{item.name}</span>
                </div>
                {item.badge && (
                  <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                    item.badge === 'AI' 
                      ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                      : item.badge === 'PRO'
                      ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                      : 'bg-green-500/20 text-green-300 border border-green-500/30'
                  }`}>
                    {item.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>
        
        <div className="p-4 border-t border-slate-700/50">
          <div className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 p-4 rounded-xl border border-cyan-500/20">
            <div className="flex items-center space-x-2 mb-3">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-semibold text-green-400">Evolving</span>
              <TrendingUp className="w-4 h-4 text-green-400 ml-auto" />
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Pollen LLMX is learning from your interactions and optimizing intelligence in real-time.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
