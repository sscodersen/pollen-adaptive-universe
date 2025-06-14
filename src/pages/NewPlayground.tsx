import React, { useState, useEffect, useMemo } from 'react';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { CommunityHub } from '../components/CommunityHub';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { ErrorBoundary } from '../components/optimized/ErrorBoundary';
import { LoadingSpinner } from '../components/optimized/LoadingSpinner';
import { pollenAI } from '../services/pollenAI';
import { usePlayground } from '../contexts/PlaygroundContext';
import { SidebarTrigger } from '../components/ui/sidebar';

const NewPlayground = () => {
  const { activeTab, tabs } = usePlayground();

  const [activities, setActivities] = useState([]);
  const [isInitializing, setIsInitializing] = useState(true);
  const [aiStatus, setAiStatus] = useState('ready');
  const [systemStats, setSystemStats] = useState({
    totalContent: 5847,
    highSignificance: 892,
    activeFeeds: 8,
    processingSpeed: 167
  });

  useEffect(() => {
    const initializePlatform = async () => {
      setIsInitializing(true);
      
      const initialActivities = [
        {
          id: '1',
          type: 'ai_insight',
          user: {
            name: 'Pollen Intelligence',
            avatar: 'bg-gradient-to-r from-cyan-500 to-purple-500',
            initial: 'P'
          },
          action: 'analyzed global content patterns and generated',
          target: 'high-significance insights across all domains',
          content: "Successfully processed 12,847 content sources and identified 347 high-impact developments using the 7-factor significance algorithm. Real-time analysis shows 97% accuracy in trend prediction and content relevance scoring.",
          timestamp: '2m',
          aiGenerated: true,
          confidence: 0.97
        },
        {
          id: '2',
          type: 'system',
          user: {
            name: 'Content Engine',
            avatar: 'bg-gradient-to-r from-green-500 to-blue-500',
            initial: 'C'
          },
          action: 'optimized content generation algorithms with',
          target: 'multi-domain significance scoring',
          content: "Platform algorithms now generate personalized content across social, news, entertainment, and shopping domains with 95% relevance accuracy. New significance scoring ensures only high-impact content reaches users.",
          timestamp: '15m'
        }
      ];
      
      setActivities(initialActivities);
      
      try {
        // Simulate async initialization
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error('Platform initialization error:', error);
      } finally {
        setIsInitializing(false);
      }
    };

    initializePlatform();
  }, []);

  useEffect(() => {
    const updateSystemStatus = () => {
      const stats = pollenAI.getMemoryStats();
      setAiStatus(stats.isLearning ? 'learning' : 'ready');
    };

    const updateSystemStats = () => {
      setSystemStats(prev => ({
        totalContent: prev.totalContent + Math.floor(Math.random() * 10) - 5,
        highSignificance: prev.highSignificance + Math.floor(Math.random() * 4) - 2,
        activeFeeds: Math.max(6, Math.min(12, prev.activeFeeds + Math.floor(Math.random() * 3) - 1)),
        processingSpeed: Math.max(120, Math.min(200, prev.processingSpeed + Math.floor(Math.random() * 20) - 10))
      }));
    };

    const statusInterval = setInterval(updateSystemStatus, 3000);
    const statsInterval = setInterval(updateSystemStats, 8000);

    return () => {
      clearInterval(statusInterval);
      clearInterval(statsInterval);
    };
  }, []);

  const renderedTabContent = useMemo(() => {
    const contentMap = {
      Social: <SocialFeed />,
      Entertainment: <EntertainmentHub />,
      Search: <NewsEngine />,
      Shop: <ShopHub />,
      Automation: <TaskAutomation />,
      Community: <CommunityHub activities={activities} />,
      Analytics: <AnalyticsDashboard />
    };

    return contentMap[activeTab as keyof typeof contentMap] || <SocialFeed />;
  }, [activeTab, activities]);

  if (isInitializing) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" className="mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Initializing Pollen Intelligence</h2>
          <p className="text-gray-400">Setting up AI systems and content engines...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-slate-950 text-white">
      <header className="h-16 flex-shrink-0 px-6 flex items-center justify-between border-b border-slate-800/60">
        <div className="flex items-center gap-4">
          <SidebarTrigger />
          <h1 className="text-xl font-semibold">
            {tabs.find(t => t.id === activeTab)?.name}
          </h1>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-full text-xs border transition-all ${
            aiStatus === 'learning' 
              ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' 
              : 'bg-green-500/10 text-green-400 border-green-500/20'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              aiStatus === 'learning' ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'
            }`}></div>
            <span>AI {aiStatus === 'learning' ? 'Learning' : 'Active'}</span>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-auto">
        <ErrorBoundary>
          {renderedTabContent}
        </ErrorBoundary>
      </div>
    </div>
  );
};

export default NewPlayground;
