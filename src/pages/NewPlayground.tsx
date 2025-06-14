
import React, { useState, useEffect, useMemo } from 'react';
import { usePlayground } from '../contexts/PlaygroundContext';
import { ActivityFeed } from '../components/ActivityFeed';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { CommunityHub } from '../components/CommunityHub';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { ErrorBoundary } from '../components/optimized/ErrorBoundary';
import { LoadingSpinner } from '../components/optimized/LoadingSpinner';

const NewPlayground = () => {
  const { activeTab } = usePlayground();
  const [activities, setActivities] = useState<any[]>([]);
  const [isInitializing, setIsInitializing] = useState(true);

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
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error('Platform initialization error:', error);
      } finally {
        setIsInitializing(false);
      }
    };

    initializePlatform();
  }, []);

  const renderedTabContent = useMemo(() => {
    const contentMap: { [key: string]: React.ReactNode } = {
      Social: <SocialFeed />,
      Entertainment: <EntertainmentHub />,
      Search: <NewsEngine />,
      Shop: <ShopHub />,
      Automation: <TaskAutomation />,
      Community: <CommunityHub activities={activities} />,
      Analytics: <AnalyticsDashboard />
    };

    return contentMap[activeTab] || <SocialFeed />;
  }, [activeTab, activities]);

  if (isInitializing) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" className="mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Initializing Pollen Intelligence</h2>
          <p className="text-gray-400">Setting up AI systems and content engines...</p>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      {renderedTabContent}
    </ErrorBoundary>
  );
};

export default NewPlayground;
