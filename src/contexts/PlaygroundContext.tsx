
import React, { createContext, useState, useContext, useMemo } from 'react';
import { Users, Play, Search, ShoppingBag, Bot, Globe, BarChart3, LucideIcon } from 'lucide-react';

interface Tab {
  id: string;
  name: string;
  icon: LucideIcon;
}

export const tabs: Tab[] = [
  { id: 'Social', name: 'Social Feed', icon: Users },
  { id: 'Entertainment', name: 'Entertainment', icon: Play },
  { id: 'Search', name: 'News Intelligence', icon: Search },
  { id: 'Shop', name: 'Smart Shopping', icon: ShoppingBag },
  { id: 'Automation', name: 'Task Automation', icon: Bot },
  { id: 'Community', name: 'Community Hub', icon: Globe },
  { id: 'Analytics', name: 'Analytics', icon: BarChart3 }
];

type TabId = typeof tabs[number]['id'];

interface PlaygroundContextType {
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;
  tabs: typeof tabs;
}

const PlaygroundContext = createContext<PlaygroundContextType | null>(null);

export const PlaygroundProvider = ({ children }: { children: React.ReactNode }) => {
  const [activeTab, setActiveTab] = useState<TabId>('Social');

  const value = useMemo(() => ({
    activeTab,
    setActiveTab,
    tabs
  }), [activeTab]);

  return (
    <PlaygroundContext.Provider value={value}>
      {children}
    </PlaygroundContext.Provider>
  );
};

export const usePlayground = () => {
  const context = useContext(PlaygroundContext);
  if (!context) {
    throw new Error('usePlayground must be used within a PlaygroundProvider');
  }
  return context;
};
