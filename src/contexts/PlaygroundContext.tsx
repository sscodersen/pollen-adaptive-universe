
import React, { createContext, useState, useContext, useMemo, ReactNode } from 'react';
import { Users, Play, Search, ShoppingBag, Bot, Globe, BarChart3, LucideIcon } from 'lucide-react';

interface Tab {
  id: string;
  name: string;
  icon: LucideIcon;
}

interface PlaygroundContextType {
  tabs: Tab[];
  activeTab: string;
  setActiveTab: (id: string) => void;
}

const PlaygroundContext = createContext<PlaygroundContextType | undefined>(undefined);

export const PlaygroundProvider = ({ children }: { children: ReactNode }) => {
  const tabs = useMemo(() => [
    { id: 'Social', name: 'Social Feed', icon: Users },
    { id: 'Entertainment', name: 'Entertainment', icon: Play },
    { id: 'Search', name: 'News Intelligence', icon: Search },
    { id: 'Shop', name: 'Smart Shopping', icon: ShoppingBag },
    { id: 'Automation', name: 'Task Automation', icon: Bot },
    { id: 'Community', name: 'Community Hub', icon: Globe },
    { id: 'Analytics', name: 'Analytics', icon: BarChart3 }
  ], []);

  const [activeTab, setActiveTab] = useState('Social');

  const value = {
    tabs,
    activeTab,
    setActiveTab,
  };

  return (
    <PlaygroundContext.Provider value={value}>
      {children}
    </PlaygroundContext.Provider>
  );
};

export const usePlayground = () => {
  const context = useContext(PlaygroundContext);
  if (context === undefined) {
    throw new Error('usePlayground must be used within a PlaygroundProvider');
  }
  return context;
};
