
import React from 'react';
import { ChevronLeft } from 'lucide-react';

const tabs = ["All Workspace", "Personal", "Team", "Community"];

interface FeedHeaderProps {
  title: string;
}

export const FeedHeader: React.FC<FeedHeaderProps> = ({ title }) => {
  const [activeTab, setActiveTab] = React.useState("Personal");

  return (
    <div className="p-4 md:p-6 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10 border-b border-slate-800">
      <div className="flex items-center mb-4">
        <button className="p-2 mr-2 rounded-full hover:bg-slate-800">
          <ChevronLeft className="w-5 h-5" />
        </button>
        <h1 className="text-xl font-bold text-white">{title}</h1>
      </div>
      <div className="flex space-x-2">
        {tabs.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
              activeTab === tab
                ? 'bg-white text-slate-900'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            {tab}
            {tab === 'Team' && <span className="ml-1.5 bg-red-500 text-white text-xs font-bold px-1.5 py-0.5 rounded-full">3</span>}
          </button>
        ))}
      </div>
    </div>
  );
}
