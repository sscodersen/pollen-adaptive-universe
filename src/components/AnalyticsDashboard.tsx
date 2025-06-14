
import React from 'react';
import { AiCoreStats } from './AiCoreStats';

export const AnalyticsDashboard = () => {
  return (
    <div className="p-6 md:p-10 h-full overflow-y-auto">
      <header className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight text-white">AI Analytics</h1>
        <p className="text-lg text-gray-400 mt-2">
          Monitoring the real-time performance and learning cycles of the Pollen Intelligence core.
        </p>
      </header>
      
      <div className="max-w-4xl mx-auto">
        <AiCoreStats />
      </div>
    </div>
  );
};
