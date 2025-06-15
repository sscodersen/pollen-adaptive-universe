
import React from 'react';
import { IntelligenceDashboardOptimized } from '../components/IntelligenceDashboardOptimized';
import InsightFeed from '../components/InsightFeed';

const NewPlayground = () => {
  return (
    <div className="p-4 md:p-6 grid grid-cols-1 xl:grid-cols-12 gap-6 h-full">
      <div className="xl:col-span-8">
        <IntelligenceDashboardOptimized />
      </div>
      <div className="xl:col-span-4 h-full">
        <InsightFeed />
      </div>
    </div>
  );
};

export default NewPlayground;
