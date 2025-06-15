
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { UnifiedHeader } from '../components/UnifiedHeader';
import { IntelligenceDashboardOptimized } from '../components/IntelligenceDashboardOptimized';
import { SmartProductSection } from '../components/shop/SmartProductSection';
import { PLATFORM_CONFIG } from '../lib/platformConfig';

const NavLink = ({ to, children }: { to: string, children: React.ReactNode }) => {
    const location = useLocation();
    const isActive = location.pathname === to;
    return (
        <Link to={to} className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
            isActive ? 'text-white bg-cyan-500/10' : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
        }`}>
            {children}
        </Link>
    )
}

const NewPlayground = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-gray-950 to-blue-1000 text-white">
      <UnifiedHeader 
        title={PLATFORM_CONFIG.name}
        subtitle={`${PLATFORM_CONFIG.tagline} • A Simplified, Intelligent Hub`}
        activeFeatures={['ai', 'learning', 'optimized']}
      />
      
      <div className="border-b border-gray-800/60 bg-gray-900/40 backdrop-blur-sm sticky top-0 z-40">
        <div className="flex space-x-4 px-6 py-2">
            <NavLink to="/">Dashboard</NavLink>
            <NavLink to="/search">News</NavLink>
            <NavLink to="/social">Social</NavLink>
            <NavLink to="/shop">Shop</NavLink>
        </div>
      </div>

      {/* Simplified Content Area */}
      <div className="p-6 space-y-8 animate-fade-in">
        <IntelligenceDashboardOptimized />
        <SmartProductSection />
      </div>
    </div>
  );
};

export default NewPlayground;
