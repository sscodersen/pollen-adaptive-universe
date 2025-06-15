
export const PLATFORM_CONFIG = {
  name: 'Pollen Intelligence Platform',
  tagline: 'Operating System for Digital Life',
  version: '2.0',
  
  // Performance settings
  updateIntervals: {
    metrics: 3000,
    activities: 8000,
    insights: 15000,
    search: 500
  },
  
  // UI Configuration
  ui: {
    animations: {
      fast: 150,
      normal: 300,
      slow: 500
    },
    colors: {
      primary: {
        cyan: '#06b6d4',
        purple: '#8b5cf6',
        green: '#10b981',
        orange: '#f97316',
        violet: '#7c3aed',
        pink: '#ec4899'
      },
      gradients: {
        primary: 'from-cyan-500 to-purple-500',
        intelligence: 'from-purple-500 to-pink-500',
        system: 'from-cyan-500 to-blue-500',
        success: 'from-green-500 to-emerald-500',
        warning: 'from-orange-500 to-red-500',
        analytics: 'from-violet-500 to-purple-500'
      },
      status: {
        optimal: '#10b981',
        good: '#06b6d4',
        warning: '#f59e0b',
        critical: '#ef4444'
      }
    }
  },
  
  // Feature flags
  features: {
    realTimeUpdates: true,
    crossDomainIntelligence: true,
    predictiveAnalytics: true,
    aiOptimization: true,
    globalSearch: true,
    intelligentAutomation: true
  },
  
  // Domain configuration
  domains: {
    intelligence: { 
      name: 'Intelligence Hub', 
      color: 'purple', 
      icon: 'Brain',
      description: 'AI insights & cross-domain intelligence'
    },
    dashboard: { 
      name: 'Intelligence Dashboard', 
      color: 'cyan', 
      icon: 'BarChart3',
      description: 'Real-time AI metrics & system health'
    },
    social: { 
      name: 'Social Intelligence', 
      color: 'orange', 
      icon: 'Users',
      description: 'AI-curated social insights'
    },
    entertainment: { 
      name: 'Content Studio', 
      color: 'purple', 
      icon: 'Play',
      description: 'AI-generated entertainment'
    },
    search: { 
      name: 'News Intelligence', 
      color: 'cyan', 
      icon: 'Search',
      description: 'Real-time news analysis'
    },
    shop: { 
      name: 'Smart Commerce', 
      color: 'green', 
      icon: 'ShoppingBag',
      description: 'Intelligent shopping assistant'
    },
    automation: { 
      name: 'Task Automation', 
      color: 'violet', 
      icon: 'Target',
      description: 'Automated workflow intelligence'
    },
    analytics: { 
      name: 'Deep Analytics', 
      color: 'cyan', 
      icon: 'BarChart3',
      description: 'Advanced insights dashboard'
    }
  }
};

export const getSystemStatus = (metrics: any) => {
  const healthScore = (metrics.systemHealth + metrics.intelligenceSynergy + metrics.learningVelocity) / 3;
  
  if (healthScore >= 95) return { status: 'optimal', color: PLATFORM_CONFIG.ui.colors.status.optimal };
  if (healthScore >= 85) return { status: 'good', color: PLATFORM_CONFIG.ui.colors.status.good };
  if (healthScore >= 70) return { status: 'warning', color: PLATFORM_CONFIG.ui.colors.status.warning };
  return { status: 'critical', color: PLATFORM_CONFIG.ui.colors.status.critical };
};

export const formatMetric = (value: number, unit: string, precision: number = 1): string => {
  if (unit === '' && value >= 1000000) {
    return `${(value / 1000000).toFixed(precision)}M`;
  }
  if (unit === '' && value >= 1000) {
    return `${(value / 1000).toFixed(precision)}K`;
  }
  if (unit === '%' || unit === '/10') {
    return value.toFixed(precision);
  }
  if (unit === 'ms' || unit === 'ops/s') {
    return Math.floor(value).toString();
  }
  return Math.floor(value).toLocaleString();
};
