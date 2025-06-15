
export const platformTheme = {
  colors: {
    primary: {
      cyan: 'from-cyan-500 to-blue-500',
      purple: 'from-purple-500 to-pink-500',
      green: 'from-green-500 to-emerald-500',
      orange: 'from-orange-500 to-red-500',
      violet: 'from-violet-500 to-purple-500',
      yellow: 'from-yellow-500 to-orange-500'
    },
    status: {
      excellent: 'text-green-400',
      good: 'text-cyan-400',
      warning: 'text-yellow-400',
      critical: 'text-red-400'
    }
  },
  animations: {
    pulse: 'animate-pulse',
    fadeIn: 'animate-fade-in',
    bounce: 'animate-bounce'
  },
  domains: {
    intelligence: { color: 'purple', icon: 'Brain' },
    system: { color: 'cyan', icon: 'Zap' },
    social: { color: 'orange', icon: 'Users' },
    entertainment: { color: 'purple', icon: 'Play' },
    news: { color: 'cyan', icon: 'Search' },
    commerce: { color: 'green', icon: 'ShoppingBag' },
    automation: { color: 'violet', icon: 'Target' },
    analytics: { color: 'cyan', icon: 'BarChart3' },
    workspace: { color: 'green', icon: 'Briefcase' }
  }
};

export const getDomainColor = (domain: string) => {
  const domainInfo = platformTheme.domains[domain as keyof typeof platformTheme.domains];
  return domainInfo ? platformTheme.colors.primary[domainInfo.color as keyof typeof platformTheme.colors.primary] : platformTheme.colors.primary.cyan;
};

export const formatMetricValue = (value: number, unit: string) => {
  if (unit === '' && value > 1000) {
    return `${(value / 1000).toFixed(1)}K`;
  }
  if (unit === '/10' || unit === '%') {
    return value.toFixed(1);
  }
  if (unit === 'ops/s') {
    return Math.floor(value).toString();
  }
  return Math.floor(value).toLocaleString();
};
