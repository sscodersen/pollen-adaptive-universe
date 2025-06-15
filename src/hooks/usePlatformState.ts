
import { useState, useEffect } from 'react';

interface PlatformMetrics {
  crossDomainConnections: number;
  significanceScore: number;
  globalUsers: number;
  aiOptimizations: number;
  intelligenceSynergy: number;
  systemHealth: number;
  learningVelocity: number;
  responseTime: number;
}

interface PlatformState {
  metrics: PlatformMetrics;
  aiStatus: 'ready' | 'learning' | 'optimizing';
  isGenerating: boolean;
  lastUpdate: string;
}

export const usePlatformState = () => {
  const [state, setState] = useState<PlatformState>({
    metrics: {
      crossDomainConnections: 127,
      significanceScore: 9.2,
      globalUsers: 24891,
      aiOptimizations: 156,
      intelligenceSynergy: 96.8,
      systemHealth: 98.4,
      learningVelocity: 97.2,
      responseTime: 12
    },
    aiStatus: 'ready',
    isGenerating: false,
    lastUpdate: new Date().toISOString()
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setState(prev => ({
        ...prev,
        metrics: {
          ...prev.metrics,
          crossDomainConnections: prev.metrics.crossDomainConnections + Math.floor(Math.random() * 3),
          significanceScore: Math.min(10, Math.max(8, prev.metrics.significanceScore + (Math.random() * 0.2 - 0.1))),
          globalUsers: prev.metrics.globalUsers + Math.floor(Math.random() * 20 - 10),
          aiOptimizations: prev.metrics.aiOptimizations + Math.floor(Math.random() * 5),
          intelligenceSynergy: Math.min(99.9, Math.max(90, prev.metrics.intelligenceSynergy + (Math.random() * 0.3 - 0.15))),
          systemHealth: Math.min(99.9, Math.max(95, prev.metrics.systemHealth + (Math.random() * 0.2 - 0.1))),
          learningVelocity: Math.min(99.9, Math.max(95, prev.metrics.learningVelocity + (Math.random() * 0.1))),
          responseTime: Math.max(8, prev.metrics.responseTime + Math.floor(Math.random() * 6 - 3))
        },
        aiStatus: Math.random() > 0.8 ? 'learning' : 'ready',
        lastUpdate: new Date().toISOString()
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const updateGeneratingState = (isGenerating: boolean) => {
    setState(prev => ({ ...prev, isGenerating }));
  };

  return {
    ...state,
    updateGeneratingState
  };
};
