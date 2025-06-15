
import { useState, useEffect } from 'react';
import { intelligenceEngine } from '../services/intelligenceEngine';

interface IntelligenceData {
  metrics: {
    accuracy: number;
    processingSpeed: number;
    learningRate: number;
    crossDomainConnections: number;
    significanceScore: number;
    systemHealth: number;
  };
  realTimeData: {
    globalUsers: number;
    activeConnections: number;
    dataProcessed: number;
    aiOptimizations: number;
    responseTime: number;
    throughput: number;
  };
  insights: Array<{
    id: string;
    type: string;
    title: string;
    description: string;
    significance: number;
    domains: string[];
    actionable: boolean;
    timestamp: string;
    confidence: number;
  }>;
}

export const useIntelligenceEngine = () => {
  const [data, setData] = useState<IntelligenceData>({
    metrics: intelligenceEngine.getMetrics(),
    realTimeData: intelligenceEngine.getRealTimeData(),
    insights: intelligenceEngine.getInsights()
  });
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    const unsubscribe = intelligenceEngine.subscribe((newData) => {
      setData(newData);
    });

    return unsubscribe;
  }, []);

  const generateInsights = async () => {
    setIsGenerating(true);
    
    // Simulate AI processing time
    setTimeout(() => {
      intelligenceEngine.generateInsights();
      setIsGenerating(false);
    }, 1500);
  };

  return {
    ...data,
    isGenerating,
    generateInsights
  };
};
