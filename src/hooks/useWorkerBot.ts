import { useEffect, useState, useCallback } from 'react';
import { workerBotClient, type WorkerBotMessage } from '@/services/workerBotClient';
import { analyticsEngine } from '@/services/analyticsEngine';

export function useWorkerBot(userId?: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Connect to Worker Bot SSE stream
    workerBotClient.connect(userId);

    // Listen for connection events
    const handleConnected = (data: WorkerBotMessage) => {
      setIsConnected(true);
      console.log('Worker Bot connected:', data);
    };

    workerBotClient.on('connected', handleConnected);

    // Cleanup
    return () => {
      workerBotClient.off('connected', handleConnected);
    };
  }, [userId]);

  useEffect(() => {
    // Load stats periodically
    const loadStats = async () => {
      const stats = await workerBotClient.getStats();
      setStats(stats);
    };

    loadStats();
    const interval = setInterval(loadStats, 5000);

    return () => clearInterval(interval);
  }, []);

  const generateContent = useCallback(async (prompt: string, type = 'general') => {
    setLoading(true);
    try {
      const result = await workerBotClient.generateContent(prompt, type, userId);
      
      // Track analytics
      analyticsEngine.trackEvent(userId || 'anonymous', 'ai_content_generated', {
        type,
        promptLength: prompt.length
      });

      return result;
    } catch (error) {
      console.error('Content generation failed:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const generateMusic = useCallback(async (mood: string, genre: string, occasion: string) => {
    setLoading(true);
    try {
      const result = await workerBotClient.generateMusic(mood, genre, occasion);
      
      analyticsEngine.trackEvent(userId || 'anonymous', 'ai_music_generated', {
        mood,
        genre,
        occasion
      });

      return result;
    } catch (error) {
      console.error('Music generation failed:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const generateAds = useCallback(async (targetAudience: string, product: string, goals: string) => {
    setLoading(true);
    try {
      const result = await workerBotClient.generateAds(targetAudience, product, goals);
      
      analyticsEngine.trackEvent(userId || 'anonymous', 'ai_ads_generated', {
        targetAudience,
        product
      });

      return result;
    } catch (error) {
      console.error('Ad generation failed:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const analyzeTrends = useCallback(async (data: any, timeRange: string, category: string) => {
    setLoading(true);
    try {
      const result = await workerBotClient.analyzeTrends(data, timeRange, category);
      
      analyticsEngine.trackEvent(userId || 'anonymous', 'ai_trends_analyzed', {
        timeRange,
        category
      });

      return result;
    } catch (error) {
      console.error('Trend analysis failed:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const performAnalytics = useCallback(async (userData: any, metrics: any, insights: any) => {
    setLoading(true);
    try {
      const result = await workerBotClient.performAnalytics(userData, metrics, insights);
      
      analyticsEngine.trackEvent(userId || 'anonymous', 'ai_analytics_performed', {
        metricsCount: Object.keys(metrics || {}).length
      });

      return result;
    } catch (error) {
      console.error('Analytics failed:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const personalizeContent = useCallback(async (userProfile: any, contentPool: any, preferences: any) => {
    setLoading(true);
    try {
      const result = await workerBotClient.personalizeContent(userProfile, contentPool, preferences);
      
      analyticsEngine.trackEvent(userId || 'anonymous', 'ai_content_personalized', {
        contentCount: contentPool?.length || 0
      });

      return result;
    } catch (error) {
      console.error('Personalization failed:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [userId]);

  return {
    isConnected,
    stats,
    loading,
    generateContent,
    generateMusic,
    generateAds,
    analyzeTrends,
    performAnalytics,
    personalizeContent
  };
}
