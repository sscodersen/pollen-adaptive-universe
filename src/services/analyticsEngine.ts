import axios from 'axios';

interface UserBehavior {
  userId: string;
  actions: Array<{
    type: string;
    timestamp: Date;
    metadata?: any;
  }>;
  preferences: Record<string, any>;
  demographics?: Record<string, any>;
}

interface AnalyticsPattern {
  id: string;
  type: 'engagement' | 'behavior' | 'preference' | 'trend';
  description: string;
  confidence: number;
  impact: 'low' | 'medium' | 'high';
  recommendation: string;
  data: any;
}

interface AnalyticsInsight {
  type: string;
  description: string;
  impact: string;
  recommendation: string;
  confidence?: number;
}

class AnalyticsEngine {
  private behaviorData: Map<string, UserBehavior> = new Map();
  private patterns: AnalyticsPattern[] = [];
  private eventBuffer: any[] = [];

  // Track user behavior
  trackEvent(userId: string, eventType: string, metadata?: any) {
    const behavior = this.behaviorData.get(userId) || {
      userId,
      actions: [],
      preferences: {}
    };

    behavior.actions.push({
      type: eventType,
      timestamp: new Date(),
      metadata
    });

    // Keep last 1000 actions per user
    if (behavior.actions.length > 1000) {
      behavior.actions = behavior.actions.slice(-1000);
    }

    this.behaviorData.set(userId, behavior);
    this.eventBuffer.push({ userId, eventType, metadata, timestamp: new Date() });

    // Process patterns if buffer is large enough
    if (this.eventBuffer.length >= 50) {
      this.detectPatterns();
    }
  }

  // ML Pattern Detection
  private detectPatterns() {
    this.eventBuffer = [];

    // Engagement Pattern Detection
    this.behaviorData.forEach((behavior, userId) => {
      const recentActions = behavior.actions.slice(-50);
      
      // High engagement pattern
      if (recentActions.length >= 40) {
        const timeSpan = recentActions[recentActions.length - 1].timestamp.getTime() - 
                        recentActions[0].timestamp.getTime();
        const avgTime = timeSpan / recentActions.length;

        if (avgTime < 60000) { // Less than 1 min between actions
          this.addPattern({
            id: `engagement_${userId}_${Date.now()}`,
            type: 'engagement',
            description: `High engagement detected for user ${userId}`,
            confidence: 0.85,
            impact: 'high',
            recommendation: 'Provide premium content and personalized recommendations',
            data: { userId, actionsCount: recentActions.length, avgTimeBetweenActions: avgTime }
          });
        }
      }

      // Content preference patterns
      const contentTypes = recentActions
        .filter(a => a.type === 'view_content')
        .map(a => a.metadata?.contentType)
        .filter(Boolean);

      if (contentTypes.length > 10) {
        const typeCounts = contentTypes.reduce((acc: Record<string, number>, type) => {
          acc[type] = (acc[type] || 0) + 1;
          return acc;
        }, {});

        const dominantType = Object.entries(typeCounts)
          .sort(([, a], [, b]) => b - a)[0];

        if (dominantType && dominantType[1] / contentTypes.length > 0.6) {
          this.addPattern({
            id: `preference_${userId}_${Date.now()}`,
            type: 'preference',
            description: `Strong preference for ${dominantType[0]} content`,
            confidence: dominantType[1] / contentTypes.length,
            impact: 'medium',
            recommendation: `Increase ${dominantType[0]} content in feed`,
            data: { userId, preferredType: dominantType[0], frequency: dominantType[1] }
          });
        }
      }

      // Time-based behavior patterns
      const actionTimes = recentActions.map(a => a.timestamp.getHours());
      const peakHour = this.findPeakHour(actionTimes);

      if (peakHour !== null) {
        this.addPattern({
          id: `behavior_${userId}_${Date.now()}`,
          type: 'behavior',
          description: `Peak activity at ${peakHour}:00`,
          confidence: 0.75,
          impact: 'medium',
          recommendation: `Send notifications around ${peakHour}:00 for optimal engagement`,
          data: { userId, peakHour }
        });
      }
    });

    // Cross-user trend detection
    this.detectTrends();
  }

  private findPeakHour(hours: number[]): number | null {
    if (hours.length < 10) return null;

    const hourCounts = hours.reduce((acc: Record<number, number>, hour) => {
      acc[hour] = (acc[hour] || 0) + 1;
      return acc;
    }, {});

    const peak = Object.entries(hourCounts)
      .sort(([, a], [, b]) => b - a)[0];

    return peak && peak[1] > hours.length * 0.3 ? parseInt(peak[0]) : null;
  }

  private detectTrends() {
    const allActions = Array.from(this.behaviorData.values())
      .flatMap(b => b.actions.slice(-20));

    if (allActions.length < 50) return;

    // Detect trending content types
    const recentContentViews = allActions
      .filter(a => a.type === 'view_content' && a.timestamp.getTime() > Date.now() - 3600000)
      .map(a => a.metadata?.contentType)
      .filter(Boolean);

    if (recentContentViews.length > 20) {
      const typeCounts = recentContentViews.reduce((acc: Record<string, number>, type) => {
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      }, {});

      const trending = Object.entries(typeCounts)
        .filter(([, count]) => count > recentContentViews.length * 0.25)
        .sort(([, a], [, b]) => b - a)[0];

      if (trending) {
        this.addPattern({
          id: `trend_${Date.now()}`,
          type: 'trend',
          description: `${trending[0]} content is trending`,
          confidence: 0.8,
          impact: 'high',
          recommendation: `Feature ${trending[0]} content prominently`,
          data: { contentType: trending[0], views: trending[1], percentage: trending[1] / recentContentViews.length }
        });
      }
    }
  }

  private addPattern(pattern: AnalyticsPattern) {
    // Avoid duplicates
    const exists = this.patterns.some(p => 
      p.type === pattern.type && 
      p.data?.userId === pattern.data?.userId &&
      Date.now() - new Date(p.data?.timestamp || 0).getTime() < 3600000
    );

    if (!exists) {
      this.patterns.push(pattern);
      
      // Keep last 100 patterns
      if (this.patterns.length > 100) {
        this.patterns = this.patterns.slice(-100);
      }
    }
  }

  // Get insights for a specific user
  getUserInsights(userId: string): AnalyticsPattern[] {
    return this.patterns.filter(p => p.data?.userId === userId);
  }

  // Get global insights
  getGlobalInsights(): AnalyticsPattern[] {
    return this.patterns.filter(p => p.type === 'trend');
  }

  // Get all patterns
  getAllPatterns(): AnalyticsPattern[] {
    return this.patterns;
  }

  // Calculate user engagement score
  calculateEngagementScore(userId: string): number {
    const behavior = this.behaviorData.get(userId);
    if (!behavior) return 0;

    const recentActions = behavior.actions.slice(-100);
    if (recentActions.length === 0) return 0;

    // Factors: recency, frequency, diversity
    const recencyScore = this.calculateRecencyScore(recentActions);
    const frequencyScore = Math.min(recentActions.length / 100, 1);
    const diversityScore = this.calculateDiversityScore(recentActions);

    return (recencyScore * 0.4 + frequencyScore * 0.4 + diversityScore * 0.2) * 100;
  }

  private calculateRecencyScore(actions: any[]): number {
    if (actions.length === 0) return 0;

    const lastAction = actions[actions.length - 1].timestamp.getTime();
    const hoursSince = (Date.now() - lastAction) / (1000 * 60 * 60);

    if (hoursSince < 1) return 1;
    if (hoursSince < 24) return 0.8;
    if (hoursSince < 168) return 0.5;
    return 0.2;
  }

  private calculateDiversityScore(actions: any[]): number {
    const types = new Set(actions.map(a => a.type));
    return Math.min(types.size / 10, 1);
  }

  // Advanced analytics with AI Worker Bot
  async performAdvancedAnalytics(userId: string): Promise<AnalyticsInsight[]> {
    const behavior = this.behaviorData.get(userId);
    if (!behavior) return [];

    try {
      const response = await axios.post('/api/worker/perform-analytics', {
        userData: {
          userId,
          actionsCount: behavior.actions.length,
          recentActions: behavior.actions.slice(-50),
          preferences: behavior.preferences
        },
        metrics: {
          engagementScore: this.calculateEngagementScore(userId),
          patterns: this.getUserInsights(userId)
        },
        insights: this.getUserInsights(userId)
      });

      if (response.data.taskId) {
        // Poll for results
        return this.pollTaskResults(response.data.taskId);
      }

      return [];
    } catch (error) {
      console.error('Analytics error:', error);
      return [];
    }
  }

  private async pollTaskResults(taskId: string, maxAttempts = 30): Promise<any> {
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      try {
        const response = await axios.get(`/api/worker/tasks/${taskId}`);
        const task = response.data.task;

        if (task.status === 'completed') {
          return task.result.insights || [];
        } else if (task.status === 'failed') {
          throw new Error(task.error || 'Task failed');
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }

    throw new Error('Task timeout');
  }

  // User segmentation
  segmentUsers(): Map<string, string[]> {
    const segments = new Map<string, string[]>();
    segments.set('high_engagement', []);
    segments.set('medium_engagement', []);
    segments.set('low_engagement', []);
    segments.set('at_risk', []);
    segments.set('new_users', []);

    this.behaviorData.forEach((behavior, userId) => {
      const score = this.calculateEngagementScore(userId);
      const actionCount = behavior.actions.length;

      if (actionCount < 10) {
        segments.get('new_users')!.push(userId);
      } else if (score > 75) {
        segments.get('high_engagement')!.push(userId);
      } else if (score > 50) {
        segments.get('medium_engagement')!.push(userId);
      } else if (score > 25) {
        segments.get('low_engagement')!.push(userId);
      } else {
        segments.get('at_risk')!.push(userId);
      }
    });

    return segments;
  }

  // Get analytics summary
  getAnalyticsSummary() {
    const segments = this.segmentUsers();
    const totalUsers = this.behaviorData.size;

    return {
      totalUsers,
      segments: {
        high_engagement: segments.get('high_engagement')!.length,
        medium_engagement: segments.get('medium_engagement')!.length,
        low_engagement: segments.get('low_engagement')!.length,
        at_risk: segments.get('at_risk')!.length,
        new_users: segments.get('new_users')!.length
      },
      patterns: {
        total: this.patterns.length,
        byType: this.patterns.reduce((acc, p) => {
          acc[p.type] = (acc[p.type] || 0) + 1;
          return acc;
        }, {} as Record<string, number>)
      },
      insights: this.patterns.slice(-10)
    };
  }
}

export const analyticsEngine = new AnalyticsEngine();
export type { AnalyticsPattern, AnalyticsInsight, UserBehavior };
