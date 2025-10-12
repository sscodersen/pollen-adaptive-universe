import { personalizationEngine } from '../personalizationEngine';
import { enhancedTrendEngine } from '../enhancedTrendEngine';
import pollenAIUnified from '../pollenAIUnified';

export interface TrendAlert {
  id: string;
  topic: string;
  category: string;
  alertType: 'emerging' | 'rising' | 'viral' | 'declining';
  confidence: number;
  momentum: number;
  message: string;
  createdAt: string;
  read: boolean;
  priority: 'high' | 'medium' | 'low';
}

export interface TrendPrediction {
  id: string;
  topic: string;
  category: string;
  predictedGrowth: number;
  confidence: number;
  timeframe: string;
  factors: string[];
  reasoning: string;
  createdAt: string;
}

export interface TrendAnalytics {
  topic: string;
  timeline: TimelinePoint[];
  engagement: EngagementMetrics;
  demographics: DemographicData;
  sentiment: SentimentData;
  relatedTrends: string[];
}

export interface TimelinePoint {
  date: string;
  value: number;
  events: string[];
}

export interface EngagementMetrics {
  totalViews: number;
  totalShares: number;
  totalComments: number;
  engagementRate: number;
  growthRate: number;
}

export interface DemographicData {
  ageGroups: Record<string, number>;
  locations: Record<string, number>;
  interests: Record<string, number>;
}

export interface SentimentData {
  positive: number;
  neutral: number;
  negative: number;
  keywords: string[];
}

class TrendAlertsService {
  private readonly ALERTS_KEY = 'trend_alerts';
  private readonly PREDICTIONS_KEY = 'trend_predictions';
  private readonly SUBSCRIPTIONS_KEY = 'trend_subscriptions';

  async monitorTrends(): Promise<TrendAlert[]> {
    const trends = await enhancedTrendEngine.getTrends();
    const alerts: TrendAlert[] = [];

    for (const trend of trends) {
      const alert = this.analyzeTrendForAlert(trend);
      if (alert) {
        alerts.push(alert);
      }
    }

    this.saveAlerts(alerts);
    return alerts;
  }

  private analyzeTrendForAlert(trend: any): TrendAlert | null {
    const momentum = trend.momentum || 0;
    const engagement = trend.engagement || 0;

    let alertType: 'emerging' | 'rising' | 'viral' | 'declining';
    let priority: 'high' | 'medium' | 'low';
    let message: string;

    if (momentum > 0.8 && engagement > 1000) {
      alertType = 'viral';
      priority = 'high';
      message = `🔥 ${trend.topic} is going viral! Massive engagement spike detected.`;
    } else if (momentum > 0.6) {
      alertType = 'rising';
      priority = 'high';
      message = `📈 ${trend.topic} is rapidly gaining traction.`;
    } else if (momentum > 0.3 && engagement < 500) {
      alertType = 'emerging';
      priority = 'medium';
      message = `🌱 ${trend.topic} is emerging as a new trend.`;
    } else if (momentum < -0.3) {
      alertType = 'declining';
      priority = 'low';
      message = `📉 ${trend.topic} interest is declining.`;
    } else {
      return null;
    }

    return {
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      topic: trend.topic,
      category: trend.type || 'general',
      alertType,
      confidence: trend.confidence || 0.7,
      momentum,
      message,
      createdAt: new Date().toISOString(),
      read: false,
      priority
    };
  }

  async generatePredictions(count: number = 10): Promise<TrendPrediction[]> {
    const trends = await enhancedTrendEngine.getTrends();
    const predictions: TrendPrediction[] = [];

    for (let i = 0; i < Math.min(count, trends.length); i++) {
      const trend = trends[i];
      
      try {
        const response = await pollenAIUnified.generate({
          prompt: `Predict the future trajectory of the trend "${trend.topic}" based on current momentum and engagement. Explain key factors.`,
          mode: 'social' as any,
          type: 'analysis'
        });

        const prediction: TrendPrediction = {
          id: `prediction_${Date.now()}_${i}`,
          topic: trend.topic,
          category: trend.type || 'general',
          predictedGrowth: this.calculatePredictedGrowth(trend),
          confidence: 0.6 + Math.random() * 0.3,
          timeframe: this.getTimeframe(trend),
          factors: this.extractFactors(response.content),
          reasoning: response.content,
          createdAt: new Date().toISOString()
        };

        predictions.push(prediction);
      } catch (error) {
        console.error('Failed to generate prediction:', error);
      }
    }

    this.savePredictions(predictions);
    return predictions.sort((a, b) => b.confidence - a.confidence);
  }

  private calculatePredictedGrowth(trend: any): number {
    const momentum = trend.momentum || 0;
    const engagement = trend.engagement || 0;
    const baseGrowth = momentum * 100;
    const engagementBoost = (engagement / 1000) * 10;
    return Math.min(baseGrowth + engagementBoost, 200);
  }

  private getTimeframe(trend: any): string {
    const momentum = trend.momentum || 0;
    if (momentum > 0.7) return '1-2 weeks';
    if (momentum > 0.4) return '2-4 weeks';
    return '1-2 months';
  }

  private extractFactors(content: string): string[] {
    const factors = [];
    const lines = content.split('\n');
    
    for (const line of lines) {
      if (line.includes('factor') || line.includes('because') || line.includes('due to')) {
        factors.push(line.trim());
      }
    }

    if (factors.length === 0) {
      factors.push('High engagement', 'Growing momentum', 'Positive sentiment');
    }

    return factors.slice(0, 5);
  }

  async getTrendAnalytics(topic: string): Promise<TrendAnalytics> {
    const timeline = this.generateTimeline(topic, 30);
    const engagement = this.calculateEngagement(timeline);
    const sentiment = this.analyzeSentiment(topic);

    return {
      topic,
      timeline,
      engagement,
      demographics: {
        ageGroups: {
          '18-24': 0.35,
          '25-34': 0.40,
          '35-44': 0.15,
          '45+': 0.10
        },
        locations: {
          'North America': 0.45,
          'Europe': 0.30,
          'Asia': 0.20,
          'Other': 0.05
        },
        interests: {
          'Technology': 0.40,
          'Business': 0.25,
          'Entertainment': 0.20,
          'Education': 0.15
        }
      },
      sentiment,
      relatedTrends: await this.findRelatedTrends(topic)
    };
  }

  private generateTimeline(topic: string, days: number): TimelinePoint[] {
    const timeline: TimelinePoint[] = [];
    let value = Math.random() * 100;

    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);

      value += (Math.random() - 0.4) * 20;
      value = Math.max(0, Math.min(1000, value));

      timeline.push({
        date: date.toISOString().split('T')[0],
        value: Math.floor(value),
        events: i % 7 === 0 ? [`${topic} milestone`] : []
      });
    }

    return timeline;
  }

  private calculateEngagement(timeline: TimelinePoint[]): EngagementMetrics {
    const totalViews = timeline.reduce((sum, point) => sum + point.value, 0);
    const avgValue = totalViews / timeline.length;
    const recentAvg = timeline.slice(-7).reduce((sum, p) => sum + p.value, 0) / 7;
    const growthRate = ((recentAvg - avgValue) / avgValue) * 100;

    return {
      totalViews,
      totalShares: Math.floor(totalViews * 0.1),
      totalComments: Math.floor(totalViews * 0.05),
      engagementRate: 5 + Math.random() * 10,
      growthRate
    };
  }

  private analyzeSentiment(topic: string): SentimentData {
    const positive = 0.5 + Math.random() * 0.3;
    const negative = Math.random() * 0.2;
    const neutral = 1 - positive - negative;

    return {
      positive,
      neutral,
      negative,
      keywords: [topic, 'trending', 'popular', 'discussion', 'viral']
    };
  }

  private async findRelatedTrends(topic: string): Promise<string[]> {
    const trends = await enhancedTrendEngine.getTrends();
    return trends
      .filter(t => t.topic !== topic)
      .slice(0, 5)
      .map(t => t.topic);
  }

  subscribeToTopic(userId: string, topic: string, categories: string[]): void {
    const subscriptions = this.getSubscriptions(userId);
    subscriptions.push({ topic, categories, createdAt: new Date().toISOString() });
    localStorage.setItem(`${this.SUBSCRIPTIONS_KEY}_${userId}`, JSON.stringify(subscriptions));

    personalizationEngine.trackBehavior({
      action: 'save',
      contentId: topic,
      contentType: 'educational',
      metadata: { type: 'trend_subscription', categories }
    });
  }

  getSubscriptions(userId: string): any[] {
    try {
      const stored = localStorage.getItem(`${this.SUBSCRIPTIONS_KEY}_${userId}`);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  private saveAlerts(alerts: TrendAlert[]): void {
    const existing = this.getAlerts();
    const combined = [...existing, ...alerts];
    const unique = combined.filter((alert, index, self) => 
      self.findIndex(a => a.topic === alert.topic && a.alertType === alert.alertType) === index
    );
    localStorage.setItem(this.ALERTS_KEY, JSON.stringify(unique.slice(-50)));
  }

  getAlerts(): TrendAlert[] {
    try {
      const stored = localStorage.getItem(this.ALERTS_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  markAlertAsRead(alertId: string): void {
    const alerts = this.getAlerts();
    const alert = alerts.find(a => a.id === alertId);
    if (alert) {
      alert.read = true;
      localStorage.setItem(this.ALERTS_KEY, JSON.stringify(alerts));
    }
  }

  private savePredictions(predictions: TrendPrediction[]): void {
    localStorage.setItem(this.PREDICTIONS_KEY, JSON.stringify(predictions));
  }

  getPredictions(): TrendPrediction[] {
    try {
      const stored = localStorage.getItem(this.PREDICTIONS_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }
}

export const trendAlertsService = new TrendAlertsService();
