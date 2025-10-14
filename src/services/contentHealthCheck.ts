/**
 * Content Health Check Service
 * Phase 15: Monitor content generation and display
 */

import { loggingService } from './loggingService';

export interface ContentHealth {
  section: string;
  status: 'healthy' | 'degraded' | 'failed';
  lastUpdate: string;
  itemCount: number;
  generationRate: number;
  errors: string[];
  recommendations: string[];
}

export interface HealthCheckResult {
  overall: 'healthy' | 'degraded' | 'failed';
  sections: ContentHealth[];
  timestamp: string;
  alerts: string[];
}

class ContentHealthCheckService {
  private healthData = new Map<string, ContentHealth>();
  private checkInterval: NodeJS.Timeout | null = null;
  private listeners: ((result: HealthCheckResult) => void)[] = [];

  startMonitoring(intervalMs: number = 60000) {
    console.log('🏥 Content Health Monitoring started');
    
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
    }

    this.checkInterval = setInterval(() => {
      this.performHealthCheck();
    }, intervalMs);

    // Initial check
    this.performHealthCheck();
  }

  stopMonitoring() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  async performHealthCheck(): Promise<HealthCheckResult> {
    const sections = [
      'feed',
      'news',
      'entertainment',
      'shop',
      'music',
      'wellness',
      'opportunities'
    ];

    const healthResults: ContentHealth[] = [];
    const alerts: string[] = [];

    for (const section of sections) {
      const health = await this.checkSectionHealth(section);
      healthResults.push(health);

      if (health.status === 'failed') {
        alerts.push(`${section} section is not generating content`);
      } else if (health.status === 'degraded') {
        alerts.push(`${section} section has reduced content generation`);
      }
    }

    const failedCount = healthResults.filter(h => h.status === 'failed').length;
    const degradedCount = healthResults.filter(h => h.status === 'degraded').length;

    const overall = 
      failedCount > 2 ? 'failed' : 
      degradedCount > 3 ? 'degraded' : 
      'healthy';

    const result: HealthCheckResult = {
      overall,
      sections: healthResults,
      timestamp: new Date().toISOString(),
      alerts
    };

    // Log the health check
    loggingService.log(
      overall === 'healthy' ? 'info' : overall === 'degraded' ? 'warn' : 'error',
      'content_generation',
      `Health check completed: ${overall}`,
      { sections: healthResults.length, alerts: alerts.length }
    );

    // Notify listeners
    this.notifyListeners(result);

    return result;
  }

  private async checkSectionHealth(section: string): Promise<ContentHealth> {
    // Get content from localStorage or indexedDB
    const contentKey = `${section}_content`;
    const content = this.getStoredContent(contentKey);
    
    const lastUpdateKey = `${section}_last_update`;
    const lastUpdate = localStorage.getItem(lastUpdateKey) || new Date().toISOString();
    
    const timeSinceUpdate = Date.now() - new Date(lastUpdate).getTime();
    const itemCount = content?.length || 0;
    
    // Calculate generation rate (items per hour)
    const generationRate = timeSinceUpdate > 0 
      ? (itemCount / (timeSinceUpdate / 3600000)) 
      : 0;

    const errors: string[] = [];
    const recommendations: string[] = [];

    // Determine status
    let status: 'healthy' | 'degraded' | 'failed' = 'healthy';

    if (itemCount === 0) {
      status = 'failed';
      errors.push('No content available');
      recommendations.push('Verify content generation service is running');
      recommendations.push('Check Pollen AI backend connectivity');
    } else if (timeSinceUpdate > 30 * 60 * 1000) { // 30 minutes
      status = 'degraded';
      errors.push('Content not updated in 30 minutes');
      recommendations.push('Check Worker Bot status');
      recommendations.push('Verify content refresh interval');
    } else if (generationRate < 1) {
      status = 'degraded';
      errors.push('Low generation rate');
      recommendations.push('Optimize content generation pipeline');
    }

    this.healthData.set(section, {
      section,
      status,
      lastUpdate,
      itemCount,
      generationRate,
      errors,
      recommendations
    });

    return this.healthData.get(section)!;
  }

  private getStoredContent(key: string): any[] {
    try {
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  getSectionHealth(section: string): ContentHealth | undefined {
    return this.healthData.get(section);
  }

  getAllHealth(): ContentHealth[] {
    return Array.from(this.healthData.values());
  }

  onHealthChange(callback: (result: HealthCheckResult) => void) {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter(cb => cb !== callback);
    };
  }

  private notifyListeners(result: HealthCheckResult) {
    this.listeners.forEach(cb => cb(result));
  }

  reportContentGeneration(section: string, itemCount: number) {
    localStorage.setItem(`${section}_last_update`, new Date().toISOString());
    loggingService.logContentGeneration(section, itemCount, 'automated', {
      timestamp: new Date().toISOString()
    });
  }
}

export const contentHealthCheck = new ContentHealthCheckService();
