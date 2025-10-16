/**
 * SSE Trend Scraper Client
 * Connects to the Python SSE trend scraper service
 * Streams real-time trends from Exploding Topics enhanced by Pollen AI
 */

export interface ScrapedTrend {
  id: string;
  topic: string;
  category: string;
  score: number;
  growth_rate: number;
  search_volume: number;
  timestamp: string;
  source: string;
  keywords: string[];
  description?: string;
  ai_insights?: string;
  ai_confidence?: number;
}

export interface SSETrendEvent {
  event: 'connected' | 'scraping' | 'trend_update' | 'batch_complete' | 'heartbeat';
  data: any;
}

type TrendCallback = (trend: ScrapedTrend) => void;
type StatusCallback = (status: string, data: any) => void;

class TrendScraperSSEClient {
  private eventSource: EventSource | null = null;
  private connected = false;
  private trendCallbacks: TrendCallback[] = [];
  private statusCallbacks: StatusCallback[] = [];
  private latestTrends: ScrapedTrend[] = [];
  private readonly scraperUrl: string;

  constructor() {
    // Use environment variable or default to port 8099
    this.scraperUrl = import.meta.env.VITE_TREND_SCRAPER_URL || 'http://localhost:8099';
  }

  /**
   * Connect to the SSE trend stream
   */
  connect(): void {
    if (this.connected) {
      console.log('✅ Trend Scraper SSE already connected');
      return;
    }

    try {
      this.eventSource = new EventSource(`${this.scraperUrl}/trends/stream`);

      // Handle connection open
      this.eventSource.onopen = () => {
        this.connected = true;
        console.log('✅ Trend Scraper SSE connected');
        this.notifyStatus('connected', { message: 'Stream connected' });
      };

      // Handle messages
      this.eventSource.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          this.handleSSEEvent({
            event: 'trend_update',
            data: parsedData
          });
        } catch (error) {
          console.error('Error parsing SSE message:', error);
        }
      };

      // Handle specific event types
      this.eventSource.addEventListener('connected', (event: any) => {
        const data = JSON.parse(event.data);
        console.log('📡 Trend Scraper connected:', data);
        this.notifyStatus('connected', data);
      });

      this.eventSource.addEventListener('scraping', (event: any) => {
        const data = JSON.parse(event.data);
        console.log('🔍 Scraping trends:', data.message);
        this.notifyStatus('scraping', data);
      });

      this.eventSource.addEventListener('trend_update', (event: any) => {
        const trend: ScrapedTrend = JSON.parse(event.data);
        console.log('📊 New trend:', trend.topic);
        this.latestTrends.push(trend);
        this.notifyTrend(trend);
      });

      this.eventSource.addEventListener('batch_complete', (event: any) => {
        const data = JSON.parse(event.data);
        console.log('✅ Batch complete:', data.trends_count, 'trends');
        this.notifyStatus('batch_complete', data);
      });

      this.eventSource.addEventListener('heartbeat', (event: any) => {
        const data = JSON.parse(event.data);
        // Silent heartbeat, just for connection keep-alive
      });

      // Handle errors
      this.eventSource.onerror = (error) => {
        console.error('❌ Trend Scraper SSE error:', error);
        this.connected = false;
        this.notifyStatus('error', { error: 'Connection lost' });
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          this.disconnect();
          this.connect();
        }, 5000);
      };

    } catch (error) {
      console.error('Failed to connect to Trend Scraper SSE:', error);
      this.notifyStatus('error', { error: 'Failed to connect' });
    }
  }

  /**
   * Disconnect from the SSE stream
   */
  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.connected = false;
      console.log('🔌 Trend Scraper SSE disconnected');
    }
  }

  /**
   * Subscribe to trend updates
   */
  onTrend(callback: TrendCallback): () => void {
    this.trendCallbacks.push(callback);
    
    // Return unsubscribe function
    return () => {
      this.trendCallbacks = this.trendCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Subscribe to status updates
   */
  onStatus(callback: StatusCallback): () => void {
    this.statusCallbacks.push(callback);
    
    // Return unsubscribe function
    return () => {
      this.statusCallbacks = this.statusCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Get latest scraped trends (non-SSE)
   */
  async getLatestTrends(): Promise<ScrapedTrend[]> {
    try {
      const response = await fetch(`${this.scraperUrl}/trends/latest`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      this.latestTrends = data.trends || [];
      return this.latestTrends;
    } catch (error) {
      console.error('Failed to fetch latest trends:', error);
      return this.latestTrends;
    }
  }

  /**
   * Get cached trends
   */
  getCachedTrends(): ScrapedTrend[] {
    return this.latestTrends;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * Check scraper health
   */
  async checkHealth(): Promise<any> {
    try {
      const response = await fetch(`${this.scraperUrl}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      return { status: 'unavailable', error: String(error) };
    }
  }

  private handleSSEEvent(event: SSETrendEvent): void {
    // Handle different event types
    switch (event.event) {
      case 'trend_update':
        this.latestTrends.push(event.data);
        this.notifyTrend(event.data);
        break;
      default:
        this.notifyStatus(event.event, event.data);
    }
  }

  private notifyTrend(trend: ScrapedTrend): void {
    this.trendCallbacks.forEach(callback => {
      try {
        callback(trend);
      } catch (error) {
        console.error('Error in trend callback:', error);
      }
    });
  }

  private notifyStatus(status: string, data: any): void {
    this.statusCallbacks.forEach(callback => {
      try {
        callback(status, data);
      } catch (error) {
        console.error('Error in status callback:', error);
      }
    });
  }
}

// Export singleton instance
export const trendScraperSSE = new TrendScraperSSEClient();
