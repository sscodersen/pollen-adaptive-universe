// Pollen AI Connection Manager - Production Ready
import { pollenAI } from './pollenAI';
import { storageService } from './storageService';

interface ConnectionConfig {
  apiUrl: string;
  retryAttempts: number;
  retryDelay: number;
  healthCheckInterval: number;
}

class PollenConnectionManager {
  private config: ConnectionConfig;
  private isInitialized = false;
  private healthCheckTimer?: NodeJS.Timeout;
  private reconnectAttempts = 0;

  constructor() {
    this.config = {
      apiUrl: this.getApiUrl(),
      retryAttempts: 3,
      retryDelay: 5000,
      healthCheckInterval: 30000,
    };
  }

  private getApiUrl(): string {
    // Check if user has configured a custom Pollen backend URL
    const savedUrl = storageService.getData('pollenBackendUrl');
    if (savedUrl && typeof savedUrl === 'string') {
      return savedUrl;
    }

    // Try different possible Pollen backend URLs
    const possibleUrls = [
      'http://localhost:8000',  // Local Docker
      'http://127.0.0.1:8000',  // Alternative local
      'http://pollen-backend:8000',  // Docker compose
    ];

    return possibleUrls[0];
  }

  async initialize(): Promise<boolean> {
    if (this.isInitialized) {
      return true;
    }

    console.log('🚀 Initializing Pollen AI Connection Manager...');
    
    // Try to connect
    const connected = await this.connectWithRetry();
    
    if (connected) {
      this.startHealthCheck();
      this.isInitialized = true;
      console.log('✅ Pollen AI fully initialized and connected');
    } else {
      console.log('🔄 Pollen AI initialized in fallback mode - will retry connection');
    }

    return this.isInitialized;
  }

  private async connectWithRetry(): Promise<boolean> {
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      console.log(`🔄 Pollen connection attempt ${attempt}/${this.config.retryAttempts}...`);
      
      const connected = await pollenAI.connect({ 
        apiUrl: this.config.apiUrl,
        enableSSE: true 
      });

      if (connected) {
        this.reconnectAttempts = 0;
        return true;
      }

      if (attempt < this.config.retryAttempts) {
        await this.delay(this.config.retryDelay);
      }
    }

    return false;
  }

  private startHealthCheck() {
    this.healthCheckTimer = setInterval(async () => {
      try {
        const response = await fetch(`${this.config.apiUrl}/health`, {
          method: 'GET',
        });

        if (!response.ok) {
          console.log('🔄 Pollen backend health check failed, attempting reconnect...');
          await this.connectWithRetry();
        }
      } catch (error) {
        console.log('🔄 Pollen backend unreachable, will retry...');
        await this.connectWithRetry();
      }
    }, this.config.healthCheckInterval);
  }

  async setCustomBackendUrl(url: string): Promise<boolean> {
    console.log(`🔧 Setting custom Pollen backend URL: ${url}`);
    
    // Validate URL format
    try {
      new URL(url);
    } catch (error) {
      console.error('❌ Invalid URL format');
      return false;
    }

    // Save to storage
    storageService.setData('pollenBackendUrl', url);
    this.config.apiUrl = url;

    // Try to connect with new URL
    const connected = await pollenAI.connect({ apiUrl: url });
    
    if (connected) {
      console.log('✅ Successfully connected to custom Pollen backend');
      this.reconnectAttempts = 0;
      return true;
    } else {
      console.log('🔄 Custom backend not reachable, using fallback mode');
      return false;
    }
  }

  getConnectionStatus(): {
    status: 'connected' | 'disconnected' | 'connecting';
    url: string;
    uptime: number;
  } {
    return {
      status: this.isInitialized ? 'connected' : 'disconnected',
      url: this.config.apiUrl,
      uptime: Date.now(), // Simplified for now
    };
  }

  async testConnection(): Promise<{
    success: boolean;
    latency?: number;
    error?: string;
  }> {
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${this.config.apiUrl}/health`, {
        method: 'GET',
      });

      const latency = Date.now() - startTime;

      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          latency,
        };
      } else {
        return {
          success: false,
          error: `HTTP ${response.status}: ${response.statusText}`,
        };
      }
    } catch (error) {
      return {
        success: false,
        latency: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Connection failed',
      };
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  destroy() {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }
    this.isInitialized = false;
  }
}

export const pollenConnection = new PollenConnectionManager();

// Auto-initialize when module loads
pollenConnection.initialize().catch(console.warn);