/**
 * Pollen Platform Initialization
 * 
 * Integrates all AI and content generation services:
 * - Pollen AI Backend
 * - Trend Scraper SSE
 * - Bento News Algorithm
 * - Content Orchestrator
 */

import { pollenAI } from './pollenAIUnified';
import { trendScraperSSE } from './trendScraperSSE';
import { bentoNewsAlgorithm } from './bentoNewsAlgorithm';
import { contentOrchestrator } from './contentOrchestrator';
import { workerBotClient } from './workerBotClient';

export interface PlatformConfig {
  pollenAIUrl?: string;
  trendScraperUrl?: string;
  enableBentoAlgorithm?: boolean;
  enableTrendScraping?: boolean;
  bentoRefreshInterval?: number; // milliseconds
  categories?: string[];
}

class PollenPlatformInitializer {
  private initialized = false;
  private config: PlatformConfig = {
    pollenAIUrl: 'http://localhost:8000',
    trendScraperUrl: 'http://localhost:8099',
    enableBentoAlgorithm: true,
    enableTrendScraping: true,
    bentoRefreshInterval: 15 * 60 * 1000, // 15 minutes
    categories: ['social', 'news', 'entertainment', 'shop', 'music', 'wellness']
  };

  /**
   * Initialize the entire Pollen AI platform
   */
  async initialize(customConfig?: Partial<PlatformConfig>): Promise<void> {
    if (this.initialized) {
      console.log('✅ Platform already initialized');
      return;
    }

    // Merge custom config
    if (customConfig) {
      this.config = { ...this.config, ...customConfig };
    }

    console.log('🚀 Initializing Pollen AI Platform...');

    try {
      // Step 1: Setup Pollen AI backend
      await this.setupPollenAI();

      // Step 2: Setup backend integrations
      await this.setupBackendIntegrations();

      // Step 3: Initialize Trend Scraper SSE
      if (this.config.enableTrendScraping) {
        await this.setupTrendScraper();
      }

      // Step 4: Start Bento News Algorithm
      if (this.config.enableBentoAlgorithm) {
        await this.startBentoAlgorithm();
      }

      // Step 5: Connect Worker Bot
      await this.setupWorkerBot();

      this.initialized = true;
      console.log('✅ Pollen AI Platform Initialized Successfully!');
      console.log('📊 Services Running:');
      console.log(`   - Pollen AI: ${this.config.pollenAIUrl}`);
      console.log(`   - Trend Scraper: ${this.config.trendScraperUrl}`);
      console.log(`   - Bento Algorithm: ${this.config.enableBentoAlgorithm ? 'Active' : 'Inactive'}`);
      console.log(`   - Content Generation: Continuous`);

    } catch (error) {
      console.error('❌ Platform initialization failed:', error);
      throw error;
    }
  }

  /**
   * Shutdown the platform
   */
  shutdown(): void {
    if (!this.initialized) {
      return;
    }

    console.log('🛑 Shutting down Pollen AI Platform...');

    // Stop Bento algorithm
    if (this.config.enableBentoAlgorithm) {
      bentoNewsAlgorithm.stopContinuousGeneration();
    }

    // Disconnect trend scraper
    if (this.config.enableTrendScraping) {
      trendScraperSSE.disconnect();
    }

    // Disconnect worker bot
    workerBotClient.disconnect();

    // Stop all content streams
    contentOrchestrator.stopAllStreams();

    this.initialized = false;
    console.log('✅ Platform shutdown complete');
  }

  /**
   * Get platform status
   */
  async getStatus(): Promise<any> {
    const pollenAIHealth = await pollenAI.checkHealth();
    const trendScraperHealth = await trendScraperSSE.checkHealth();

    return {
      initialized: this.initialized,
      services: {
        pollenAI: {
          healthy: pollenAIHealth,
          url: this.config.pollenAIUrl
        },
        trendScraper: {
          healthy: trendScraperHealth.status === 'healthy',
          url: this.config.trendScraperUrl,
          activeConnections: trendScraperHealth.active_connections,
          cachedTrends: trendScraperHealth.cached_trends
        },
        bentoAlgorithm: {
          enabled: this.config.enableBentoAlgorithm,
          generatedPosts: this.getBentoPostsCount()
        }
      },
      timestamp: new Date().toISOString()
    };
  }

  // Private methods

  private async setupPollenAI(): Promise<void> {
    console.log('🧠 Setting up Pollen AI...');

    const isHealthy = await pollenAI.checkHealth();

    if (isHealthy) {
      console.log('✅ Pollen AI connected and healthy');
    } else {
      console.warn('⚠️ Pollen AI not available, using fallback mode');
    }
  }

  private async setupBackendIntegrations(): Promise<void> {
    console.log('🔌 Setting up backend integrations...');

    await contentOrchestrator.setupBackendIntegration({
      pollenEndpoint: this.config.pollenAIUrl,
      enableSSE: true,
      pythonScripts: {
        'trend_scraper': this.config.trendScraperUrl || ''
      }
    });

    console.log('✅ Backend integrations configured');
  }

  private async setupTrendScraper(): Promise<void> {
    console.log('📡 Setting up Trend Scraper SSE...');

    const health = await trendScraperSSE.checkHealth();

    if (health.status === 'healthy') {
      console.log('✅ Trend Scraper connected');

      // Get initial trends
      const trends = await trendScraperSSE.getLatestTrends();
      console.log(`✅ Loaded ${trends.length} initial trends`);

      // Connect to SSE stream (connection happens in Bento algorithm)
      console.log('✅ Trend Scraper ready for real-time streaming');
    } else {
      console.warn('⚠️ Trend Scraper not available');
    }
  }

  private async startBentoAlgorithm(): Promise<void> {
    console.log('🎯 Starting Bento News Algorithm...');

    await bentoNewsAlgorithm.startContinuousGeneration({
      categories: this.config.categories as any[],
      postsPerCategory: 5,
      refreshInterval: this.config.bentoRefreshInterval || 15 * 60 * 1000,
      qualityThreshold: 7.5,
      diversityWeight: 0.6,
      trendWeight: 0.8
    });

    console.log('✅ Bento News Algorithm started');
    console.log(`   Generating posts for: ${this.config.categories?.join(', ')}`);
  }

  private async setupWorkerBot(): Promise<void> {
    console.log('🤖 Setting up Worker Bot...');

    try {
      // Connect Worker Bot for background AI tasks
      workerBotClient.connect();

      // Setup event listeners
      workerBotClient.on('connected', () => {
        console.log('✅ Worker Bot connected');
      });

      workerBotClient.on('task_completed', (data) => {
        console.log('✅ Worker Bot task completed:', data.task?.type);
      });

    } catch (error) {
      console.warn('⚠️ Worker Bot setup failed:', error);
    }
  }

  private getBentoPostsCount(): number {
    const allPosts = bentoNewsAlgorithm.getAllPosts();
    let total = 0;
    for (const posts of allPosts.values()) {
      total += posts.length;
    }
    return total;
  }

  /**
   * Check if platform is ready
   */
  isReady(): boolean {
    return this.initialized;
  }

  /**
   * Get Bento posts for a category
   */
  getBentoPosts(category: string): any[] {
    if (!this.config.enableBentoAlgorithm) {
      return [];
    }
    return bentoNewsAlgorithm.getPosts(category as any);
  }
}

// Export singleton instance
export const pollenPlatform = new PollenPlatformInitializer();

// Auto-initialize on module load (can be disabled if needed)
if (typeof window !== 'undefined') {
  // Initialize platform when app starts
  pollenPlatform.initialize().catch(err => {
    console.error('Failed to auto-initialize Pollen Platform:', err);
  });

  // Cleanup on window unload
  window.addEventListener('beforeunload', () => {
    pollenPlatform.shutdown();
  });
}
