import { pollenAI } from './pollenAIUnified';
import { universalSSE } from './universalSSE';
import { unifiedContentEngine, ContentType, UnifiedContent, GenerationStrategy } from './unifiedContentEngine';
import { enhancedContentEngine } from './enhancedContentEngine';
import { musicSSEService } from './musicSSE';
import { storageService } from './storageService';

export interface ContentRequest {
  type: ContentType;
  query?: string;
  count?: number;
  strategy?: GenerationStrategy;
  realtime?: boolean;
  useSSE?: boolean;
}

export interface ContentResponse<T = UnifiedContent> {
  content: T[];
  metadata: {
    totalGenerated: number;
    avgQuality: number;
    strategy: GenerationStrategy;
    timestamp: string;
    cacheHit: boolean;
  };
}

class ContentOrchestrator {
  private cacheExpiry = 5 * 60 * 1000; // 5 minutes
  private activeStreams = new Map<string, AbortController>();

  // Enhanced content generation with quality controls
  async generateContent<T extends UnifiedContent>(
    request: ContentRequest
  ): Promise<ContentResponse<T>> {
    const cacheKey = this.buildCacheKey(request);
    
    // Check cache first
    const cached = await this.getCachedContent<T>(cacheKey);
    if (cached && !request.realtime) {
      return cached;
    }

    // Use enhanced content engine for better quality
    const query = request.query || 'trending content';
    const enhancedContent = await enhancedContentEngine.generateQualityContent(
      request.type,
      query,
      request.count || 10
    );

    // Convert enhanced content to unified format
    const content = enhancedContent
      .filter(item => item.approved)
      .map(item => item.content) as T[];

    // Fallback to regular generation if not enough quality content
    if (content.length < (request.count || 10) / 2) {
      const strategy = request.strategy || this.getDefaultStrategy(request.type);
      const fallbackContent = await unifiedContentEngine.generateContent(
        request.type,
        (request.count || 10) - content.length,
        strategy
      ) as T[];
      content.push(...fallbackContent);
    }

    const response: ContentResponse<T> = {
      content: content.slice(0, request.count || 10),
      metadata: {
        totalGenerated: content.length,
        avgQuality: this.calculateEnhancedQuality(enhancedContent),
        strategy: request.strategy || this.getDefaultStrategy(request.type),
        timestamp: new Date().toISOString(),
        cacheHit: false
      }
    };

    // Cache the result
    await this.cacheContent(cacheKey, response);
    
    return response;
  }

  // Streaming content generation for real-time updates
  async *streamContent<T extends UnifiedContent>(
    request: ContentRequest
  ): AsyncGenerator<T> {
    const streamId = `stream-${request.type}-${Date.now()}`;
    const controller = new AbortController();
    this.activeStreams.set(streamId, controller);

    try {
      if (request.type === 'music') {
        // Special handling for music generation
        yield* this.streamMusicContent<T>(request, controller);
      } else {
        // Universal content streaming
        yield* this.streamUniversalContent<T>(request, controller);
      }
    } catch (error) {
      console.error(`Streaming error for ${request.type}:`, error);
    } finally {
      this.activeStreams.delete(streamId);
    }
  }

  private async *streamMusicContent<T>(
    request: ContentRequest,
    controller: AbortController
  ): AsyncGenerator<T> {
    if (request.query) {
      const musicStream = await musicSSEService.generateMusic({
        prompt: request.query,
        style: 'electronic',
        duration: 120
      });

      for await (const musicResponse of musicStream) {
        if (controller.signal.aborted) break;
        
        // Convert music response to unified content format
        const unifiedContent = await this.convertMusicToUnified(musicResponse);
        yield unifiedContent as T;
      }
    }
  }

  private async *streamUniversalContent<T>(
    request: ContentRequest,
    controller: AbortController
  ): AsyncGenerator<T> {
    const batchSize = 3;
    const totalBatches = Math.ceil((request.count || 10) / batchSize);

    for (let batch = 0; batch < totalBatches; batch++) {
      if (controller.signal.aborted) break;

      const strategy = this.getStreamingStrategy(request.type, batch);
      const batchContent = await unifiedContentEngine.generateContent(
        request.type,
        batchSize,
        strategy
      ) as T[];

      for (const item of batchContent) {
        if (controller.signal.aborted) break;
        yield item;
        
        // Add realistic delay between items
        await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));
      }
    }
  }

  // Ready for your Python script integration
  async setupBackendIntegration(config: {
    pollenEndpoint?: string;
    pythonScripts?: { [key: string]: string };
    enableSSE?: boolean;
  }): Promise<void> {
    // Configure Pollen AI with runtime endpoint
    if (config.pollenEndpoint) {
      pollenAI.configure({
        apiUrl: config.pollenEndpoint,
        enableSSE: config.enableSSE || true
      });
      
      const isHealthy = await pollenAI.checkHealth();
      if (isHealthy) {
        console.log('✅ Pollen AI integration ready');
      } else {
        console.warn('⚠️ Pollen AI backend not available, using fallback mode');
      }
    }

    // Setup Python script connections
    if (config.pythonScripts) {
      for (const [scriptName, endpoint] of Object.entries(config.pythonScripts)) {
        await universalSSE.setupPythonScript(scriptName, endpoint);
        console.log(`✅ Python script connected: ${scriptName}`);
      }
    }

    console.log('🚀 Backend integration complete');
  }

  // Continuous content refresh for all sections
  startContinuousGeneration(
    sections: ContentType[],
    intervalMs: number = 30000
  ): () => void {
    const intervals: NodeJS.Timeout[] = [];

    for (const section of sections) {
      const interval = setInterval(async () => {
        try {
          const request: ContentRequest = {
            type: section,
            count: 5,
            realtime: true,
            strategy: this.getDefaultStrategy(section)
          };

          await this.generateContent(request);
          console.log(`🔄 Refreshed content for ${section}`);
        } catch (error) {
          console.error(`Failed to refresh ${section}:`, error);
        }
      }, intervalMs + Math.random() * 10000); // Add jitter

      intervals.push(interval);
    }

    // Return cleanup function
    return () => {
      intervals.forEach(clearInterval);
      console.log('🛑 Stopped continuous generation');
    };
  }

  private async convertMusicToUnified(musicResponse: any): Promise<UnifiedContent> {
    return {
      id: musicResponse.id,
      type: 'music',
      title: musicResponse.title,
      description: `AI-generated music track: ${musicResponse.title}`,
      timestamp: new Date().toISOString(),
      significance: 8.5,
      trending: true,
      quality: 9,
      views: Math.floor(Math.random() * 10000),
      engagement: 85,
      impact: 'high',
      tags: ['AI Generated', 'Music'],
      category: 'Music',
      artist: musicResponse.artist,
      album: 'AI Compositions',
      duration: '3:24',
      plays: '1.2K',
      genre: 'Electronic',
      thumbnail: 'bg-gradient-to-br from-purple-500 to-pink-500'
    } as any;
  }

  private getDefaultStrategy(type: ContentType): GenerationStrategy {
    const strategies = {
      social: { diversity: 0.8, freshness: 0.9, personalization: 0.5, qualityThreshold: 7.0, trendingBoost: 1.3 },
      explore: { diversity: 0.9, freshness: 0.8, personalization: 0.6, qualityThreshold: 7.5, trendingBoost: 1.2 },
      shop: { diversity: 0.6, freshness: 0.7, personalization: 0.4, qualityThreshold: 8.0, trendingBoost: 1.4 },
      entertainment: { diversity: 0.7, freshness: 0.8, personalization: 0.5, qualityThreshold: 7.5, trendingBoost: 1.3 },
      games: { diversity: 0.8, freshness: 0.8, personalization: 0.7, qualityThreshold: 7.0, trendingBoost: 1.4 },
      music: { diversity: 0.9, freshness: 0.9, personalization: 0.8, qualityThreshold: 8.0, trendingBoost: 1.2 },
      news: { diversity: 0.6, freshness: 0.95, personalization: 0.3, qualityThreshold: 8.5, trendingBoost: 1.5 }
    };

    return strategies[type] || strategies.social;
  }

  private getStreamingStrategy(type: ContentType, batch: number): GenerationStrategy {
    const base = this.getDefaultStrategy(type);
    return {
      ...base,
      freshness: Math.max(0.5, base.freshness - batch * 0.1),
      diversity: Math.min(1.0, base.diversity + batch * 0.05)
    };
  }

  private buildCacheKey(request: ContentRequest): string {
    return `content-${request.type}-${JSON.stringify(request.strategy)}-${request.count}`;
  }

  private async getCachedContent<T>(key: string): Promise<ContentResponse<T> | null> {
    try {
      const cached = await storageService.getItem(key);
      if (cached && typeof cached === 'object' && cached !== null && 'timestamp' in cached && 'metadata' in cached) {
        const typedCached = cached as ContentResponse<T> & { timestamp: number };
        if (Date.now() - typedCached.timestamp < this.cacheExpiry) {
          return { ...typedCached, metadata: { ...typedCached.metadata, cacheHit: true } };
        }
      }
    } catch (error) {
      console.warn('Cache retrieval failed:', error);
    }
    return null;
  }

  private async cacheContent<T>(key: string, content: ContentResponse<T>): Promise<void> {
    try {
      await storageService.setItem(key, { ...content, timestamp: Date.now() });
    } catch (error) {
      console.warn('Cache storage failed:', error);
    }
  }

  private calculateAverageQuality(content: UnifiedContent[]): number {
    const total = content.reduce((sum, item) => sum + item.quality, 0);
    return Number((total / content.length).toFixed(1));
  }

  private calculateEnhancedQuality(enhancedContent: any[]): number {
    if (enhancedContent.length === 0) return 8.0;
    
    const total = enhancedContent.reduce((sum, item) => {
      const metrics = item.qualityMetrics;
      const qualityScore = (metrics.truthfulness + metrics.factualAccuracy + 
                           (10 - metrics.bias) + metrics.originality + metrics.relevance) / 5;
      return sum + qualityScore;
    }, 0);
    
    return Number((total / enhancedContent.length).toFixed(1));
  }

  stopStream(streamId: string): void {
    const controller = this.activeStreams.get(streamId);
    if (controller) {
      controller.abort();
      this.activeStreams.delete(streamId);
    }
  }

  stopAllStreams(): void {
    for (const [streamId] of this.activeStreams) {
      this.stopStream(streamId);
    }
  }
}

export const contentOrchestrator = new ContentOrchestrator();