/**
 * Unified Pollen AI Integration Service
 * 
 * Central service for all AI operations across the platform
 * Ensures all AI features use Pollen AI backend exclusively
 */

import axios from 'axios';

const POLLEN_AI_URL = import.meta.env.VITE_POLLEN_AI_URL || 'http://localhost:8000';

export interface PollenAIRequest {
  prompt: string;
  mode: 'social' | 'news' | 'entertainment' | 'shop' | 'wellness' | 'chat' | 'analysis';
  type?: string;
  context?: Record<string, any>;
  use_cache?: boolean;
  compression_level?: 'high' | 'medium' | 'low';
}

export interface PollenAIResponse {
  content: string;
  confidence: number;
  learning: boolean;
  reasoning?: string;
  cached?: boolean;
  compressed?: boolean;
  processing_time_ms?: number;
}

export interface PollenAIHealth {
  status: string;
  model_version: string;
  optimizations: {
    edge_caching: string;
    request_batching: string;
    response_quantization: string;
    compression: string;
  };
  performance?: {
    cache: {
      hit_rate: string;
      compression_ratio: string;
      memory_saved: string;
    };
  };
}

class PollenAIUnifiedService {
  private static instance: PollenAIUnifiedService;
  private baseURL: string;
  private healthy: boolean = false;
  private healthCheckInterval: NodeJS.Timeout | null = null;

  private constructor() {
    this.baseURL = POLLEN_AI_URL;
    this.startHealthCheck();
  }

  static getInstance(): PollenAIUnifiedService {
    if (!PollenAIUnifiedService.instance) {
      PollenAIUnifiedService.instance = new PollenAIUnifiedService();
    }
    return PollenAIUnifiedService.instance;
  }

  /**
   * Start periodic health checks
   */
  private startHealthCheck() {
    this.checkHealth();
    this.healthCheckInterval = setInterval(() => {
      this.checkHealth();
    }, 30000); // Every 30 seconds
  }

  /**
   * Check Pollen AI backend health
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await axios.get<PollenAIHealth>(`${this.baseURL}/health`, {
        timeout: 5000
      });
      this.healthy = response.data.status === 'healthy';
      return this.healthy;
    } catch (error) {
      this.healthy = false;
      return false;
    }
  }

  /**
   * Generate content using Pollen AI
   */
  async generate(request: PollenAIRequest): Promise<PollenAIResponse> {
    try {
      const response = await axios.post<PollenAIResponse>(
        `${this.baseURL}/generate`,
        {
          prompt: request.prompt,
          mode: request.mode,
          type: request.type || 'general',
          context: request.context,
          use_cache: request.use_cache !== false,
          compression_level: request.compression_level || 'medium'
        },
        { timeout: 30000 }
      );

      return response.data;
    } catch (error) {
      console.error('Pollen AI generation error:', error);
      throw new Error('Failed to generate content with Pollen AI');
    }
  }

  /**
   * Generate social media content
   */
  async generateSocial(prompt: string): Promise<string> {
    const response = await this.generate({
      prompt,
      mode: 'social',
      type: 'post',
      compression_level: 'medium'
    });
    return response.content;
  }

  /**
   * Generate news content
   */
  async generateNews(prompt: string): Promise<string> {
    const response = await this.generate({
      prompt,
      mode: 'news',
      type: 'article',
      compression_level: 'medium'
    });
    return response.content;
  }

  /**
   * Generate entertainment content
   */
  async generateEntertainment(prompt: string): Promise<string> {
    const response = await this.generate({
      prompt,
      mode: 'entertainment',
      type: 'content',
      compression_level: 'medium'
    });
    return response.content;
  }

  /**
   * Generate product/shopping content
   */
  async generateProduct(prompt: string): Promise<string> {
    const response = await this.generate({
      prompt,
      mode: 'shop',
      type: 'product',
      compression_level: 'medium'
    });
    return response.content;
  }

  /**
   * Generate wellness content
   */
  async generateWellness(prompt: string): Promise<string> {
    const response = await this.generate({
      prompt,
      mode: 'wellness',
      type: 'tip',
      compression_level: 'medium'
    });
    return response.content;
  }

  /**
   * Analyze content with AI
   */
  async analyze(content: string, analysisType: string): Promise<string> {
    const response = await this.generate({
      prompt: `Analyze the following content for ${analysisType}: ${content}`,
      mode: 'analysis',
      type: analysisType,
      compression_level: 'low' // Less compression for analytical content
    });
    return response.content;
  }

  /**
   * Generate insights from data
   */
  async generateInsights(data: any, context?: string): Promise<string> {
    const prompt = context 
      ? `Generate insights about: ${context}. Data: ${JSON.stringify(data)}`
      : `Generate insights from this data: ${JSON.stringify(data)}`;

    const response = await this.generate({
      prompt,
      mode: 'analysis',
      type: 'insights',
      compression_level: 'low'
    });
    return response.content;
  }

  /**
   * Enhance existing content with AI
   */
  async enhanceContent(content: string, style: 'professional' | 'casual' | 'technical' = 'professional'): Promise<string> {
    const response = await this.generate({
      prompt: `Enhance this content in a ${style} style: ${content}`,
      mode: 'chat',
      type: 'enhancement',
      compression_level: 'medium'
    });
    return response.content;
  }

  /**
   * Generate personalized recommendations
   */
  async generateRecommendations(userPreferences: any, category: string): Promise<string[]> {
    const response = await this.generate({
      prompt: `Generate personalized ${category} recommendations based on: ${JSON.stringify(userPreferences)}`,
      mode: 'analysis',
      type: 'recommendations',
      context: { preferences: userPreferences, category }
    });

    // Parse recommendations from response
    const lines = response.content.split('\n')
      .filter(line => line.trim().length > 0)
      .slice(0, 5);
    
    return lines;
  }

  /**
   * Get optimization statistics
   */
  async getOptimizationStats() {
    try {
      const response = await axios.get(`${this.baseURL}/optimization/stats`, {
        timeout: 5000
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get optimization stats:', error);
      return null;
    }
  }

  /**
   * Clear Pollen AI cache
   */
  async clearCache() {
    try {
      const response = await axios.post(`${this.baseURL}/optimization/clear-cache`);
      return response.data;
    } catch (error) {
      console.error('Failed to clear cache:', error);
      return null;
    }
  }

  /**
   * Check if Pollen AI is available
   */
  isHealthy(): boolean {
    return this.healthy;
  }

  /**
   * Get base URL
   */
  getBaseURL(): string {
    return this.baseURL;
  }

  /**
   * Cleanup
   */
  destroy() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }
}

// Export singleton instance
export const pollenAI = PollenAIUnifiedService.getInstance();

// Export helper functions for common operations
export const generateWithPollenAI = async (
  prompt: string,
  mode: PollenAIRequest['mode'] = 'chat'
): Promise<string> => {
  const response = await pollenAI.generate({ prompt, mode });
  return response.content;
};

export const enhanceWithPollenAI = async (
  content: string,
  style: 'professional' | 'casual' | 'technical' = 'professional'
): Promise<string> => {
  return pollenAI.enhanceContent(content, style);
};

export const analyzeWithPollenAI = async (
  content: string,
  analysisType: string
): Promise<string> => {
  return pollenAI.analyze(content, analysisType);
};

export default pollenAI;
