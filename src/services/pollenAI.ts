
// Production Pollen AI - connects to Vercel backend
import { PollenResponse, PollenConfig } from './pollenTypes';

// Environment-based API configuration
const API_BASE_URL = import.meta.env.PROD 
  ? '/api'  // Use relative path for Vercel deployment
  : '/api';  // Use same for development (works with current setup)

class PollenAI {
  private config: PollenConfig & { endpoint: string };

  constructor(config: PollenConfig = {}) {
    this.config = {
      endpoint: config.endpoint || config.apiUrl || API_BASE_URL,
      timeout: 30000,
      ...config
    };
  }

  async generate(prompt: string, mode: string = 'chat', context?: any): Promise<PollenResponse> {
    try {
      const response = await fetch(`${this.config.endpoint}/ai/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          mode,
          type: mode === 'chat' ? 'general' : mode,
          context
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      return {
        content: data.content || 'No content generated',
        confidence: data.confidence || 0.8,
        learning: data.learning || false,
        reasoning: data.reasoning || undefined
      };
    } catch (error) {
      console.error('Pollen AI generation failed:', error);
      throw error;
    }
  }

  async batchGenerate(requests: Array<{prompt: string; mode: string; context?: any}>): Promise<PollenResponse[]> {
    const results = await Promise.allSettled(
      requests.map(req => this.generate(req.prompt, req.mode, req.context))
    );

    return results.map(result => 
      result.status === 'fulfilled' 
        ? result.value 
        : {
            content: 'Generation failed',
            confidence: 0,
            learning: false,
            reasoning: 'Batch generation error'
          }
    );
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.generate('test', 'general');
      return response.confidence > 0;
    } catch {
      return false;
    }
  }

  async fetchContent(type: string = 'feed', limit: number = 20): Promise<any[]> {
    try {
      const response = await fetch(`${this.config.endpoint}/content/feed?type=${type}&limit=${limit}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const result = await response.json();
      return result.data || [];
    } catch (error) {
      console.error('Failed to fetch content:', error);
      return [];
    }
  }

  // Legacy compatibility methods - delegate to main generate method
  async connect(config?: PollenConfig | string): Promise<boolean> {
    // Connection is handled automatically in fetch requests
    if (typeof config === 'string') {
      this.config.endpoint = config;
    } else if (config) {
      this.config = { ...this.config, ...config };
      if (config.apiUrl) {
        this.config.endpoint = config.apiUrl;
      }
    }
    return true;
  }

  async getMemoryStats(): Promise<any> {
    return { totalMemory: '100MB', usedMemory: '45MB', efficiency: 0.85 };
  }

  async proposeTask(inputText: string): Promise<any> {
    return this.generate(inputText, 'task_proposal');
  }

  async solveTask(inputText: string): Promise<any> {
    return this.generate(inputText, 'task_solution');
  }

  async createAdvertisement(inputText: string): Promise<any> {
    return this.generate(inputText, 'advertisement');
  }

  async generateMusic(inputText: string): Promise<any> {
    return this.generate(inputText, 'music');
  }

  async generateImage(prompt: string): Promise<any> {
    return this.generate(prompt, 'image');
  }

  async automateTask(inputText: string, schedule?: string): Promise<any> {
    return this.generate(`${inputText} ${schedule ? `Schedule: ${schedule}` : ''}`, 'automation');
  }

  async curateSocialPost(inputText: string): Promise<any> {
    return this.generate(inputText, 'social_post');
  }

  async analyzeTrends(inputText: string): Promise<any> {
    return this.generate(inputText, 'trend_analysis');
  }

  async *generateStream(prompt: string, mode: string = 'chat', context?: any): AsyncGenerator<any> {
    // Simple fallback - return single result as stream
    const result = await this.generate(prompt, mode, context);
    yield result;
  }
}

export const pollenAI = new PollenAI();
