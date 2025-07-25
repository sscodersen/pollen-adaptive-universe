
// Refactored Pollen AI - now uses modular core
import { PollenCore } from './pollenCore';
import { PollenResponse, PollenConfig } from './pollenTypes';

class PollenAI extends PollenCore {
  constructor(config: PollenConfig = {}) {
    super(config);
  }

  // Additional high-level methods can be added here
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
}

export const pollenAI = new PollenAI();
