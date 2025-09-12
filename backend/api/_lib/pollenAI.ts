import axios from 'axios';

class PollenAI {
  constructor() {
    // IMPORTANT: Your actual Pollen AI model's API endpoint
    this.baseURL = process.env.POLLEN_AI_ENDPOINT || 'http://localhost:8000';
    this.apiKey = process.env.POLLEN_AI_API_KEY;
  }

  async generate(type, prompt, mode = 'chat') {
    try {
      // Connect to real Pollen AI service (no mock fallbacks)
      const response = await axios.post(`${this.baseURL}/generate`, {
        prompt,
        mode,
        type
      }, {
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` })
        },
        timeout: 10000
      });
      
      if (response.data && response.data.content) {
        console.log(`✅ Real AI response generated for ${type} mode: ${mode}`);
        return {
          content: response.data.content,
          confidence: response.data.confidence || 0.8,
          reasoning: response.data.reasoning || 'AI-generated content',
          type: type,
          mode: mode,
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error('Invalid response from Pollen AI service');
      }
    } catch (error) {
      console.error('Pollen AI generation failed:', error.message);
      // NO MOCK FALLBACKS - throw error for production reliability
      throw new Error(`Failed to generate content: ${error.message}`);
    }
  }

  // Mock responses removed - production uses real Pollen AI only
}

export const pollenAI = new PollenAI();