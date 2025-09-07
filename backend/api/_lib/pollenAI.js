import axios from 'axios';

class PollenAI {
  constructor() {
    // IMPORTANT: Your actual Pollen AI model's API endpoint
    this.baseURL = process.env.POLLEN_AI_ENDPOINT || 'http://localhost:8000';
    this.apiKey = process.env.POLLEN_AI_API_KEY;
  }

  async generate(type, prompt, mode = 'chat') {
    try {
      // This is where you call YOUR custom model
      const response = await axios.post(`${this.baseURL}/generate`, {
        type,
        prompt,
        mode
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });
      return response.data;
    } catch (error) {
      console.error('Pollen AI generation failed:', error.message);
      // Return a mock response if your model isn't running
      return this.getMockResponse(type, prompt);
    }
  }

  getMockResponse(type, prompt) {
    console.log(`Returning MOCK response for type: ${type}`);
    const responses = {
      feed_post: {
        content: `AI Breakthrough in Material Science: Researchers have developed a new AI model that can predict the properties of undiscovered materials, accelerating innovation. Based on prompt: "${prompt}"`,
        confidence: 0.85,
        reasoning: "Generated based on current AI research trends and material science developments",
        title: "AI Breakthrough in Material Science (Mock)",
        industry: "tech",
        impact_level: "High",
      },
      general: {
        content: `Here's a thoughtful response to your query: "${prompt}". This demonstrates the AI's capability to understand and respond contextually.`,
        confidence: 0.82,
        reasoning: "Mock response demonstrating general conversation capabilities",
      },
      music: {
        content: "Upbeat electronic track with synthesized beats and ambient soundscapes",
        confidence: 0.88,
        title: "Electronic Fusion",
        genre: "Electronic",
        mood: "Energetic"
      },
      product: {
        content: "Innovative smart home device with AI-powered automation",
        confidence: 0.90,
        name: "SmartHome AI Hub",
        price: 299.99,
        category: "Smart Home"
      }
    };
    return responses[type] || responses.general;
  }
}

export const pollenAI = new PollenAI();