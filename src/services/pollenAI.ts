
export interface PollenResponse {
  content: string;
  confidence: number;
  learning: boolean;
  reasoning?: string;
}

class PollenAI {
  getMemoryStats() {
    return {
      shortTermSize: 0,
      longTermPatterns: 0,
      isLearning: true,
      topPatterns: [],
    };
  }

  generate(prompt: string, mode: string): Promise<PollenResponse> {
    console.log(`Pollen AI generating for prompt: "${prompt}" in mode: ${mode}`);
    return Promise.resolve({
      content: `This is a dummy response for: "${prompt}"`,
      confidence: 0.95,
      learning: true,
      reasoning: 'This is a mock reasoning.',
    });
  }

  clearMemory() {
    console.log('Pollen AI memory cleared.');
  }
}

export const pollenAI = new PollenAI();
