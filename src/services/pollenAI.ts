
import { toast } from "@/hooks/use-toast";

interface PollenConfig {
  modelEndpoint: string;
  maxTokens: number;
  temperature: number;
  learningRate: number;
}

interface PollenMemory {
  shortTerm: Array<{ input: string; output: string; timestamp: Date }>;
  longTerm: Array<{ pattern: string; weight: number; category: string }>;
  userPreferences: Record<string, any>;
}

interface PollenResponse {
  content: string;
  confidence: number;
  learning: boolean;
  memoryUpdated: boolean;
  reasoning?: string;
}

class PollenAI {
  private config: PollenConfig;
  private memory: PollenMemory;
  private isLearning: boolean = true;

  constructor() {
    this.config = {
      modelEndpoint: import.meta.env.VITE_POLLEN_ENDPOINT || 'http://localhost:8000',
      maxTokens: 2048,
      temperature: 0.7,
      learningRate: 0.01
    };

    this.memory = {
      shortTerm: [],
      longTerm: [],
      userPreferences: {}
    };

    this.loadMemoryFromStorage();
  }

  async generate(prompt: string, mode: string, context?: any): Promise<PollenResponse> {
    try {
      // Add to short-term memory
      const memoryContext = this.buildMemoryContext(mode);
      
      // Simulate API call to our Pollen backend
      const response = await this.callPollenAPI({
        prompt,
        mode,
        context,
        memory: memoryContext,
        config: this.config
      });

      // Update memory with new interaction
      this.updateMemory(prompt, response.content, mode);
      
      return response;
    } catch (error) {
      console.error('Pollen AI Error:', error);
      toast({
        title: "AI Error",
        description: "Pollen is experiencing issues. Falling back to local processing.",
        variant: "destructive"
      });
      
      return this.fallbackResponse(prompt, mode);
    }
  }

  private async callPollenAPI(payload: any): Promise<PollenResponse> {
    // This will call our Python FastAPI backend
    const response = await fetch(`${this.config.modelEndpoint}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-User-Session': this.getUserSession()
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Pollen API Error: ${response.statusText}`);
    }

    return await response.json();
  }

  private buildMemoryContext(mode: string) {
    return {
      recent: this.memory.shortTerm.slice(-10),
      relevant: this.memory.longTerm.filter(m => m.category === mode).slice(-5),
      preferences: this.memory.userPreferences
    };
  }

  private updateMemory(input: string, output: string, mode: string) {
    // Add to short-term memory
    this.memory.shortTerm.push({
      input,
      output,
      timestamp: new Date()
    });

    // Keep only last 100 short-term memories
    if (this.memory.shortTerm.length > 100) {
      this.memory.shortTerm = this.memory.shortTerm.slice(-100);
    }

    // Extract patterns for long-term memory
    this.extractPatterns(input, output, mode);
    
    // Persist to localStorage
    this.saveMemoryToStorage();
  }

  private extractPatterns(input: string, output: string, mode: string) {
    // Simple pattern extraction (would be more sophisticated in real implementation)
    const words = input.toLowerCase().split(' ');
    const significantWords = words.filter(w => w.length > 3);
    
    significantWords.forEach(word => {
      const existing = this.memory.longTerm.find(p => p.pattern === word && p.category === mode);
      if (existing) {
        existing.weight += 0.1;
      } else {
        this.memory.longTerm.push({
          pattern: word,
          weight: 1.0,
          category: mode
        });
      }
    });

    // Keep only top 1000 patterns
    this.memory.longTerm.sort((a, b) => b.weight - a.weight);
    this.memory.longTerm = this.memory.longTerm.slice(0, 1000);
  }

  private fallbackResponse(prompt: string, mode: string): PollenResponse {
    const fallbacks = {
      chat: `I understand you're asking about "${prompt}". While I'm currently learning and evolving, I can help you explore this topic. What specific aspect would you like to dive deeper into?`,
      
      code: `Here's a code approach for "${prompt}":

\`\`\`javascript
// Adaptive solution based on your request
function handleRequest() {
  // This implementation evolves based on patterns I've learned
  console.log('Processing: ${prompt}');
  
  // Your specific logic here
  return { success: true, data: 'result' };
}
\`\`\`

Would you like me to explain or modify this approach?`,

      creative: `Creative concept for "${prompt}":

🎨 **Vision**: Translating your idea into a unique creative expression
🔄 **Adaptive Elements**: Building on patterns I've learned from our interactions  
✨ **Evolution**: This concept grows more refined as we collaborate

What direction resonates with you?`,

      analysis: `Analysis of "${prompt}":

📊 **Pattern Recognition**: I'm identifying key elements and relationships
🧠 **Learning Mode**: Adapting my analysis based on your previous preferences
📈 **Insights**: Drawing from accumulated knowledge patterns

What specific aspects should I focus on?`
    };

    return {
      content: fallbacks[mode as keyof typeof fallbacks],
      confidence: 0.6,
      learning: true,
      memoryUpdated: true,
      reasoning: "Fallback response - local processing active"
    };
  }

  private getUserSession(): string {
    let session = localStorage.getItem('pollen-session');
    if (!session) {
      session = 'anon-' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('pollen-session', session);
    }
    return session;
  }

  private loadMemoryFromStorage() {
    try {
      const stored = localStorage.getItem('pollen-memory');
      if (stored) {
        const parsed = JSON.parse(stored);
        this.memory = {
          ...this.memory,
          ...parsed,
          shortTerm: parsed.shortTerm?.map((m: any) => ({
            ...m,
            timestamp: new Date(m.timestamp)
          })) || []
        };
      }
    } catch (error) {
      console.warn('Could not load Pollen memory:', error);
    }
  }

  private saveMemoryToStorage() {
    try {
      localStorage.setItem('pollen-memory', JSON.stringify(this.memory));
    } catch (error) {
      console.warn('Could not save Pollen memory:', error);
    }
  }

  // Public methods for memory management
  getMemoryStats() {
    return {
      shortTermSize: this.memory.shortTerm.length,
      longTermPatterns: this.memory.longTerm.length,
      topPatterns: this.memory.longTerm.slice(0, 10),
      isLearning: this.isLearning
    };
  }

  clearMemory() {
    this.memory = { shortTerm: [], longTerm: [], userPreferences: {} };
    localStorage.removeItem('pollen-memory');
    toast({
      title: "Memory Cleared",
      description: "Pollen's learning history has been reset."
    });
  }

  toggleLearning() {
    this.isLearning = !this.isLearning;
    return this.isLearning;
  }
}

// Singleton instance
export const pollenAI = new PollenAI();

// Export types
export type { PollenResponse, PollenMemory };
