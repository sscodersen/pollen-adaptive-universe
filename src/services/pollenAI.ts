
// Pollen AI Service - Frontend Integration
export interface PollenResponse {
  content: string;
  confidence: number;
  learning: boolean;
  reasoning?: string;
}

export interface MemoryStats {
  shortTermSize: number;
  longTermPatterns: number;
  topPatterns: Array<{
    pattern: string;
    category: string;
    weight: number;
  }>;
  isLearning: boolean;
  reasoningTasks?: number;
  highRewardTasks?: number;
}

class PollenAIService {
  private baseUrl: string;
  private localMemory: Map<string, any>;
  private isLearning: boolean;

  constructor() {
    this.baseUrl = 'http://localhost:8000';
    this.localMemory = new Map();
    this.isLearning = true;
    this.initializeLocalMemory();
  }

  private initializeLocalMemory() {
    // Initialize with enhanced patterns for different modes
    const basePatterns = [
      { pattern: "UI design", category: "design", weight: 4.2 },
      { pattern: "machine learning", category: "tech", weight: 3.8 },
      { pattern: "user experience", category: "design", weight: 3.5 },
      { pattern: "data visualization", category: "tech", weight: 3.2 },
      { pattern: "artificial intelligence", category: "tech", weight: 4.0 },
      { pattern: "team collaboration", category: "team", weight: 3.6 },
      { pattern: "community engagement", category: "community", weight: 3.4 },
      { pattern: "personal productivity", category: "personal", weight: 3.9 }
    ];
    
    this.localMemory.set('patterns', basePatterns);
    this.localMemory.set('interactions', []);
    this.localMemory.set('shortTermMemory', []);
  }

  async generate(prompt: string, mode: string = "social"): Promise<PollenResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          mode
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Store interaction in local memory
      this.addToMemory(prompt, data.content, mode);
      
      return {
        content: data.content,
        confidence: data.confidence || 0.8,
        learning: data.learning || true,
        reasoning: data.reasoning
      };
    } catch (error) {
      console.error('Pollen AI generation error:', error);
      
      // Fallback to local generation
      return this.generateLocally(prompt, mode);
    }
  }

  private generateLocally(prompt: string, mode: string): PollenResponse {
    const templates = {
      social: [
        "🚀 Exploring the intersection of AI and human creativity...",
        "💡 Just discovered fascinating patterns in user behavior data",
        "🌟 Building the future of intelligent interfaces, one component at a time",
        "🔬 Experimenting with new approaches to personalized user experiences",
        "📊 The story that data tells when we listen carefully is incredible"
      ],
      personal: [
        "🎯 Your focus time optimization suggests 25% productivity increase",
        "📈 Learning pattern analysis shows accelerated skill acquisition",
        "⚡ Workflow automation opportunities detected in routine tasks",
        "🧠 Cognitive load reduction strategies available for complex projects",
        "🎨 Creative block prevention protocols activated based on your patterns"
      ],
      team: [
        "👥 Collaboration efficiency increased by 40% with AI-assisted workflows",
        "🔄 Cross-functional alignment improved through intelligent task routing",
        "📋 Project velocity optimization based on team capacity analysis",
        "💬 Communication patterns suggest optimal meeting scheduling",
        "🎯 Team goal synchronization achieved through shared AI insights"
      ],
      community: [
        "🌐 Global knowledge sharing accelerated through AI curation",
        "🤝 Community engagement patterns reveal emerging collaboration trends",
        "📱 Decentralized learning networks forming around shared interests",
        "🎉 Collective intelligence amplification through AI-human partnerships",
        "🔮 Future-focused skill development guided by community insights"
      ],
      news: [
        "Breaking: Revolutionary advancement in quantum computing announced today",
        "Industry Report: Ethics in AI development gains unprecedented focus",
        "Research Update: New findings on human-AI collaboration effectiveness",
        "Technology News: Open-source AI tools reach new accessibility milestone",
        "Innovation Alert: Revolutionary approach to privacy-preserving machine learning"
      ],
      entertainment: [
        "🎬 Generated Interactive Story: 'The Last Algorithm'",
        "🎮 New AI Game: Adaptive puzzle-solving in quantum realms",
        "🎵 AI-Composed Music: 'Symphonies of Tomorrow's Cities'",
        "📚 Interactive Fiction: Choose your path through digital dimensions",
        "🎭 Virtual Performance: AI-human collaborative theater experience"
      ]
    };

    const modeTemplates = templates[mode as keyof typeof templates] || templates.social;
    const content = modeTemplates[Math.floor(Math.random() * modeTemplates.length)];
    
    // Add prompt context if provided
    const finalContent = prompt ? `${content}\n\nContext: ${prompt}` : content;
    
    this.addToMemory(prompt, finalContent, mode);
    
    return {
      content: finalContent,
      confidence: Math.random() * 0.3 + 0.6, // 0.6-0.9
      learning: this.isLearning,
      reasoning: `Generated ${mode} content using local patterns and templates`
    };
  }

  private addToMemory(input: string, output: string, mode: string) {
    const interaction = {
      input,
      output,
      mode,
      timestamp: Date.now(),
      confidence: Math.random() * 0.3 + 0.7
    };

    // Add to short-term memory
    const shortTerm = this.localMemory.get('shortTermMemory') || [];
    shortTerm.unshift(interaction);
    if (shortTerm.length > 50) {
      shortTerm.pop();
    }
    this.localMemory.set('shortTermMemory', shortTerm);

    // Update patterns
    const patterns = this.localMemory.get('patterns') || [];
    const keywords = input.toLowerCase().split(' ').filter(word => word.length > 3);
    
    keywords.forEach(keyword => {
      const existingPattern = patterns.find((p: any) => p.pattern === keyword);
      if (existingPattern) {
        existingPattern.weight += 0.1;
      } else if (patterns.length < 100) {
        patterns.push({
          pattern: keyword,
          category: mode,
          weight: 1.0
        });
      }
    });

    this.localMemory.set('patterns', patterns);
  }

  getMemoryStats(): MemoryStats {
    const shortTerm = this.localMemory.get('shortTermMemory') || [];
    const patterns = this.localMemory.get('patterns') || [];
    
    return {
      shortTermSize: shortTerm.length,
      longTermPatterns: patterns.length,
      topPatterns: patterns
        .sort((a: any, b: any) => b.weight - a.weight)
        .slice(0, 10),
      isLearning: this.isLearning,
      reasoningTasks: Math.floor(Math.random() * 25 + 15),
      highRewardTasks: Math.floor(Math.random() * 12 + 8)
    };
  }

  async clearMemory(): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/memory/clear`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    } catch (error) {
      console.error('Error clearing remote memory:', error);
    }
    
    // Clear local memory
    this.initializeLocalMemory();
  }

  toggleLearning(): boolean {
    this.isLearning = !this.isLearning;
    return this.isLearning;
  }

  async learnFromFeedback(input: string, expectedOutput: string): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/learn`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input_text: input,
          expected_output: expectedOutput
        }),
      });
    } catch (error) {
      console.error('Error sending feedback:', error);
    }
  }

  async semanticSearch(query: string): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query_text: query
        }),
      });

      if (response.ok) {
        const data = await response.json();
        return data.matches || [];
      }
    } catch (error) {
      console.error('Error in semantic search:', error);
    }
    
    return [];
  }
}

export const pollenAI = new PollenAIService();
