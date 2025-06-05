// Pollen AI Service - Enhanced Frontend Integration
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
  private generationQueue: Map<string, Promise<PollenResponse>>;

  constructor() {
    this.baseUrl = 'http://localhost:8000';
    this.localMemory = new Map();
    this.isLearning = true;
    this.generationQueue = new Map();
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
      { pattern: "personal productivity", category: "personal", weight: 3.9 },
      { pattern: "automation workflows", category: "automation", weight: 4.1 },
      { pattern: "social networks", category: "social", weight: 3.7 }
    ];
    
    this.localMemory.set('patterns', basePatterns);
    this.localMemory.set('interactions', []);
    this.localMemory.set('shortTermMemory', []);
  }

  async generate(prompt: string, mode: string = "social"): Promise<PollenResponse> {
    // Prevent duplicate simultaneous generations
    const cacheKey = `${prompt}-${mode}`;
    if (this.generationQueue.has(cacheKey)) {
      return this.generationQueue.get(cacheKey)!;
    }

    const generationPromise = this.performGeneration(prompt, mode);
    this.generationQueue.set(cacheKey, generationPromise);

    try {
      const result = await generationPromise;
      return result;
    } finally {
      this.generationQueue.delete(cacheKey);
    }
  }

  private async performGeneration(prompt: string, mode: string): Promise<PollenResponse> {
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
      
      // Fallback to enhanced local generation
      return this.generateLocally(prompt, mode);
    }
  }

  private generateLocally(prompt: string, mode: string): PollenResponse {
    // Enhanced templates with more variety
    const templates = {
      social: [
        "🚀 The convergence of AI and human creativity is reshaping how we think about innovation. What started as algorithmic assistance has evolved into true collaborative intelligence.",
        "💡 Fascinating patterns emerging in distributed networks - seeing how collective intelligence naturally forms when the right conditions are met.",
        "🌟 Building the future isn't about replacing human intuition with AI logic, but creating synergies that amplify both human creativity and machine precision.",
        "🔬 Experimenting with new models of human-AI collaboration. The most interesting breakthroughs happen at the intersection of structure and spontaneity.",
        "📊 Data visualization reveals hidden narratives - every dataset contains stories waiting to be discovered through the right analytical lens.",
        "🎨 Creative AI isn't about automation, it's about expansion - giving human imagination new tools to explore previously impossible territories.",
        "🌐 Decentralized systems are teaching us that intelligence emerges from connection patterns, not just individual computational power.",
        "⚡ The future of work isn't human vs AI - it's about creating collaborative ecosystems where both human intuition and machine learning thrive together."
      ],
      entertainment: [
        "🎬 Interactive Narrative: 'The Quantum Mirror' - A story that adapts based on your choices, creating infinite narrative possibilities.",
        "🎮 Procedural Game World: 'Infinite Gardens' - An ever-evolving ecosystem where your actions reshape the digital landscape in real-time.",
        "🎵 AI Symphony Generator: 'Harmonic Convergence' - Music that evolves with your mood, creating personalized soundscapes for any moment.",
        "📚 Collaborative Fiction Engine: 'Shared Worlds' - Stories where multiple creators contribute, and AI weaves their ideas into cohesive narratives.",
        "🎭 Virtual Performance Space: 'The Digital Stage' - Interactive theater where audience participation shapes the story's direction.",
        "🎪 Adaptive Entertainment Hub: 'Wonder Algorithms' - Content that learns your preferences and creates unique experiences tailored just for you."
      ],
      news: [
        "🔬 Breakthrough in quantum error correction brings practical quantum computing closer to reality, with implications for cryptography and scientific simulation.",
        "🌱 Revolutionary carbon capture technology developed by international research consortium shows promise for reversing atmospheric CO2 levels.",
        "🤖 New AI ethics framework proposed by leading technologists emphasizes human agency and transparent decision-making in automated systems.",
        "🌍 Global collaboration on climate technology accelerates as nations share breakthrough innovations in renewable energy storage.",
        "💊 Gene therapy advances offer new hope for rare diseases, with personalized treatments showing remarkable success in clinical trials.",
        "🔒 Privacy-preserving computation methods advance, enabling secure data analysis without compromising individual information."
      ],
      automation: [
        "⚙️ Workflow Optimization: Smart task routing system reduces manual coordination by 60% while maintaining quality standards.",
        "🔄 Process Intelligence: AI-powered workflow analysis identifies bottlenecks and suggests improvements in real-time.",
        "📋 Dynamic Project Management: Adaptive scheduling that responds to changing priorities and resource availability.",
        "🎯 Goal-Oriented Automation: Systems that understand objectives and autonomously adjust processes to achieve desired outcomes.",
        "🔧 Self-Healing Workflows: Automation that detects and corrects errors, maintaining system reliability without human intervention.",
        "📈 Performance Analytics: Continuous monitoring and optimization of automated processes for maximum efficiency."
      ],
      shop: [
        "🛍️ Curated AI Tools Collection: Discover productivity-enhancing software tailored to your workflow patterns and professional goals.",
        "💎 Premium Digital Assets: High-quality design resources, templates, and creative tools from verified creators worldwide.",
        "🎯 Personalized Recommendations: AI-powered product discovery that learns from your preferences and suggests relevant items.",
        "🔧 Professional Services: Expert consultations, custom development, and specialized solutions for your unique challenges.",
        "📚 Knowledge Products: Courses, guides, and resources created by industry experts and thought leaders.",
        "⚡ Productivity Boosters: Tools and services designed to streamline your workflow and enhance your professional capabilities."
      ],
      community: [
        "🌐 Global Knowledge Exchange: Connecting experts across disciplines to solve complex challenges through collaborative intelligence.",
        "🤝 Skill Sharing Networks: Community-driven learning where members teach and learn from each other's expertise.",
        "💡 Innovation Circles: Small groups focused on exploring emerging technologies and their practical applications.",
        "🎯 Project Collaboration Hubs: Spaces where community members form teams to work on meaningful initiatives together.",
        "📱 Peer Learning Networks: Informal knowledge sharing that happens through daily interactions and shared experiences.",
        "🔮 Future-Building Communities: Groups dedicated to envisioning and creating better technological and social systems."
      ]
    };

    const modeTemplates = templates[mode as keyof typeof templates] || templates.social;
    let content = modeTemplates[Math.floor(Math.random() * modeTemplates.length)];
    
    // Add prompt context if provided
    if (prompt && prompt.length > 10) {
      const keywords = this.extractKeywords(prompt);
      if (keywords.length > 0) {
        content += `\n\nExploring: ${keywords.slice(0, 3).join(', ')} through the lens of ${mode} innovation.`;
      }
    }
    
    this.addToMemory(prompt, content, mode);
    
    return {
      content,
      confidence: Math.random() * 0.25 + 0.7, // 0.7-0.95
      learning: this.isLearning,
      reasoning: `Generated ${mode} content using enhanced local patterns and contextual templates`
    };
  }

  private extractKeywords(text: string): string[] {
    const words = text.toLowerCase().split(/\s+/);
    const stopWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about'];
    return words
      .filter(word => word.length > 3 && !stopWords.includes(word))
      .slice(0, 5);
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

    // Update patterns with better keyword extraction
    const patterns = this.localMemory.get('patterns') || [];
    const keywords = this.extractKeywords(input);
    
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
