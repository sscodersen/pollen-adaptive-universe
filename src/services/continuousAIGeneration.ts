import { workerBotClient } from './workerBotClient';

interface GenerationConfig {
  enabled: boolean;
  intervalMinutes: number;
  maxConcurrentTasks: number;
  contentTypes: Array<'social' | 'music' | 'entertainment' | 'news' | 'shop' | 'wellness'>;
}

class ContinuousAIGeneration {
  private config: GenerationConfig = {
    enabled: false,
    intervalMinutes: 15, // Every 15 minutes - sustainable rate
    maxConcurrentTasks: 3,
    contentTypes: ['social', 'wellness', 'news'],
  };

  private intervalId: NodeJS.Timeout | null = null;
  private activeTasks = 0;
  private totalGenerated = 0;

  start(customConfig?: Partial<GenerationConfig>) {
    if (customConfig) {
      this.config = { ...this.config, ...customConfig };
    }

    if (this.intervalId) {
      console.log('⚠️ Continuous generation already running');
      return;
    }

    this.config.enabled = true;
    const intervalMs = this.config.intervalMinutes * 60 * 1000;

    console.log(`🔄 Starting continuous AI generation (every ${this.config.intervalMinutes} min)`);
    
    // Generate immediately on start
    this.generateBatch();

    // Then schedule regular generation
    this.intervalId = setInterval(() => {
      if (this.config.enabled) {
        this.generateBatch();
      }
    }, intervalMs);
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
      this.config.enabled = false;
      console.log('🛑 Stopped continuous AI generation');
    }
  }

  private async generateBatch() {
    if (this.activeTasks >= this.config.maxConcurrentTasks) {
      console.log('⏸️ Skipping generation - max concurrent tasks reached');
      return;
    }

    const contentType = this.getNextContentType();
    const prompt = this.getPromptForType(contentType);

    this.activeTasks++;

    try {
      console.log(`🤖 Generating ${contentType} content via Pollen AI`);
      
      await workerBotClient.generateContent(
        prompt,
        contentType,
        'continuous_generation'
      );

      this.totalGenerated++;
      console.log(`✅ Generated ${contentType} content (Total: ${this.totalGenerated})`);
    } catch (error) {
      console.error(`❌ Failed to generate ${contentType} content:`, error);
    } finally {
      this.activeTasks--;
    }
  }

  private getNextContentType() {
    const types = this.config.contentTypes;
    return types[Math.floor(Math.random() * types.length)];
  }

  private getPromptForType(type: string): string {
    const prompts: Record<string, string[]> = {
      social: [
        'Latest innovations in sustainable technology and their impact on communities',
        'Emerging trends in digital wellness and mental health awareness',
        'Breakthrough developments in renewable energy and climate solutions',
      ],
      wellness: [
        'Science-backed wellness practices for modern lifestyle',
        'Innovative approaches to holistic health and well-being',
        'Mental health strategies for digital age professionals',
      ],
      news: [
        'Positive developments in global technology and innovation',
        'Environmental progress and sustainability achievements',
        'Scientific breakthroughs improving quality of life',
      ],
      music: [
        'Curate a wellness-focused ambient playlist for meditation',
        'Create an energizing playlist for morning motivation',
        'Generate a focus-enhancing playlist for productivity',
      ],
      entertainment: [
        'Discover inspiring documentaries about human innovation',
        'Explore creative content celebrating cultural diversity',
        'Find uplifting entertainment promoting positive values',
      ],
      shop: [
        'Eco-friendly products for sustainable living',
        'Innovative tech gadgets for productivity and wellness',
        'Ethical brands promoting social impact',
      ],
    };

    const typePrompts = prompts[type] || prompts.social;
    return typePrompts[Math.floor(Math.random() * typePrompts.length)];
  }

  getStatus() {
    return {
      enabled: this.config.enabled,
      intervalMinutes: this.config.intervalMinutes,
      activeTasks: this.activeTasks,
      totalGenerated: this.totalGenerated,
      contentTypes: this.config.contentTypes,
    };
  }

  updateConfig(config: Partial<GenerationConfig>) {
    const wasEnabled = this.config.enabled;
    
    this.config = { ...this.config, ...config };
    
    if (wasEnabled && !this.config.enabled) {
      this.stop();
    } else if (!wasEnabled && this.config.enabled) {
      this.start();
    }

    console.log('⚙️ Continuous generation config updated:', this.config);
  }
}

export const continuousAIGeneration = new ContinuousAIGeneration();
