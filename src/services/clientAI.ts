import { pipeline } from '@huggingface/transformers';

interface AITask {
  id: string;
  type: 'sentiment' | 'summarization' | 'classification' | 'embedding' | 'generation';
  status: 'pending' | 'processing' | 'completed' | 'error';
  input: any;
  output?: any;
  error?: string;
  progress?: number;
}

class ClientAIService {
  private static instance: ClientAIService;
  private pipelines: Map<string, any> = new Map();
  private isLoading: Set<string> = new Set();
  private tasks: Map<string, AITask> = new Map();

  static getInstance(): ClientAIService {
    if (!ClientAIService.instance) {
      ClientAIService.instance = new ClientAIService();
    }
    return ClientAIService.instance;
  }

  // Initialize AI models on demand
  async initializePipeline(task: string, model?: string): Promise<any> {
    if (this.pipelines.has(task)) {
      return this.pipelines.get(task)!;
    }

    if (this.isLoading.has(task)) {
      // Wait for existing initialization
      return new Promise((resolve) => {
        const checkInterval = setInterval(() => {
          if (this.pipelines.has(task)) {
            clearInterval(checkInterval);
            resolve(this.pipelines.get(task)!);
          }
        }, 100);
      });
    }

    this.isLoading.add(task);

    try {
      let pipelineTask: string;
      let modelName: string;

      switch (task) {
        case 'sentiment':
          pipelineTask = 'sentiment-analysis';
          modelName = model || 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';
          break;
        case 'summarization':
          pipelineTask = 'summarization';
          modelName = model || 'Xenova/distilbart-cnn-12-6';
          break;
        case 'classification':
          pipelineTask = 'zero-shot-classification';
          modelName = model || 'Xenova/distilbert-base-uncased-mnli';
          break;
        case 'embedding':
          pipelineTask = 'feature-extraction';
          modelName = model || 'Xenova/all-MiniLM-L6-v2';
          break;
        case 'generation':
          pipelineTask = 'text-generation';
          modelName = model || 'Xenova/distilgpt2';
          break;
        default:
          throw new Error(`Unknown task: ${task}`);
      }

      console.log(`Loading AI model for ${task}:`, modelName);
      const pipe = await pipeline(pipelineTask as any, modelName, {
        device: 'webgpu', // Use WebGPU if available
      });

      this.pipelines.set(task, pipe);
      return pipe;
    } catch (error) {
      console.error(`Failed to load AI model for ${task}:`, error);
      throw error;
    } finally {
      this.isLoading.delete(task);
    }
  }

  // Analyze sentiment of text
  async analyzeSentiment(text: string): Promise<{ label: string; score: number }> {
    const taskId = this.createTask('sentiment', text);
    
    try {
      this.updateTaskStatus(taskId, 'processing');
      const pipe = await this.initializePipeline('sentiment');
      const result = await pipe(text);
      
      const output = Array.isArray(result) ? result[0] : result;
      this.updateTaskStatus(taskId, 'completed', output);
      
      return output;
    } catch (error) {
      this.updateTaskStatus(taskId, 'error', null, (error as Error).message);
      throw error;
    }
  }

  // Summarize text
  async summarizeText(text: string, maxLength: number = 150): Promise<{ summary_text: string }> {
    const taskId = this.createTask('summarization', { text, maxLength });
    
    try {
      this.updateTaskStatus(taskId, 'processing');
      const pipe = await this.initializePipeline('summarization');
      const result = await pipe(text, { max_length: maxLength, min_length: 30 });
      
      const output = Array.isArray(result) ? result[0] : result;
      this.updateTaskStatus(taskId, 'completed', output);
      
      return output;
    } catch (error) {
      this.updateTaskStatus(taskId, 'error', null, (error as Error).message);
      throw error;
    }
  }

  // Classify text into categories
  async classifyText(text: string, labels: string[]): Promise<{ labels: string[]; scores: number[] }> {
    const taskId = this.createTask('classification', { text, labels });
    
    try {
      this.updateTaskStatus(taskId, 'processing');
      const pipe = await this.initializePipeline('classification');
      const result = await pipe(text, labels);
      
      this.updateTaskStatus(taskId, 'completed', result);
      return result;
    } catch (error) {
      this.updateTaskStatus(taskId, 'error', null, (error as Error).message);
      throw error;
    }
  }

  // Generate text embeddings
  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    const taskId = this.createTask('embedding', texts);
    
    try {
      this.updateTaskStatus(taskId, 'processing');
      const pipe = await this.initializePipeline('embedding');
      const result = await pipe(texts, { pooling: 'mean', normalize: true });
      
      const embeddings = result.tolist();
      this.updateTaskStatus(taskId, 'completed', embeddings);
      
      return embeddings;
    } catch (error) {
      this.updateTaskStatus(taskId, 'error', null, (error as Error).message);
      throw error;
    }
  }

  // Generate text
  async generateText(prompt: string, maxLength: number = 100): Promise<{ generated_text: string }> {
    const taskId = this.createTask('generation', { prompt, maxLength });
    
    try {
      this.updateTaskStatus(taskId, 'processing');
      const pipe = await this.initializePipeline('generation');
      const result = await pipe(prompt, { 
        max_length: maxLength,
        temperature: 0.7,
        do_sample: true,
      });
      
      const output = Array.isArray(result) ? result[0] : result;
      this.updateTaskStatus(taskId, 'completed', output);
      
      return output;
    } catch (error) {
      this.updateTaskStatus(taskId, 'error', null, (error as Error).message);
      throw error;
    }
  }

  // Smart content analysis (combines multiple AI tasks)
  async analyzeContent(content: string): Promise<{
    sentiment: { label: string; score: number };
    summary: { summary_text: string };
    categories: { labels: string[]; scores: number[] };
    keywords: string[];
  }> {
    const categories = [
      'technology', 'science', 'business', 'entertainment', 'sports', 
      'politics', 'health', 'education', 'travel', 'food'
    ];

    try {
      const [sentiment, summary, classification] = await Promise.all([
        this.analyzeSentiment(content),
        content.length > 200 ? this.summarizeText(content) : Promise.resolve({ summary_text: content }),
        this.classifyText(content, categories),
      ]);

      // Extract keywords (simple approach)
      const keywords = this.extractKeywords(content);

      return {
        sentiment,
        summary,
        categories: classification,
        keywords,
      };
    } catch (error) {
      console.error('Content analysis failed:', error);
      throw error;
    }
  }

  // Simple keyword extraction
  private extractKeywords(text: string): string[] {
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 3);

    const frequency: { [key: string]: number } = {};
    words.forEach(word => {
      frequency[word] = (frequency[word] || 0) + 1;
    });

    return Object.entries(frequency)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([word]) => word);
  }

  private createTask(type: AITask['type'], input: any): string {
    const taskId = `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const task: AITask = {
      id: taskId,
      type,
      status: 'pending',
      input,
    };
    
    this.tasks.set(taskId, task);
    return taskId;
  }

  private updateTaskStatus(
    taskId: string, 
    status: AITask['status'], 
    output?: any, 
    error?: string
  ): void {
    const task = this.tasks.get(taskId);
    if (task) {
      task.status = status;
      if (output !== undefined) task.output = output;
      if (error) task.error = error;
    }
  }

  getTask(taskId: string): AITask | undefined {
    return this.tasks.get(taskId);
  }

  getAllTasks(): AITask[] {
    return Array.from(this.tasks.values());
  }

  clearCompletedTasks(): void {
    for (const [id, task] of this.tasks.entries()) {
      if (task.status === 'completed' || task.status === 'error') {
        this.tasks.delete(id);
      }
    }
  }

  // Check if WebGPU is available
  async checkWebGPUSupport(): Promise<boolean> {
    try {
      if ('gpu' in navigator) {
        const adapter = await (navigator as any).gpu.requestAdapter();
        return !!adapter;
      }
      return false;
    } catch {
      return false;
    }
  }
}

export const clientAI = ClientAIService.getInstance();