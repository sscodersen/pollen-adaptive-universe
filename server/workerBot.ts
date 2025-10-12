import { EventEmitter } from 'events';
import axios from 'axios';

// Pollen AI Backend Configuration
const POLLEN_AI_URL = process.env.POLLEN_AI_ENDPOINT || 'http://localhost:8000';

interface Task {
  id: string;
  type: 'content' | 'music' | 'ads' | 'trends' | 'analytics' | 'personalization';
  payload: any;
  priority: number;
  createdAt: Date;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: any;
  error?: string;
}

interface SSEClient {
  id: string;
  response: any;
  userId?: string;
}

class WorkerBotService extends EventEmitter {
  private taskQueue: Task[] = [];
  private processingTasks: Map<string, Task> = new Map();
  private sseClients: Map<string, SSEClient> = new Map();
  private isProcessing = false;
  private pollenAIAvailable = false;

  constructor() {
    super();
    this.initializePollenAI();
    this.startProcessing();
  }

  private async initializePollenAI() {
    try {
      const response = await axios.get(`${POLLEN_AI_URL}/health`, { timeout: 3000 });
      if (response.status === 200) {
        this.pollenAIAvailable = true;
        console.log('✅ Worker Bot: Connected to Pollen AI');
      }
    } catch (error) {
      console.log('⚠️ Worker Bot: Pollen AI not available, will retry');
      setTimeout(() => this.initializePollenAI(), 5000);
    }
  }

  // SSE Management
  addSSEClient(clientId: string, response: any, userId?: string) {
    this.sseClients.set(clientId, { id: clientId, response, userId });
    
    // Send initial connection message
    this.sendSSEMessage(clientId, {
      type: 'connected',
      message: 'Worker Bot connected',
      clientId
    });

    console.log(`✅ SSE client connected: ${clientId}`);
  }

  removeSSEClient(clientId: string) {
    this.sseClients.delete(clientId);
    console.log(`🔌 SSE client disconnected: ${clientId}`);
  }

  private sendSSEMessage(clientId: string, data: any) {
    const client = this.sseClients.get(clientId);
    if (client && client.response.writable) {
      client.response.write(`data: ${JSON.stringify(data)}\n\n`);
    }
  }

  private broadcastSSE(data: any, userId?: string) {
    this.sseClients.forEach((client) => {
      if (!userId || client.userId === userId) {
        this.sendSSEMessage(client.id, data);
      }
    });
  }

  // Task Queue Management
  addTask(task: Omit<Task, 'id' | 'createdAt' | 'status'>): string {
    const newTask: Task = {
      ...task,
      id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date(),
      status: 'pending'
    };

    this.taskQueue.push(newTask);
    this.taskQueue.sort((a, b) => b.priority - a.priority);

    console.log(`📋 Task queued: ${newTask.id} (${newTask.type})`);
    
    this.broadcastSSE({
      type: 'task_queued',
      task: { id: newTask.id, type: newTask.type, priority: newTask.priority }
    });

    return newTask.id;
  }

  getTaskStatus(taskId: string): Task | undefined {
    return this.processingTasks.get(taskId) || 
           this.taskQueue.find(t => t.id === taskId);
  }

  // Task Processing
  private async startProcessing() {
    setInterval(() => this.processNextTask(), 1000);
  }

  private async processNextTask() {
    if (this.isProcessing || this.taskQueue.length === 0) return;

    const task = this.taskQueue.shift();
    if (!task) return;

    this.isProcessing = true;
    task.status = 'processing';
    this.processingTasks.set(task.id, task);

    this.broadcastSSE({
      type: 'task_started',
      task: { id: task.id, type: task.type }
    });

    try {
      let result;
      
      switch (task.type) {
        case 'content':
          result = await this.generateContent(task.payload);
          break;
        case 'music':
          result = await this.generateMusic(task.payload);
          break;
        case 'ads':
          result = await this.generateAds(task.payload);
          break;
        case 'trends':
          result = await this.analyzeTrends(task.payload);
          break;
        case 'analytics':
          result = await this.performAnalytics(task.payload);
          break;
        case 'personalization':
          result = await this.personalizeContent(task.payload);
          break;
        default:
          throw new Error(`Unknown task type: ${task.type}`);
      }

      task.status = 'completed';
      task.result = result;

      this.broadcastSSE({
        type: 'task_completed',
        task: { id: task.id, type: task.type, result }
      });

      console.log(`✅ Task completed: ${task.id}`);
    } catch (error: any) {
      task.status = 'failed';
      task.error = error.message;

      this.broadcastSSE({
        type: 'task_failed',
        task: { id: task.id, type: task.type, error: error.message }
      });

      console.error(`❌ Task failed: ${task.id}`, error);
    } finally {
      this.isProcessing = false;
      
      // Keep task in history for 5 minutes
      setTimeout(() => {
        this.processingTasks.delete(task.id);
      }, 5 * 60 * 1000);
    }
  }

  // AI Task Handlers - Using Pollen AI
  private async generateContent(payload: any) {
    const { prompt, type = 'general', userId } = payload;

    try {
      const response = await axios.post(`${POLLEN_AI_URL}/generate`, {
        prompt,
        mode: this.getModeFromType(type),
        type,
        context: { userId }
      }, { timeout: 10000 });

      return {
        ...response.data,
        source: 'pollen-ai'
      };
    } catch (error: any) {
      console.error('Pollen AI error:', error.message);
      return this.fallbackContent(payload);
    }
  }

  private async generateMusic(payload: any) {
    const { mood, genre, occasion } = payload;

    try {
      const prompt = `Create a ${mood} ${genre} playlist for ${occasion}. Include 5-10 curated tracks.`;
      const response = await axios.post(`${POLLEN_AI_URL}/generate`, {
        prompt,
        mode: 'entertainment',
        type: 'music',
        context: { mood, genre, occasion }
      }, { timeout: 10000 });

      return {
        ...response.data,
        source: 'pollen-ai'
      };
    } catch (error: any) {
      console.error('Pollen AI music error:', error.message);
      return this.fallbackMusic(payload);
    }
  }

  private async generateAds(payload: any) {
    const { targetAudience, product, goals } = payload;

    try {
      const prompt = `Create ethical advertising for ${product} targeting ${targetAudience} with goals: ${goals}`;
      const response = await axios.post(`${POLLEN_AI_URL}/generate`, {
        prompt,
        mode: 'social',
        type: 'product',
        context: { targetAudience, product, goals }
      }, { timeout: 10000 });

      return {
        ...response.data,
        source: 'pollen-ai'
      };
    } catch (error: any) {
      console.error('Pollen AI ads error:', error.message);
      return this.fallbackAds(payload);
    }
  }

  private async analyzeTrends(payload: any) {
    const { data, timeRange, category } = payload;

    try {
      const prompt = `Analyze ${category} trends over ${timeRange}. Identify patterns and momentum.`;
      const response = await axios.post(`${POLLEN_AI_URL}/generate`, {
        prompt,
        mode: 'analysis',
        type: 'trend_analysis',
        context: { data, timeRange, category }
      }, { timeout: 10000 });

      return {
        ...response.data,
        source: 'pollen-ai'
      };
    } catch (error: any) {
      console.error('Pollen AI trends error:', error.message);
      return this.fallbackTrends(payload);
    }
  }

  private async performAnalytics(payload: any) {
    const { userData, metrics, insights } = payload;

    try {
      const prompt = `Analyze user data and provide actionable insights based on metrics and patterns.`;
      const response = await axios.post(`${POLLEN_AI_URL}/generate`, {
        prompt,
        mode: 'analysis',
        type: 'analytics',
        context: { userData, metrics, insights }
      }, { timeout: 10000 });

      return {
        ...response.data,
        source: 'pollen-ai'
      };
    } catch (error: any) {
      console.error('Pollen AI analytics error:', error.message);
      return this.fallbackAnalytics(payload);
    }
  }

  private async personalizeContent(payload: any) {
    const { userProfile, contentPool, preferences } = payload;

    try {
      const prompt = `Personalize content recommendations based on user preferences and behavior patterns.`;
      const response = await axios.post(`${POLLEN_AI_URL}/generate`, {
        prompt,
        mode: 'chat',
        type: 'personalization',
        context: { userProfile, contentPool, preferences }
      }, { timeout: 10000 });

      return {
        ...response.data,
        source: 'pollen-ai'
      };
    } catch (error: any) {
      console.error('Pollen AI personalization error:', error.message);
      return this.fallbackPersonalization(payload);
    }
  }

  // Fallback methods (when OpenAI is not available)
  private fallbackContent(payload: any) {
    return {
      content: `[Demo Mode] Generated content for: ${payload.prompt}`,
      type: payload.type || 'general',
      confidence: 0.8,
      source: 'fallback'
    };
  }

  private fallbackMusic(payload: any) {
    return {
      tracks: [
        {
          title: `${payload.mood} Vibes`,
          artist: 'Demo Artist',
          mood: payload.mood,
          genre: payload.genre,
          duration: 180,
          previewUrl: '#'
        }
      ],
      source: 'fallback'
    };
  }

  private fallbackAds(payload: any) {
    return {
      ads: [
        {
          headline: `Discover ${payload.product}`,
          body: 'Experience something amazing',
          cta: 'Learn More',
          targeting: payload.targetAudience,
          ethicsScore: 0.9
        }
      ],
      source: 'fallback'
    };
  }

  private fallbackTrends(payload: any) {
    return {
      trends: [
        {
          title: `${payload.category} Trend`,
          description: 'Emerging pattern detected',
          confidence: 0.75,
          category: payload.category,
          momentum: 'rising'
        }
      ],
      source: 'fallback'
    };
  }

  private fallbackAnalytics(payload: any) {
    return {
      insights: [
        {
          type: 'engagement',
          description: 'User engagement pattern detected',
          impact: 'medium',
          recommendation: 'Continue monitoring'
        }
      ],
      metrics: payload.metrics || {},
      source: 'fallback'
    };
  }

  private fallbackPersonalization(payload: any) {
    return {
      recommendations: payload.contentPool?.slice(0, 5).map((item: any, i: number) => ({
        id: item.id || `item_${i}`,
        score: 0.8 - (i * 0.1),
        reason: 'Based on your interests',
        type: item.type || 'content'
      })) || [],
      source: 'fallback'
    };
  }

  private getModeFromType(type: string): string {
    const modeMap: Record<string, string> = {
      general: 'chat',
      wellness: 'wellness',
      community: 'community',
      agriculture: 'agriculture',
      feed_post: 'social',
      news: 'news',
      product: 'shop',
      entertainment: 'entertainment',
      learning: 'learning'
    };

    return modeMap[type] || 'chat';
  }

  // Statistics
  getStats() {
    return {
      queueLength: this.taskQueue.length,
      processingTasks: this.processingTasks.size,
      connectedClients: this.sseClients.size,
      hasPollenAI: this.pollenAIAvailable,
      isProcessing: this.isProcessing
    };
  }
}

export const workerBot = new WorkerBotService();
