import OpenAI from 'openai';
import { EventEmitter } from 'events';

// the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
const OPENAI_MODEL = 'gpt-5';

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
  private openai: OpenAI | null = null;
  private isProcessing = false;

  constructor() {
    super();
    this.initializeOpenAI();
    this.startProcessing();
  }

  private initializeOpenAI() {
    if (process.env.OPENAI_API_KEY) {
      this.openai = new OpenAI({ 
        apiKey: process.env.OPENAI_API_KEY 
      });
      console.log('✅ Worker Bot: OpenAI initialized');
    } else {
      console.log('⚠️ Worker Bot: Running in fallback mode (no OpenAI key)');
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

  // AI Task Handlers
  private async generateContent(payload: any) {
    if (!this.openai) {
      return this.fallbackContent(payload);
    }

    const { prompt, type = 'general', userId } = payload;

    const systemPrompt = this.getSystemPrompt(type);
    
    const response = await this.openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt }
      ],
      response_format: { type: 'json_object' },
      max_completion_tokens: 2048
    });

    return JSON.parse(response.choices[0].message.content || '{}');
  }

  private async generateMusic(payload: any) {
    if (!this.openai) {
      return this.fallbackMusic(payload);
    }

    const { mood, genre, occasion } = payload;

    const response = await this.openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: [
        { 
          role: 'system', 
          content: 'You are a music curator AI. Generate music recommendations with metadata. Respond with JSON in this format: { "tracks": [{"title": string, "artist": string, "mood": string, "genre": string, "duration": number, "previewUrl": string}] }'
        },
        { 
          role: 'user', 
          content: `Create a playlist for mood: ${mood}, genre: ${genre}, occasion: ${occasion}. Include 5-10 tracks.`
        }
      ],
      response_format: { type: 'json_object' }
    });

    return JSON.parse(response.choices[0].message.content || '{}');
  }

  private async generateAds(payload: any) {
    if (!this.openai) {
      return this.fallbackAds(payload);
    }

    const { targetAudience, product, goals } = payload;

    const response = await this.openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: [
        { 
          role: 'system', 
          content: 'You are an advertising AI. Create ethical, engaging ad content. Respond with JSON in this format: { "ads": [{"headline": string, "body": string, "cta": string, "targeting": object, "ethicsScore": number}] }'
        },
        { 
          role: 'user', 
          content: `Create ads for product: ${product}, target audience: ${targetAudience}, goals: ${goals}`
        }
      ],
      response_format: { type: 'json_object' }
    });

    return JSON.parse(response.choices[0].message.content || '{}');
  }

  private async analyzeTrends(payload: any) {
    if (!this.openai) {
      return this.fallbackTrends(payload);
    }

    const { data, timeRange, category } = payload;

    const response = await this.openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: [
        { 
          role: 'system', 
          content: 'You are a trend analysis AI. Analyze data patterns and identify emerging trends. Respond with JSON in this format: { "trends": [{"title": string, "description": string, "confidence": number, "category": string, "momentum": string}] }'
        },
        { 
          role: 'user', 
          content: `Analyze trends in category: ${category}, timeRange: ${timeRange}, data: ${JSON.stringify(data)}`
        }
      ],
      response_format: { type: 'json_object' }
    });

    return JSON.parse(response.choices[0].message.content || '{}');
  }

  private async performAnalytics(payload: any) {
    if (!this.openai) {
      return this.fallbackAnalytics(payload);
    }

    const { userData, metrics, insights } = payload;

    const response = await this.openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: [
        { 
          role: 'system', 
          content: 'You are an analytics AI. Analyze user data and provide actionable insights. Respond with JSON in this format: { "insights": [{"type": string, "description": string, "impact": string, "recommendation": string}], "metrics": object }'
        },
        { 
          role: 'user', 
          content: `Analyze: ${JSON.stringify({ userData, metrics, insights })}`
        }
      ],
      response_format: { type: 'json_object' }
    });

    return JSON.parse(response.choices[0].message.content || '{}');
  }

  private async personalizeContent(payload: any) {
    if (!this.openai) {
      return this.fallbackPersonalization(payload);
    }

    const { userProfile, contentPool, preferences } = payload;

    const response = await this.openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: [
        { 
          role: 'system', 
          content: 'You are a personalization AI. Curate content based on user preferences and behavior. Respond with JSON in this format: { "recommendations": [{"id": string, "score": number, "reason": string, "type": string}] }'
        },
        { 
          role: 'user', 
          content: `Personalize for: ${JSON.stringify({ userProfile, preferences })}, from content pool: ${JSON.stringify(contentPool)}`
        }
      ],
      response_format: { type: 'json_object' }
    });

    return JSON.parse(response.choices[0].message.content || '{}');
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

  private getSystemPrompt(type: string): string {
    const prompts: Record<string, string> = {
      general: 'You are Pollen AI, an ethical AI assistant focused on wellness, community, and social impact. Generate helpful, accurate, and ethically-conscious content. Respond with JSON format.',
      wellness: 'You are a wellness AI expert. Provide health and wellness content that is evidence-based, supportive, and promotes holistic well-being. Respond with JSON format.',
      community: 'You are a community-focused AI. Generate content that fosters connection, inclusivity, and positive social impact. Respond with JSON format.',
      agriculture: 'You are an agricultural AI expert. Provide insights on sustainable farming, crop management, and agricultural innovation. Respond with JSON format.'
    };

    return prompts[type] || prompts.general;
  }

  // Statistics
  getStats() {
    return {
      queueLength: this.taskQueue.length,
      processingTasks: this.processingTasks.size,
      connectedClients: this.sseClients.size,
      hasOpenAI: !!this.openai,
      isProcessing: this.isProcessing
    };
  }
}

export const workerBot = new WorkerBotService();
