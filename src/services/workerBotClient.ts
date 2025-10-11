import axios from 'axios';
import { analyticsEngine } from './analyticsEngine';

interface WorkerBotTask {
  id: string;
  type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: any;
  error?: string;
}

interface WorkerBotMessage {
  type: 'connected' | 'task_queued' | 'task_started' | 'task_completed' | 'task_failed';
  task?: any;
  message?: string;
  clientId?: string;
}

class WorkerBotClient {
  private eventSource: EventSource | null = null;
  private callbacks: Map<string, ((data: any) => void)[]> = new Map();
  private taskCallbacks: Map<string, ((result: any) => void)> = new Map();
  private connected = false;

  connect(userId?: string) {
    if (this.connected) return;

    const params = userId ? `?userId=${userId}` : '';
    this.eventSource = new EventSource(`/api/worker/stream${params}`);

    this.eventSource.onmessage = (event) => {
      try {
        const data: WorkerBotMessage = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('Error parsing SSE message:', error);
      }
    };

    this.eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      this.connected = false;
      
      // Attempt reconnect after 5 seconds
      setTimeout(() => {
        this.disconnect();
        this.connect(userId);
      }, 5000);
    };

    this.eventSource.onopen = () => {
      this.connected = true;
      console.log('✅ Worker Bot connected');
    };
  }

  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.connected = false;
    }
  }

  private handleMessage(data: WorkerBotMessage) {
    // Trigger type-specific callbacks
    const typeCallbacks = this.callbacks.get(data.type) || [];
    typeCallbacks.forEach(cb => cb(data));

    // Trigger global callbacks
    const globalCallbacks = this.callbacks.get('*') || [];
    globalCallbacks.forEach(cb => cb(data));

    // Handle task completion callbacks
    if (data.type === 'task_completed' && data.task) {
      const taskCallback = this.taskCallbacks.get(data.task.id);
      if (taskCallback) {
        taskCallback(data.task.result);
        this.taskCallbacks.delete(data.task.id);
      }
    }

    // Track analytics events
    if (data.type === 'task_completed') {
      analyticsEngine.trackEvent('system', 'worker_bot_task_completed', {
        taskType: data.task?.type,
        taskId: data.task?.id
      });
    }
  }

  on(eventType: string, callback: (data: any) => void) {
    if (!this.callbacks.has(eventType)) {
      this.callbacks.set(eventType, []);
    }
    this.callbacks.get(eventType)!.push(callback);
  }

  off(eventType: string, callback: (data: any) => void) {
    const callbacks = this.callbacks.get(eventType);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  // Task submission methods
  async submitTask(type: string, payload: any, priority = 5): Promise<string> {
    const response = await axios.post('/api/worker/tasks', {
      type,
      payload,
      priority
    });
    return response.data.taskId;
  }

  async submitTaskAndWait<T = any>(type: string, payload: any, priority = 5, timeout = 30000): Promise<T> {
    const taskId = await this.submitTask(type, payload, priority);
    
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.taskCallbacks.delete(taskId);
        reject(new Error('Task timeout'));
      }, timeout);

      this.taskCallbacks.set(taskId, (result) => {
        clearTimeout(timeoutId);
        resolve(result);
      });
    });
  }

  async getTaskStatus(taskId: string): Promise<WorkerBotTask | null> {
    try {
      const response = await axios.get(`/api/worker/tasks/${taskId}`);
      return response.data.task;
    } catch (error) {
      return null;
    }
  }

  async getStats() {
    try {
      const response = await axios.get('/api/worker/stats');
      return response.data.stats;
    } catch (error) {
      console.error('Error fetching worker bot stats:', error);
      return null;
    }
  }

  // Quick action methods
  async generateContent(prompt: string, type = 'general', userId?: string) {
    return this.submitTaskAndWait('content', { prompt, type, userId });
  }

  async generateMusic(mood: string, genre: string, occasion: string) {
    return this.submitTaskAndWait('music', { mood, genre, occasion });
  }

  async generateAds(targetAudience: string, product: string, goals: string) {
    return this.submitTaskAndWait('ads', { targetAudience, product, goals });
  }

  async analyzeTrends(data: any, timeRange: string, category: string) {
    return this.submitTaskAndWait('trends', { data, timeRange, category });
  }

  async performAnalytics(userData: any, metrics: any, insights: any) {
    return this.submitTaskAndWait('analytics', { userData, metrics, insights });
  }

  async personalizeContent(userProfile: any, contentPool: any, preferences: any) {
    return this.submitTaskAndWait('personalization', { userProfile, contentPool, preferences });
  }

  isConnected() {
    return this.connected;
  }
}

export const workerBotClient = new WorkerBotClient();
export type { WorkerBotTask, WorkerBotMessage };
