import { pollenAI } from './pollenAI';

export interface SSEConfig {
  endpoint: string;
  reconnectInterval?: number;
  maxRetries?: number;
  enablePollen?: boolean;
}

export interface SSEResponse<T = any> {
  id: string;
  type: string;
  data: T;
  timestamp: string;
  status: 'connecting' | 'connected' | 'generating' | 'complete' | 'error';
  progress?: number;
  error?: string;
}

class UniversalSSEService {
  private connections = new Map<string, EventSource>();
  private configs = new Map<string, SSEConfig>();
  
  // Ready for your backend Python scripts integration
  async connect(connectionId: string, config: SSEConfig): Promise<void> {
    if (this.connections.has(connectionId)) {
      this.disconnect(connectionId);
    }

    this.configs.set(connectionId, config);
    
    return new Promise((resolve, reject) => {
      try {
        const eventSource = new EventSource(config.endpoint);
        
        eventSource.onopen = () => {
          console.log(`SSE connected: ${connectionId}`);
          resolve();
        };
        
        eventSource.onerror = (error) => {
          console.error(`SSE error for ${connectionId}:`, error);
          reject(error);
        };
        
        this.connections.set(connectionId, eventSource);
      } catch (error) {
        reject(error);
      }
    });
  }

  async *streamContent<T>(
    connectionId: string, 
    request: any,
    type: 'feed' | 'explore' | 'shop' | 'music' | 'learning'
  ): AsyncGenerator<SSEResponse<T>> {
    const config = this.configs.get(connectionId);
    
    if (!config) {
      throw new Error(`No configuration found for connection: ${connectionId}`);
    }

    const requestId = `${type}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Yield initial response
    yield {
      id: requestId,
      type,
      data: null as T,
      timestamp: new Date().toISOString(),
      status: 'connecting'
    };

    try {
      // Use Pollen AI for intelligent content generation
      if (config.enablePollen) {
        yield* this.streamWithPollen<T>(requestId, type, request);
      } else {
        // Fallback to direct API streaming
        yield* this.streamWithAPI<T>(requestId, type, request, config);
      }
    } catch (error) {
      yield {
        id: requestId,
        type,
        data: null as T,
        timestamp: new Date().toISOString(),
        status: 'error',
        error: error instanceof Error ? error.message : 'Stream failed'
      };
    }
  }

  private async *streamWithPollen<T>(
    requestId: string,
    type: string,
    request: any
  ): AsyncGenerator<SSEResponse<T>> {
    yield {
      id: requestId,
      type,
      data: null as T,
      timestamp: new Date().toISOString(),
      status: 'connected'
    };

    // Generate content using Pollen AI streaming
    const prompt = this.buildPrompt(type, request);
    const pollenStream = pollenAI.generateStream(prompt, type, request);

    let progress = 0;
    for await (const pollenResponse of pollenStream) {
      progress += 20;
      
      yield {
        id: requestId,
        type,
        data: pollenResponse as T,
        timestamp: new Date().toISOString(),
        status: progress >= 100 ? 'complete' : 'generating',
        progress: Math.min(progress, 100)
      };
    }
  }

  private async *streamWithAPI<T>(
    requestId: string,
    type: string,
    request: any,
    config: SSEConfig
  ): AsyncGenerator<SSEResponse<T>> {
    // Direct API streaming (for your Python scripts)
    const response = await fetch(config.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify({ type, request, requestId })
    });

    if (!response.body) {
      throw new Error('No response body for streaming');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield {
                id: requestId,
                type,
                data: data as T,
                timestamp: new Date().toISOString(),
                status: data.status || 'generating',
                progress: data.progress
              };
            } catch (e) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private buildPrompt(type: string, request: any): string {
    switch (type) {
      case 'feed':
        return `Generate engaging social media content about: ${JSON.stringify(request)}`;
      case 'explore':
        return `Create diverse exploratory content about: ${JSON.stringify(request)}`;
      case 'shop':
        return `Generate product information for: ${JSON.stringify(request)}`;
      case 'music':
        return `Create music content for: ${JSON.stringify(request)}`;
      case 'learning':
        return `Generate educational content about: ${JSON.stringify(request)}`;
      default:
        return `Generate content for ${type}: ${JSON.stringify(request)}`;
    }
  }

  disconnect(connectionId: string): void {
    const connection = this.connections.get(connectionId);
    if (connection) {
      connection.close();
      this.connections.delete(connectionId);
      this.configs.delete(connectionId);
      console.log(`SSE disconnected: ${connectionId}`);
    }
  }

  disconnectAll(): void {
    for (const [connectionId] of this.connections) {
      this.disconnect(connectionId);
    }
  }

  // Helper methods for your Python script integration
  async setupPythonScript(scriptName: string, endpoint: string): Promise<string> {
    const connectionId = `python-${scriptName}`;
    
    await this.connect(connectionId, {
      endpoint,
      reconnectInterval: 5000,
      maxRetries: 3,
      enablePollen: false // Direct Python script connection
    });
    
    return connectionId;
  }

  async setupPollenIntegration(endpoint?: string): Promise<string> {
    const connectionId = 'pollen-main';
    
    await this.connect(connectionId, {
      endpoint: endpoint || 'ws://localhost:8000/stream',
      reconnectInterval: 3000,
      maxRetries: 5,
      enablePollen: true
    });
    
    return connectionId;
  }
}

export const universalSSE = new UniversalSSEService();