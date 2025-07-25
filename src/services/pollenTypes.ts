// Type definitions for Pollen AI system

export interface PollenResponse {
  content: string;
  confidence: number;
  learning: boolean;
  reasoning?: string;
  metadata?: {
    contentType?: string;
    category?: string;
    tags?: string[];
    quality?: number;
  };
}

export interface PollenConfig {
  apiUrl?: string;
  apiKey?: string;
  modelVersion?: string;
  enableSSE?: boolean;
}

export interface PollenMemoryStats {
  shortTermSize: number;
  longTermPatterns: number;
  isLearning: boolean;
  topPatterns: string[];
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
}

export interface PollenGenerationRequest {
  prompt: string;
  mode: string;
  context?: any;
  temperature?: number;
  maxTokens?: number;
  streaming?: boolean;
}

export interface PollenStreamResponse extends PollenResponse {
  isComplete: boolean;
  chunkIndex: number;
}