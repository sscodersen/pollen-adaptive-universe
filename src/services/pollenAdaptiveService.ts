// Pollen Adaptive Intelligence Service
import { pollenAI } from './pollenAI';

export interface TaskProposal {
  id: string;
  title: string;
  description: string;
  complexity: 'low' | 'medium' | 'high';
  estimatedTime: string;
  category: string;
}

export interface TaskSolution {
  id: string;
  solution: string;
  confidence: number;
  steps: string[];
  reasoning: string;
}

export interface AdCreationResult {
  id: string;
  title: string;
  content: string;
  targetAudience: string;
  estimatedCTR: number;
  platform: string;
}

export interface MusicGenerationResult {
  id: string;
  title: string;
  genre: string;
  duration: string;
  audioUrl?: string;
  description: string;
}

export interface ImageGenerationResult {
  id: string;
  description: string;
  imageUrl?: string;
  style: string;
  dimensions: string;
}

export interface TaskAutomationResult {
  id: string;
  taskType: string;
  schedule: string;
  status: 'scheduled' | 'running' | 'completed' | 'failed';
  automationScript: string;
}

export interface SocialPostResult {
  id: string;
  platform: string;
  content: string;
  hashtags: string[];
  optimalPostTime: string;
  engagementScore: number;
}

export interface TrendAnalysisResult {
  id: string;
  topic: string;
  trendScore: number;
  insights: string[];
  predictions: string[];
  timeframe: string;
}

class PollenAdaptiveService {
  private generateId(): string {
    return `pollen-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async proposeTask(description: string): Promise<TaskProposal> {
    try {
      const response = await pollenAI.proposeTask(description);
      
      if (response.error) {
        throw new Error(response.error);
      }

      return {
        id: this.generateId(),
        title: response.title || `Task: ${description.substring(0, 50)}...`,
        description: response.description || description,
        complexity: response.complexity || 'medium',
        estimatedTime: response.estimatedTime || '30 minutes',
        category: response.category || 'general'
      };
    } catch (error) {
      console.error('Task proposal failed:', error);
      return {
        id: this.generateId(),
        title: `Task: ${description.substring(0, 50)}...`,
        description,
        complexity: 'medium',
        estimatedTime: '30 minutes',
        category: 'general'
      };
    }
  }

  async solveTask(problem: string): Promise<TaskSolution> {
    try {
      const response = await pollenAI.solveTask(problem);
      
      if (response.error) {
        throw new Error(response.error);
      }

      return {
        id: this.generateId(),
        solution: response.solution || 'Solution not available',
        confidence: response.confidence || 0.7,
        steps: response.steps || [],
        reasoning: response.reasoning || 'AI-generated solution'
      };
    } catch (error) {
      console.error('Task solving failed:', error);
      return {
        id: this.generateId(),
        solution: 'Unable to solve the problem at this time',
        confidence: 0.1,
        steps: ['Try breaking down the problem into smaller parts'],
        reasoning: 'Error in AI processing'
      };
    }
  }

  async createAdvertisement(productDescription: string): Promise<AdCreationResult> {
    try {
      const response = await pollenAI.createAdvertisement(productDescription);
      
      if (response.error) {
        throw new Error(response.error);
      }

      return {
        id: this.generateId(),
        title: response.title || 'Generated Advertisement',
        content: response.content || 'Advertisement content not available',
        targetAudience: response.targetAudience || 'General audience',
        estimatedCTR: response.estimatedCTR || 2.5,
        platform: response.platform || 'Multi-platform'
      };
    } catch (error) {
      console.error('Ad creation failed:', error);
      return {
        id: this.generateId(),
        title: 'Advertisement',
        content: `Discover our amazing product: ${productDescription}`,
        targetAudience: 'General audience',
        estimatedCTR: 2.0,
        platform: 'Multi-platform'
      };
    }
  }

  async generateMusic(prompt: string): Promise<MusicGenerationResult> {
    try {
      const response = await pollenAI.generateMusic(prompt);
      
      if (response.error) {
        throw new Error(response.error);
      }

      return {
        id: this.generateId(),
        title: response.title || 'Generated Music',
        genre: response.genre || 'Electronic',
        duration: response.duration || '3:30',
        audioUrl: response.audioUrl,
        description: response.description || `Music generated from: ${prompt}`
      };
    } catch (error) {
      console.error('Music generation failed:', error);
      return {
        id: this.generateId(),
        title: 'AI Generated Music',
        genre: 'Electronic',
        duration: '3:30',
        description: `Music concept: ${prompt}`
      };
    }
  }

  async generateImage(description: string): Promise<ImageGenerationResult> {
    try {
      const response = await pollenAI.generateImage(description);
      
      if (response.error) {
        throw new Error(response.error);
      }

      return {
        id: this.generateId(),
        description: response.description || description,
        imageUrl: response.imageUrl,
        style: response.style || 'AI Generated',
        dimensions: response.dimensions || '1024x1024'
      };
    } catch (error) {
      console.error('Image generation failed:', error);
      return {
        id: this.generateId(),
        description,
        style: 'AI Generated',
        dimensions: '1024x1024'
      };
    }
  }

  async automateTask(taskType: string, schedule: string): Promise<TaskAutomationResult> {
    try {
      const response = await pollenAI.automateTask(taskType, schedule);
      
      if (response.error) {
        throw new Error(response.error);
      }

      return {
        id: this.generateId(),
        taskType: response.taskType || taskType,
        schedule: response.schedule || schedule,
        status: response.status || 'scheduled',
        automationScript: response.script || 'Automation script generated'
      };
    } catch (error) {
      console.error('Task automation failed:', error);
      return {
        id: this.generateId(),
        taskType,
        schedule,
        status: 'failed',
        automationScript: 'Automation setup failed'
      };
    }
  }

  async curateSocialPost(topic: string): Promise<SocialPostResult> {
    try {
      const response = await pollenAI.curateSocialPost(topic);
      
      if (response.error) {
        throw new Error(response.error);
      }

      return {
        id: this.generateId(),
        platform: response.platform || 'Multi-platform',
        content: response.content || `Interesting content about ${topic}`,
        hashtags: response.hashtags || [`#${topic.replace(/\s+/g, '')}`],
        optimalPostTime: response.optimalPostTime || '12:00 PM',
        engagementScore: response.engagementScore || 7.5
      };
    } catch (error) {
      console.error('Social post curation failed:', error);
      return {
        id: this.generateId(),
        platform: 'Multi-platform',
        content: `Check out this interesting topic: ${topic}`,
        hashtags: [`#${topic.replace(/\s+/g, '')}`],
        optimalPostTime: '12:00 PM',
        engagementScore: 6.0
      };
    }
  }

  async analyzeTrends(topic: string): Promise<TrendAnalysisResult> {
    try {
      const response = await pollenAI.analyzeTrends(topic);
      
      if (response.error) {
        throw new Error(response.error);
      }

      return {
        id: this.generateId(),
        topic: response.topic || topic,
        trendScore: response.trendScore || 0.75,
        insights: response.insights || [`${topic} is showing interesting patterns`],
        predictions: response.predictions || [`${topic} may continue to grow`],
        timeframe: response.timeframe || '30 days'
      };
    } catch (error) {
      console.error('Trend analysis failed:', error);
      return {
        id: this.generateId(),
        topic,
        trendScore: 0.5,
        insights: [`Analysis of ${topic} is currently unavailable`],
        predictions: ['Trend data will be updated soon'],
        timeframe: '30 days'
      };
    }
  }
}

export const pollenAdaptiveService = new PollenAdaptiveService();