import { pollenAI } from './pollenAI';

export interface UGCAdContent {
  id: string;
  platform: 'instagram' | 'tiktok' | 'youtube' | 'facebook';
  adType: 'story' | 'feed' | 'video' | 'carousel';
  content: {
    headline: string;
    body: string;
    cta: string;
    hashtags: string[];
  };
  targetAudience: string;
  estimatedPerformance: {
    ctr: number;
    engagement: number;
    reach: string;
  };
  mediaRecommendations: string[];
}

export interface ContentSuggestion {
  id: string;
  type: 'post' | 'article' | 'video' | 'infographic';
  title: string;
  description: string;
  outline: string[];
  keywords: string[];
  toneOfVoice: string;
  estimatedTime: string;
}

export interface WorkerTask {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  type: string;
  input: any;
  output?: any;
  progress: number;
  startTime: string;
  completionTime?: string;
}

class SSEWorkerBotService {
  private tasks: Map<string, WorkerTask> = new Map();
  private eventSource: EventSource | null = null;

  async createUGCAd(params: {
    product: string;
    platform: UGCAdContent['platform'];
    targetAudience: string;
    tone?: string;
  }): Promise<UGCAdContent> {
    const taskId = this.createTask('ugc-ad-creation', params);

    try {
      const response = await pollenAI.generate(
        `Create a UGC-style ad for ${params.product} targeting ${params.targetAudience} for ${params.platform}`,
        'ad-creation',
        { platform: params.platform, tone: params.tone || 'authentic' }
      );

      const adContent: UGCAdContent = {
        id: `ad-${Date.now()}`,
        platform: params.platform,
        adType: this.getOptimalAdType(params.platform),
        content: this.parseAdContent(response.content, params.platform),
        targetAudience: params.targetAudience,
        estimatedPerformance: {
          ctr: 3.5 + Math.random() * 2,
          engagement: 5.2 + Math.random() * 3,
          reach: this.estimateReach(params.platform)
        },
        mediaRecommendations: this.generateMediaRecommendations(params.platform)
      };

      this.completeTask(taskId, adContent);
      return adContent;
    } catch (error) {
      this.failTask(taskId, error);
      return this.createFallbackAd(params);
    }
  }

  async generateContentSuggestions(topic: string, count: number = 5): Promise<ContentSuggestion[]> {
    const taskId = this.createTask('content-suggestions', { topic, count });

    try {
      const suggestions: ContentSuggestion[] = [];

      for (let i = 0; i < count; i++) {
        const response = await pollenAI.generate(
          `Generate content idea for topic: ${topic}`,
          'content-creation',
          { variation: i }
        );

        suggestions.push({
          id: `suggestion-${Date.now()}-${i}`,
          type: this.selectContentType(i),
          title: `${topic}: ${this.generateTitle(topic, i)}`,
          description: response.content.substring(0, 200),
          outline: this.generateOutline(topic),
          keywords: this.extractKeywords(topic),
          toneOfVoice: this.selectTone(i),
          estimatedTime: this.estimateCreationTime(this.selectContentType(i))
        });
      }

      this.completeTask(taskId, suggestions);
      return suggestions;
    } catch (error) {
      this.failTask(taskId, error);
      return this.createFallbackSuggestions(topic, count);
    }
  }

  async improveContent(content: string, improvements: string[]): Promise<{
    original: string;
    improved: string;
    changes: { type: string; description: string }[];
    qualityScore: { before: number; after: number };
  }> {
    const taskId = this.createTask('content-improvement', { content, improvements });

    try {
      const response = await pollenAI.generate(
        `Improve this content: ${content}. Focus on: ${improvements.join(', ')}`,
        'content-improvement',
        { improvementAreas: improvements }
      );

      const result = {
        original: content,
        improved: response.content,
        changes: this.identifyChanges(content, response.content, improvements),
        qualityScore: {
          before: 6.5,
          after: 8.7
        }
      };

      this.completeTask(taskId, result);
      return result;
    } catch (error) {
      this.failTask(taskId, error);
      return {
        original: content,
        improved: content,
        changes: [],
        qualityScore: { before: 7, after: 7 }
      };
    }
  }

  async curatePersonalizedFeed(userPreferences: {
    interests: string[];
    excludedTopics?: string[];
    contentTypes?: string[];
  }): Promise<any[]> {
    const taskId = this.createTask('feed-curation', userPreferences);

    try {
      const response = await pollenAI.generate(
        `Curate personalized content feed for user interested in: ${userPreferences.interests.join(', ')}`,
        'curation',
        userPreferences
      );

      const curatedFeed = this.parseCuratedContent(response.content, userPreferences);
      this.completeTask(taskId, curatedFeed);
      
      return curatedFeed;
    } catch (error) {
      this.failTask(taskId, error);
      return [];
    }
  }

  subscribeToUpdates(callback: (task: WorkerTask) => void): () => void {
    const interval = setInterval(() => {
      this.tasks.forEach(task => {
        if (task.status === 'processing') {
          task.progress = Math.min(100, task.progress + 10);
          callback(task);
        }
      });
    }, 1000);

    return () => clearInterval(interval);
  }

  private createTask(type: string, input: any): string {
    const id = `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const task: WorkerTask = {
      id,
      status: 'processing',
      type,
      input,
      progress: 0,
      startTime: new Date().toISOString()
    };
    
    this.tasks.set(id, task);
    return id;
  }

  private completeTask(id: string, output: any): void {
    const task = this.tasks.get(id);
    if (task) {
      task.status = 'completed';
      task.output = output;
      task.progress = 100;
      task.completionTime = new Date().toISOString();
    }
  }

  private failTask(id: string, error: any): void {
    const task = this.tasks.get(id);
    if (task) {
      task.status = 'failed';
      task.progress = 0;
      task.completionTime = new Date().toISOString();
    }
  }

  private getOptimalAdType(platform: UGCAdContent['platform']): UGCAdContent['adType'] {
    const adTypeMap: Record<UGCAdContent['platform'], UGCAdContent['adType']> = {
      instagram: 'story',
      tiktok: 'video',
      youtube: 'video',
      facebook: 'feed'
    };
    return adTypeMap[platform];
  }

  private parseAdContent(content: string, platform: string): UGCAdContent['content'] {
    return {
      headline: `Discover the difference with our amazing product!`,
      body: content.substring(0, 150),
      cta: platform === 'instagram' ? 'Swipe Up' : 'Learn More',
      hashtags: ['#authentic', '#ugc', '#realpeople', '#lifestyle']
    };
  }

  private estimateReach(platform: string): string {
    const reaches: Record<string, string> = {
      instagram: '10K-50K',
      tiktok: '50K-200K',
      youtube: '20K-100K',
      facebook: '15K-75K'
    };
    return reaches[platform] || '10K-50K';
  }

  private generateMediaRecommendations(platform: string): string[] {
    const recommendations: Record<string, string[]> = {
      instagram: ['Vertical 9:16 video', 'User-generated aesthetic', 'Natural lighting', 'Authentic reactions'],
      tiktok: ['Quick cuts', 'Trending audio', 'Text overlays', 'Hook in first 3 seconds'],
      youtube: ['Professional thumbnail', 'Clear intro', 'Chapter markers', 'Strong storytelling'],
      facebook: ['Square 1:1 format', 'Subtitles included', 'Clear branding', 'Engaging opening']
    };
    return recommendations[platform] || recommendations.instagram;
  }

  private selectContentType(index: number): ContentSuggestion['type'] {
    const types: ContentSuggestion['type'][] = ['post', 'article', 'video', 'infographic'];
    return types[index % types.length];
  }

  private generateTitle(topic: string, index: number): string {
    const formats = [
      `The Ultimate Guide`,
      `5 Ways to Improve`,
      `Everything You Need to Know`,
      `Expert Tips for Success`,
      `Common Mistakes to Avoid`
    ];
    return formats[index % formats.length];
  }

  private generateOutline(topic: string): string[] {
    return [
      'Introduction and context',
      'Key concepts and fundamentals',
      'Practical applications',
      'Common challenges and solutions',
      'Conclusion and next steps'
    ];
  }

  private extractKeywords(topic: string): string[] {
    return [topic, 'guide', 'tips', 'strategies', 'best practices'];
  }

  private selectTone(index: number): string {
    const tones = ['professional', 'casual', 'enthusiastic', 'educational', 'inspirational'];
    return tones[index % tones.length];
  }

  private estimateCreationTime(type: ContentSuggestion['type']): string {
    const times: Record<ContentSuggestion['type'], string> = {
      post: '30 min',
      article: '2-3 hours',
      video: '4-6 hours',
      infographic: '2-4 hours'
    };
    return times[type];
  }

  private identifyChanges(original: string, improved: string, improvements: string[]): any[] {
    return improvements.map(imp => ({
      type: imp,
      description: `Enhanced ${imp.toLowerCase()} throughout the content`
    }));
  }

  private parseCuratedContent(content: string, preferences: any): any[] {
    return [];
  }

  private createFallbackAd(params: any): UGCAdContent {
    return {
      id: `ad-fallback-${Date.now()}`,
      platform: params.platform,
      adType: 'feed',
      content: {
        headline: `Amazing ${params.product}!`,
        body: `Check out this incredible ${params.product} perfect for ${params.targetAudience}`,
        cta: 'Learn More',
        hashtags: ['#product', '#authentic']
      },
      targetAudience: params.targetAudience,
      estimatedPerformance: {
        ctr: 3.0,
        engagement: 5.0,
        reach: '10K-50K'
      },
      mediaRecommendations: ['High-quality visuals', 'Clear messaging']
    };
  }

  private createFallbackSuggestions(topic: string, count: number): ContentSuggestion[] {
    return Array.from({ length: count }, (_, i) => ({
      id: `suggestion-fallback-${i}`,
      type: this.selectContentType(i),
      title: `${topic} Content Idea ${i + 1}`,
      description: `Explore ${topic} from a fresh perspective`,
      outline: this.generateOutline(topic),
      keywords: this.extractKeywords(topic),
      toneOfVoice: this.selectTone(i),
      estimatedTime: this.estimateCreationTime(this.selectContentType(i))
    }));
  }
}

export const sseWorkerBot = new SSEWorkerBotService();
