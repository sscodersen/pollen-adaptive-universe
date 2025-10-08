import { pollenAI } from './pollenAI';

export interface WellnessTip {
  id: string;
  title: string;
  content: string;
  category: 'physical' | 'mental' | 'nutrition' | 'productivity' | 'sleep';
  duration: string;
  difficulty: 'easy' | 'medium' | 'advanced';
  impact: number;
  tags: string[];
}

export interface LongFormContent {
  id: string;
  title: string;
  content: string;
  type: 'article' | 'video' | 'interactive';
  readTime: string;
  category: string;
  sections: { title: string; content: string }[];
  keyTakeaways: string[];
}

class WellnessContentService {
  private quickTips: WellnessTip[] = [
    {
      id: 'tip-1',
      title: '2-Minute Desk Stretch',
      content: 'Stand up, reach for the sky, and do 3 shoulder rolls backward. Your body will thank you!',
      category: 'physical',
      duration: '2 min',
      difficulty: 'easy',
      impact: 7,
      tags: ['desk-work', 'stretching', 'quick-break']
    },
    {
      id: 'tip-2',
      title: 'Mindful Breathing',
      content: 'Take 5 deep breaths: inhale for 4 counts, hold for 4, exhale for 4. Instant calm.',
      category: 'mental',
      duration: '1 min',
      difficulty: 'easy',
      impact: 8,
      tags: ['stress-relief', 'breathing', 'mindfulness']
    },
    {
      id: 'tip-3',
      title: 'Hydration Check',
      content: 'Drink a glass of water right now. Aim for 8 glasses daily for optimal brain function.',
      category: 'nutrition',
      duration: '30 sec',
      difficulty: 'easy',
      impact: 9,
      tags: ['hydration', 'health', 'energy']
    },
    {
      id: 'tip-4',
      title: 'Pomodoro Power',
      content: 'Work for 25 minutes, then take a 5-minute break. Repeat 4 times, then take a longer break.',
      category: 'productivity',
      duration: '25 min',
      difficulty: 'medium',
      impact: 9,
      tags: ['focus', 'time-management', 'productivity']
    },
    {
      id: 'tip-5',
      title: 'Screen Break',
      content: 'Every 20 minutes, look at something 20 feet away for 20 seconds. Protect your eyes!',
      category: 'physical',
      duration: '20 sec',
      difficulty: 'easy',
      impact: 8,
      tags: ['eye-health', 'screen-time', 'prevention']
    }
  ];

  async getDailyTip(): Promise<WellnessTip> {
    const index = new Date().getDate() % this.quickTips.length;
    return this.quickTips[index];
  }

  async getRandomTips(count: number = 3): Promise<WellnessTip[]> {
    const shuffled = [...this.quickTips].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, count);
  }

  async getTipsByCategory(category: WellnessTip['category']): Promise<WellnessTip[]> {
    return this.quickTips.filter(tip => tip.category === category);
  }

  async generateLongFormContent(topic: string): Promise<LongFormContent> {
    try {
      const response = await pollenAI.generate(
        `Create comprehensive wellness content about: ${topic}`,
        'wellness',
        { contentType: 'longform' }
      );

      return {
        id: `wellness-${Date.now()}`,
        title: `Complete Guide: ${topic}`,
        content: response.content,
        type: 'article',
        readTime: '8 min read',
        category: 'wellness',
        sections: this.extractSections(response.content),
        keyTakeaways: this.extractTakeaways(response.content)
      };
    } catch (error) {
      return this.createFallbackContent(topic);
    }
  }

  private extractSections(content: string): { title: string; content: string }[] {
    return [
      { title: 'Introduction', content: 'Understanding the fundamentals of wellness and its impact on daily life.' },
      { title: 'Key Principles', content: 'Core concepts that form the foundation of a healthy lifestyle.' },
      { title: 'Practical Steps', content: 'Actionable advice you can implement starting today.' },
      { title: 'Common Challenges', content: 'Addressing obstacles and how to overcome them.' }
    ];
  }

  private extractTakeaways(content: string): string[] {
    return [
      'Small, consistent actions create lasting change',
      'Listen to your body and adjust accordingly',
      'Wellness is a journey, not a destination',
      'Balance is key to sustainable health'
    ];
  }

  private createFallbackContent(topic: string): LongFormContent {
    return {
      id: `wellness-fallback-${Date.now()}`,
      title: `Wellness Guide: ${topic}`,
      content: 'Explore comprehensive wellness strategies tailored to your needs.',
      type: 'article',
      readTime: '5 min read',
      category: 'wellness',
      sections: this.extractSections(''),
      keyTakeaways: this.extractTakeaways('')
    };
  }
}

export const wellnessContentService = new WellnessContentService();
