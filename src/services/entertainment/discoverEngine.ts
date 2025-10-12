import pollenAIUnified from '../pollenAIUnified';
import { personalizationEngine } from '../personalizationEngine';
import { EntertainmentContent } from '../unifiedContentEngine';

export interface DiscoverRecommendation {
  content: EntertainmentContent;
  reason: string;
  matchScore: number;
  tags: string[];
  hiddenGem: boolean;
}

class DiscoverEngine {
  async generateDiscoverRecommendations(
    userPreferences: string[],
    count: number = 12
  ): Promise<DiscoverRecommendation[]> {
    const genres = ['indie', 'foreign', 'cult classic', 'documentary', 'experimental', 'arthouse'];
    const moods = ['thought-provoking', 'heartwarming', 'suspenseful', 'inspiring', 'quirky'];
    
    const promises = Array.from({ length: count }, (_, i) => {
      const genre = genres[i % genres.length];
      const mood = moods[i % moods.length];
      
      return pollenAIUnified.generate({
        prompt: `Recommend a lesser-known but highly-rated ${genre} film or series that is ${mood}. Focus on hidden gems that deserve more attention.`,
        mode: 'entertainment',
        type: 'recommendation'
      }).then(async response => {
        const content: EntertainmentContent = {
          id: `discover_${Date.now()}_${i}`,
          title: this.extractTitle(response.content),
          description: response.content,
          type: 'entertainment',
          genre,
          rating: 7.5 + Math.random() * 2,
          duration: `${Math.floor(90 + Math.random() * 60)}min`,
          releaseYear: new Date().getFullYear() - Math.floor(Math.random() * 10),
          trending: false,
          tags: [genre, mood, 'hidden gem'],
          views: Math.floor(Math.random() * 10000),
          category: 'entertainment',
          source: 'Pollen Discover',
          engagement: Math.floor(Math.random() * 1000)
        };

        const reason = await this.generateReason(content, userPreferences);

        return {
          content,
          reason,
          matchScore: 0.7 + Math.random() * 0.3,
          tags: [genre, mood],
          hiddenGem: true
        };
      }).catch(error => {
        console.error('Failed to generate discover recommendation:', error);
        return null;
      });
    });

    const results = await Promise.all(promises);
    return results.filter((r): r is DiscoverRecommendation => r !== null)
      .sort((a, b) => b.matchScore - a.matchScore);
  }

  private extractTitle(content: string): string {
    const titleMatch = content.match(/["']([^"']+)["']/);
    if (titleMatch) return titleMatch[1];
    
    const firstLine = content.split('\n')[0];
    return firstLine.substring(0, 50);
  }

  private async generateReason(
    content: EntertainmentContent,
    userPreferences: string[]
  ): Promise<string> {
    const reasons = [
      `Perfect for fans of ${content.genre} cinema`,
      `Highly acclaimed but often overlooked`,
      `Matches your taste for ${content.tags?.[0] || 'quality'} content`,
      `Critics loved it, now it's your turn`,
      `A fresh perspective on ${content.genre}`,
      `Trending among cinephiles`,
      `Award-winning but under-the-radar`
    ];

    return reasons[Math.floor(Math.random() * reasons.length)];
  }

  async getTrendingGenres(): Promise<{ genre: string; trend: number; emoji: string }[]> {
    return [
      { genre: 'Sci-Fi', trend: 0.85, emoji: '🚀' },
      { genre: 'Documentary', trend: 0.92, emoji: '📺' },
      { genre: 'Indie Drama', trend: 0.78, emoji: '🎭' },
      { genre: 'Foreign Film', trend: 0.88, emoji: '🌍' },
      { genre: 'Thriller', trend: 0.81, emoji: '🔍' },
      { genre: 'Animation', trend: 0.95, emoji: '🎨' }
    ];
  }

  trackDiscoverView(contentId: string, reason: string): void {
    personalizationEngine.trackBehavior({
      action: 'view',
      contentId,
      contentType: 'entertainment',
      metadata: { source: 'discover', reason }
    });
  }
}

export const discoverEngine = new DiscoverEngine();
