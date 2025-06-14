
import { significanceAlgorithm, type ScoredContent } from './significanceAlgorithm';
import { pollenAI } from './pollenAI';

export interface WebContent {
  id: string;
  title: string;
  url: string;
  description: string;
  content: string;
  source: string;
  timestamp: number;
  category: 'news' | 'shop' | 'entertainment';
  significance: number;
  price?: number;
  rating?: number;
  image?: string;
}

class ContentCuratorService {
  private cache: Map<string, WebContent[]> = new Map();
  private lastUpdate: Map<string, number> = new Map();
  private updateInterval = 300000; // 5 minutes

  async scrapeAndCurateContent(category: 'news' | 'shop' | 'entertainment', limit: number = 20): Promise<WebContent[]> {
    const cacheKey = category;
    const lastUpdate = this.lastUpdate.get(cacheKey) || 0;
    const now = Date.now();

    // Return cached content if still fresh
    if (now - lastUpdate < this.updateInterval && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!.slice(0, limit);
    }

    try {
      // Simulate web scraping with AI-generated content that mimics real sources
      const scrapedContent = await this.simulateWebScraping(category, limit * 2);
      
      // Score content using significance algorithm
      const scoredContent = scrapedContent.map(content => {
        const scored = significanceAlgorithm.scoreContent(content.content, category);
        return {
          ...content,
          significance: scored.significanceScore
        };
      });

      // Filter for high significance (>7) and sort
      const highSignificanceContent = scoredContent
        .filter(content => content.significance > 7.0)
        .sort((a, b) => b.significance - a.significance)
        .slice(0, limit);

      // Cache results
      this.cache.set(cacheKey, highSignificanceContent);
      this.lastUpdate.set(cacheKey, now);

      return highSignificanceContent;
    } catch (error) {
      console.error(`Error curating ${category} content:`, error);
      return this.cache.get(cacheKey) || [];
    }
  }

  private async simulateWebScraping(category: 'news' | 'shop' | 'entertainment', count: number): Promise<WebContent[]> {
    const content: WebContent[] = [];
    const sources = this.getSourcesForCategory(category);

    for (let i = 0; i < count; i++) {
      const source = sources[Math.floor(Math.random() * sources.length)];
      
      // Generate content using Pollen AI
      const response = await pollenAI.generate(
        `Generate high-impact ${category} content for ${source.name}`,
        category,
        false // Don't filter yet, we'll do it after
      );

      const item: WebContent = {
        id: `${category}-${Date.now()}-${i}`,
        title: this.generateTitleForCategory(category),
        url: this.generateUrlForSource(source),
        description: response.content.slice(0, 200) + '...',
        content: response.content,
        source: source.name,
        timestamp: Date.now() - Math.random() * 86400000, // Within last 24 hours
        category,
        significance: 0, // Will be calculated later
        ...(category === 'shop' && {
          price: Math.floor(Math.random() * 500) + 10,
          rating: Math.random() * 2 + 3,
          image: `https://picsum.photos/400/300?random=${i}`
        })
      };

      content.push(item);
    }

    return content;
  }

  private getSourcesForCategory(category: 'news' | 'shop' | 'entertainment') {
    const sources = {
      news: [
        { name: 'Reuters', domain: 'reuters.com' },
        { name: 'Associated Press', domain: 'apnews.com' },
        { name: 'BBC News', domain: 'bbc.com' },
        { name: 'MIT Technology Review', domain: 'technologyreview.com' },
        { name: 'Nature', domain: 'nature.com' },
        { name: 'Science', domain: 'science.org' }
      ],
      shop: [
        { name: 'Amazon', domain: 'amazon.com' },
        { name: 'Best Buy', domain: 'bestbuy.com' },
        { name: 'Newegg', domain: 'newegg.com' },
        { name: 'B&H Photo', domain: 'bhphotovideo.com' },
        { name: 'Walmart', domain: 'walmart.com' },
        { name: 'Target', domain: 'target.com' }
      ],
      entertainment: [
        { name: 'Steam', domain: 'steampowered.com' },
        { name: 'Netflix', domain: 'netflix.com' },
        { name: 'Spotify', domain: 'spotify.com' },
        { name: 'YouTube', domain: 'youtube.com' },
        { name: 'Twitch', domain: 'twitch.tv' },
        { name: 'Epic Games', domain: 'epicgames.com' }
      ]
    };

    return sources[category];
  }

  private generateTitleForCategory(category: 'news' | 'shop' | 'entertainment'): string {
    const titles = {
      news: [
        'Revolutionary AI Breakthrough Changes Industry Standards',
        'Global Climate Initiative Shows Unprecedented Results',
        'Quantum Computing Milestone Achieved by Research Team',
        'Medical Innovation Offers Hope for Millions',
        'Space Exploration Reveals Groundbreaking Discovery',
        'Economic Policy Shift Creates New Opportunities'
      ],
      shop: [
        'Professional-Grade AI Development Kit - 40% Off',
        'Premium Productivity Software Bundle - Lifetime License',
        'High-Performance Computing Workstation - Verified Seller',
        'Advanced Analytics Platform - Enterprise Edition',
        'Creative Design Suite - Professional Tools',
        'Automation Hardware Kit - Complete Solution'
      ],
      entertainment: [
        'Interactive AI-Powered Adventure Game',
        'Immersive Virtual Reality Experience',
        'Creative Coding Challenge Platform',
        'Multiplayer Strategy Simulation',
        'Educational Gaming Experience',
        'Interactive Storytelling Platform'
      ]
    };

    const categoryTitles = titles[category];
    return categoryTitles[Math.floor(Math.random() * categoryTitles.length)];
  }

  private generateUrlForSource(source: { name: string; domain: string }): string {
    const paths = [
      '/article', '/story', '/news', '/product', '/item', '/content',
      '/feature', '/report', '/analysis', '/review', '/guide'
    ];
    const path = paths[Math.floor(Math.random() * paths.length)];
    const id = Math.random().toString(36).substring(2, 15);
    return `https://${source.domain}${path}/${id}`;
  }

  async getTrendingTopics(category?: 'news' | 'shop' | 'entertainment'): Promise<string[]> {
    // Combine significance algorithm trending topics with category-specific trends
    const baseTopics = significanceAlgorithm.getTrendingTopics();
    
    const categoryTopics = {
      news: [
        'artificial intelligence regulation',
        'renewable energy breakthroughs',
        'quantum computing advances',
        'biotechnology innovations',
        'space exploration missions'
      ],
      shop: [
        'AI-powered productivity tools',
        'sustainable tech products',
        'home automation systems',
        'professional development software',
        'creative design hardware'
      ],
      entertainment: [
        'interactive AI experiences',
        'virtual reality platforms',
        'educational gaming',
        'creative coding tools',
        'collaborative storytelling'
      ]
    };

    if (category) {
      return [...baseTopics.slice(0, 5), ...categoryTopics[category]];
    }

    return baseTopics;
  }

  clearCache(): void {
    this.cache.clear();
    this.lastUpdate.clear();
  }
}

export const contentCurator = new ContentCuratorService();
