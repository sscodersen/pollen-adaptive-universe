import { significanceAlgorithm } from './significanceAlgorithm';

export interface ScrapedContent {
  id: string;
  title: string;
  url: string;
  description: string;
  content: string;
  source: string;
  timestamp: number;
  category: 'news' | 'shop' | 'entertainment';
  significance: number;
  metadata: {
    author?: string;
    publishDate?: string;
    tags?: string[];
    price?: number;
    rating?: number;
    image?: string;
    brand?: string;
    retailer?: string;
    inStock?: boolean;
    discount?: number;
  };
}

interface BaseTemplate {
  title: string;
  description: string;
  content: string;
  author: string;
  tags: string[];
  path: string;
}

interface NewsTemplate extends BaseTemplate {
  category: 'news';
}

interface ShopTemplate extends BaseTemplate {
  category: 'shop';
  url?: string;
  price?: number;
  rating?: number;
  brand?: string;
  image?: string;
}

interface EntertainmentTemplate extends BaseTemplate {
  category: 'entertainment';
}

type ContentTemplate = NewsTemplate | ShopTemplate | EntertainmentTemplate;

class WebScrapingService {
  private cache: Map<string, ScrapedContent[]> = new Map();
  private lastScrape: Map<string, number> = new Map();
  private scrapeInterval = 180000; // Reduced to 3 minutes for more frequent updates
  private contentTemplates = new Map<string, any[]>();

  constructor() {
    this.initializeTemplates();
  }

  private initializeTemplates() {
    // Cache templates for better performance
    this.contentTemplates.set('news', this.getNewsTemplates());
    this.contentTemplates.set('shop', this.getShopTemplates());
    this.contentTemplates.set('entertainment', this.getEntertainmentTemplates());
  }

  async scrapeContent(category: 'news' | 'shop' | 'entertainment', limit: number = 20): Promise<ScrapedContent[]> {
    const cacheKey = `${category}-content`;
    const lastScrape = this.lastScrape.get(cacheKey) || 0;
    const now = Date.now();

    // Return cached content if recent
    if (now - lastScrape < this.scrapeInterval && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      return this.shuffleArray([...cached]).slice(0, limit);
    }

    try {
      const scrapedData = await this.performScraping(category, Math.max(limit * 2, 30));
      
      // Score content for significance
      const scoredContent = scrapedData.map(item => ({
        ...item,
        significance: significanceAlgorithm.scoreContent(item.content, category).significanceScore
      }));

      // Filter and sort by significance
      const highQualityContent = scoredContent
        .filter(item => item.significance > 6.5) // Slightly lower threshold for more content
        .sort((a, b) => b.significance - a.significance)
        .slice(0, Math.max(limit, 20));

      this.cache.set(cacheKey, highQualityContent);
      this.lastScrape.set(cacheKey, now);

      return this.shuffleArray([...highQualityContent]).slice(0, limit);
    } catch (error) {
      console.error(`Scraping failed for ${category}:`, error);
      return this.cache.get(cacheKey)?.slice(0, limit) || [];
    }
  }

  private shuffleArray<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  private async performScraping(category: 'news' | 'shop' | 'entertainment', count: number): Promise<ScrapedContent[]> {
    const sources = this.getSourcesForCategory(category);
    const templates = this.contentTemplates.get(category) || [];
    const content: ScrapedContent[] = [];

    for (let i = 0; i < count; i++) {
      const source = sources[Math.floor(Math.random() * sources.length)];
      const template = templates[Math.floor(Math.random() * templates.length)];
      const item = this.generateScrapedItem(category, source, template, i);
      content.push(item);
    }

    return content;
  }

  private generateScrapedItem(category: 'news' | 'shop' | 'entertainment', source: any, template: any, index: number): ScrapedContent {
    const variance = Math.random() * 0.3 - 0.15; // Add some variance to timestamps
    const baseTimestamp = Date.now() - Math.random() * 86400000;

    const baseItem: ScrapedContent = {
      id: `${category}-${Date.now()}-${index}-${Math.random().toString(36).substring(7)}`,
      title: this.addVarianceToTitle(template.title),
      url: category === 'shop' && template.url ? template.url : `https://${source.domain}/${template.path}/${Math.random().toString(36).substring(7)}`,
      description: template.description,
      content: template.content,
      source: source.name,
      timestamp: baseTimestamp + (variance * 3600000), // Add variance to spread timestamps
      category,
      significance: 0,
      metadata: {
        author: template.author,
        publishDate: new Date(baseTimestamp).toISOString(),
        tags: template.tags,
      }
    };

    if (category === 'shop') {
      baseItem.metadata = {
        ...baseItem.metadata,
        price: template.price || Math.floor(Math.random() * 1000) + 50,
        rating: template.rating || Math.round((Math.random() * 2 + 3) * 10) / 10,
        image: template.image || `https://picsum.photos/400/300?random=${index + Date.now()}`,
        brand: template.brand,
        retailer: source.name,
        inStock: Math.random() > 0.15, // Higher in-stock rate
        discount: Math.random() > 0.6 ? Math.floor(Math.random() * 30) + 10 : 0
      };
    }

    return baseItem;
  }

  private addVarianceToTitle(title: string): string {
    const prefixes = ['Latest:', 'Breaking:', 'New:', 'Updated:', 'Exclusive:', ''];
    const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
    return prefix ? `${prefix} ${title}` : title;
  }

  private getNewsTemplates() {
    return [
      {
        category: 'news' as const,
        title: 'Revolutionary AI Breakthrough Transforms Healthcare Industry',
        description: 'New machine learning model achieves 99% accuracy in early disease detection',
        content: 'Researchers at leading institutions have developed an AI system that can detect early-stage diseases with unprecedented accuracy. The breakthrough technology uses advanced neural networks to analyze medical imaging data, potentially saving millions of lives through early intervention.',
        author: 'Dr. Sarah Chen',
        tags: ['AI', 'Healthcare', 'Technology', 'Innovation'],
        path: 'article'
      },
      {
        category: 'news' as const,
        title: 'Quantum Computing Milestone Achieved by Tech Giants',
        description: 'Major advancement in quantum error correction brings practical quantum computing closer',
        content: 'A consortium of technology companies has achieved a significant milestone in quantum computing, demonstrating reliable quantum error correction at scale. This advancement brings us closer to practical quantum computers that could revolutionize cryptography, drug discovery, and optimization problems.',
        author: 'Michael Rodriguez',
        tags: ['Quantum Computing', 'Technology', 'Innovation', 'Science'],
        path: 'tech'
      },
      {
        category: 'news' as const,
        title: 'Large Language Models Show Emergent Reasoning Capabilities',
        description: 'New research reveals unexpected problem-solving abilities in AI systems',
        content: 'Recent studies demonstrate that large language models exhibit emergent reasoning capabilities not explicitly programmed. These findings suggest AI systems may develop complex cognitive abilities through scale and training.',
        author: 'Dr. Emily Watson',
        tags: ['AI', 'Research', 'Machine Learning', 'Cognition'],
        path: 'research'
      }
    ];
  }

  private getShopTemplates() {
    return [
      {
        category: 'shop' as const,
        title: 'NVIDIA RTX 4090 Ti - Professional AI Workstation GPU',
        description: 'Ultimate GPU for AI development, machine learning, and high-performance computing',
        content: 'The most powerful graphics card for AI researchers and developers. Features 24GB GDDR6X memory, 16,384 CUDA cores, and optimized AI acceleration.',
        author: 'NVIDIA',
        tags: ['GPU', 'AI Hardware', 'Workstation', 'NVIDIA'],
        path: 'product',
        url: 'https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/',
        price: 1599,
        rating: 4.8,
        brand: 'NVIDIA',
        image: 'https://images.unsplash.com/photo-1591488320449-011701bb6704?w=400&h=300&fit=crop'
      },
      {
        category: 'shop' as const,
        title: 'Apple MacBook Pro M3 Max - AI Development Machine',
        description: 'Unified memory architecture perfect for machine learning workflows',
        content: 'Latest MacBook Pro with M3 Max chip delivers exceptional performance for AI development. 128GB unified memory and optimized ML frameworks.',
        author: 'Apple',
        tags: ['Laptop', 'AI Development', 'Apple', 'M3'],
        path: 'product',
        url: 'https://www.apple.com/macbook-pro/',
        price: 3999,
        rating: 4.7,
        brand: 'Apple',
        image: 'https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400&h=300&fit=crop'
      }
    ];
  }

  private getEntertainmentTemplates() {
    return [
      {
        category: 'entertainment' as const,
        title: 'Neural Symphony: AI-Generated Interactive Music Experience',
        description: 'Revolutionary music platform that creates personalized symphonies in real-time',
        content: 'Experience music like never before with Neural Symphony, an AI-powered platform that generates unique musical compositions based on your emotions, activities, and preferences. Each listening session creates a completely original piece of music.',
        author: 'Harmonic AI Studios',
        tags: ['AI Music', 'Interactive', 'Generative', 'Audio'],
        path: 'experience'
      },
      {
        category: 'entertainment' as const,
        title: 'Quantum Dreamscape: Procedural Reality Simulation',
        description: 'Infinite virtual worlds generated by quantum-inspired algorithms',
        content: 'Explore limitless virtual environments in Quantum Dreamscape, where every world is procedurally generated using quantum-inspired algorithms. No two experiences are ever the same.',
        author: 'Virtual Worlds Studio',
        tags: ['Virtual Reality', 'Procedural', 'Gaming', 'Simulation'],
        path: 'game'
      }
    ];
  }

  private getSourcesForCategory(category: 'news' | 'shop' | 'entertainment') {
    const sources = {
      news: [
        { name: 'TechCrunch', domain: 'techcrunch.com' },
        { name: 'Wired', domain: 'wired.com' },
        { name: 'MIT Technology Review', domain: 'technologyreview.com' },
        { name: 'IEEE Spectrum', domain: 'spectrum.ieee.org' },
        { name: 'Nature', domain: 'nature.com' },
        { name: 'Science Daily', domain: 'sciencedaily.com' },
        { name: 'The Verge', domain: 'theverge.com' },
        { name: 'ArXiv', domain: 'arxiv.org' }
      ],
      shop: [
        { name: 'Amazon', domain: 'amazon.com' },
        { name: 'Newegg', domain: 'newegg.com' },
        { name: 'Best Buy', domain: 'bestbuy.com' },
        { name: 'B&H Photo', domain: 'bhphotovideo.com' },
        { name: 'Micro Center', domain: 'microcenter.com' },
        { name: 'Apple', domain: 'apple.com' },
        { name: 'NVIDIA', domain: 'nvidia.com' },
        { name: 'Corsair', domain: 'corsair.com' }
      ],
      entertainment: [
        { name: 'Steam', domain: 'steampowered.com' },
        { name: 'Epic Games', domain: 'epicgames.com' },
        { name: 'Spotify', domain: 'spotify.com' },
        { name: 'Netflix', domain: 'netflix.com' },
        { name: 'YouTube', domain: 'youtube.com' },
        { name: 'Twitch', domain: 'twitch.tv' },
        { name: 'Discord', domain: 'discord.com' },
        { name: 'Reddit', domain: 'reddit.com' }
      ]
    };

    return sources[category];
  }

  async getTrendingKeywords(category?: 'news' | 'shop' | 'entertainment'): Promise<string[]> {
    const trending = {
      news: [
        'artificial intelligence breakthroughs',
        'quantum computing advances',
        'machine learning research',
        'neural network optimization',
        'AI safety protocols',
        'autonomous systems development',
        'large language models',
        'computer vision improvements'
      ],
      shop: [
        'AI development hardware',
        'professional workstations',
        'machine learning GPUs',
        'high-capacity memory',
        'workstation processors',
        'AI training systems',
        'development tools',
        'cloud computing resources'
      ],
      entertainment: [
        'AI-generated content',
        'procedural entertainment',
        'interactive experiences',
        'virtual reality platforms',
        'generative gaming',
        'AI music creation',
        'adaptive storytelling',
        'personalized media'
      ]
    };

    if (category) {
      return trending[category];
    }

    return Object.values(trending).flat();
  }

  clearCache(): void {
    this.cache.clear();
    this.lastScrape.clear();
  }

  getStats() {
    return {
      cacheSize: this.cache.size,
      lastScrapes: Object.fromEntries(this.lastScrape),
      totalCachedItems: Array.from(this.cache.values()).reduce((total, items) => total + items.length, 0)
    };
  }
}

export const webScrapingService = new WebScrapingService();
