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
  private scrapeInterval = 300000; // 5 minutes

  async scrapeContent(category: 'news' | 'shop' | 'entertainment', limit: number = 20): Promise<ScrapedContent[]> {
    const cacheKey = `${category}-content`;
    const lastScrape = this.lastScrape.get(cacheKey) || 0;
    const now = Date.now();

    // Return cached content if recent
    if (now - lastScrape < this.scrapeInterval && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!.slice(0, limit);
    }

    try {
      const scrapedData = await this.performScraping(category, limit * 2);
      
      // Score content for significance
      const scoredContent = scrapedData.map(item => ({
        ...item,
        significance: significanceAlgorithm.scoreContent(item.content, category).significanceScore
      }));

      // Filter and sort by significance
      const highQualityContent = scoredContent
        .filter(item => item.significance > 7.0)
        .sort((a, b) => b.significance - a.significance)
        .slice(0, limit);

      this.cache.set(cacheKey, highQualityContent);
      this.lastScrape.set(cacheKey, now);

      return highQualityContent;
    } catch (error) {
      console.error(`Scraping failed for ${category}:`, error);
      return this.cache.get(cacheKey) || [];
    }
  }

  private async performScraping(category: 'news' | 'shop' | 'entertainment', count: number): Promise<ScrapedContent[]> {
    const sources = this.getSourcesForCategory(category);
    const content: ScrapedContent[] = [];

    for (let i = 0; i < count; i++) {
      const source = sources[Math.floor(Math.random() * sources.length)];
      const item = await this.generateScrapedItem(category, source, i);
      content.push(item);
    }

    return content;
  }

  private async generateScrapedItem(category: 'news' | 'shop' | 'entertainment', source: any, index: number): Promise<ScrapedContent> {
    const templates = this.getContentTemplates(category);
    const template = templates[Math.floor(Math.random() * templates.length)];

    const baseItem = {
      id: `${category}-${Date.now()}-${index}`,
      title: template.title,
      url: category === 'shop' && 'url' in template ? template.url : `https://${source.domain}/${template.path}/${Math.random().toString(36).substring(7)}`,
      description: template.description,
      content: template.content,
      source: source.name,
      timestamp: Date.now() - Math.random() * 86400000,
      category,
      significance: 0,
      metadata: {
        author: template.author,
        publishDate: new Date(Date.now() - Math.random() * 86400000).toISOString(),
        tags: template.tags,
      }
    };

    if (category === 'shop') {
      const shopTemplate = template as ShopTemplate;
      baseItem.metadata = {
        ...baseItem.metadata,
        price: shopTemplate.price || Math.floor(Math.random() * 1000) + 50,
        rating: shopTemplate.rating || Math.random() * 2 + 3,
        image: shopTemplate.image || `https://picsum.photos/400/300?random=${index}`,
        brand: shopTemplate.brand,
        retailer: source.name,
        inStock: Math.random() > 0.1,
        discount: Math.random() > 0.7 ? Math.floor(Math.random() * 30) + 10 : 0
      };
    }

    return baseItem;
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

  private getContentTemplates(category: 'news' | 'shop' | 'entertainment'): ContentTemplate[] {
    const templates = {
      news: [
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
      ] as NewsTemplate[],
      shop: [
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
        },
        {
          category: 'shop' as const,
          title: 'Corsair Dominator Platinum RGB 128GB DDR5-5600',
          description: 'High-capacity memory kit for demanding AI workloads',
          content: 'Professional-grade memory designed for AI training and inference. Ultra-fast DDR5-5600 speeds with RGB lighting.',
          author: 'Corsair',
          tags: ['Memory', 'DDR5', 'AI Hardware', 'RGB'],
          path: 'product',
          url: 'https://www.corsair.com/us/en/Categories/Products/Memory/DOMINATOR-PLATINUM-RGB-DDR5-Memory/p/CMT128GX5M4B5600C40',
          price: 899,
          rating: 4.6,
          brand: 'Corsair',
          image: 'https://images.unsplash.com/photo-1555617981-dac3880eac6e?w=400&h=300&fit=crop'
        },
        {
          category: 'shop' as const,
          title: 'Intel Xeon W9-3495X - AI Training Processor',
          description: '56-core workstation processor for parallel AI computations',
          content: 'Flagship workstation processor with 56 cores and 112 threads. Optimized for AI training, scientific computing, and parallel workloads.',
          author: 'Intel',
          tags: ['CPU', 'Workstation', 'AI Training', 'Intel'],
          path: 'product',
          url: 'https://www.intel.com/content/www/us/en/products/processors/xeon/w-processors.html',
          price: 5889,
          rating: 4.5,
          brand: 'Intel',
          image: 'https://images.unsplash.com/photo-1555617981-dac3880eac6e?w=400&h=300&fit=crop'
        }
      ] as ShopTemplate[],
      entertainment: [
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
        },
        {
          category: 'entertainment' as const,
          title: 'CodeCraft: AI Programming Companion Game',
          description: 'Learn programming through gamified AI-assisted challenges',
          content: 'Transform coding education with CodeCraft, where AI mentors guide you through programming challenges in a fantasy RPG setting. Level up your skills while building real applications.',
          author: 'EduTech Games',
          tags: ['Education', 'Programming', 'Gaming', 'AI'],
          path: 'game'
        }
      ] as EntertainmentTemplate[]
    };

    return templates[category];
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
}

export const webScrapingService = new WebScrapingService();
