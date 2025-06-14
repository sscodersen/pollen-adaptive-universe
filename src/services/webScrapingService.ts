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
  };
}

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
      // In production, this would call actual scraping APIs
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

      // Cache results
      this.cache.set(cacheKey, highQualityContent);
      this.lastScrape.set(cacheKey, now);

      return highQualityContent;
    } catch (error) {
      console.error(`Scraping failed for ${category}:`, error);
      return this.cache.get(cacheKey) || [];
    }
  }

  private async performScraping(category: 'news' | 'shop' | 'entertainment', count: number): Promise<ScrapedContent[]> {
    // Simulate real web scraping with high-quality content
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

    const metadata: any = {
        author: template.author,
        publishDate: new Date(Date.now() - Math.random() * 86400000).toISOString(),
        tags: template.tags,
        ...(template.metadata || {})
    };

    if (category === 'shop') {
        metadata.rating = metadata.rating || (Math.random() * 1.5 + 3.5);
        metadata.image = `https://picsum.photos/400/300?random=${index}`;
    }

    return {
      id: `${category}-${Date.now()}-${index}`,
      title: template.title,
      url: metadata.link || `https://${source.domain}/${template.path}/${Math.random().toString(36).substring(7)}`,
      description: template.description,
      content: template.content,
      source: source.name,
      timestamp: Date.now() - Math.random() * 86400000, // Last 24 hours
      category,
      significance: 0, // Will be calculated later
      metadata
    };
  }

  private getSourcesForCategory(category: 'news' | 'shop' | 'entertainment') {
    const sources = {
      news: [
        { name: 'TechCrunch', domain: 'techcrunch.com' },
        { name: 'Wired', domain: 'wired.com' },
        { name: 'MIT Technology Review', domain: 'technologyreview.com' },
        { name: 'IEEE Spectrum', domain: 'spectrum.ieee.org' },
        { name: 'Nature', domain: 'nature.com' },
        { name: 'Science Daily', domain: 'sciencedaily.com' }
      ],
      shop: [
        { name: 'Amazon', domain: 'amazon.com' },
        { name: 'Allbirds', domain: 'allbirds.com' },
        { name: 'Click & Grow', domain: 'clickandgrow.com' },
        { name: 'Fellow', domain: 'fellowproducts.com' },
        { name: 'Patagonia', domain: 'patagonia.com' },
        { name: 'Therabody', domain: 'therabody.com' },
        { name: 'Blueland', domain: 'blueland.com' },
        { name: 'Oura', domain: 'ouraring.com' },
        { name: 'LARQ', domain: 'livelarq.com' },
        { name: 'TechPro', domain: 'techpro.com' },
      ],
      entertainment: [
        { name: 'Steam', domain: 'steampowered.com' },
        { name: 'Epic Games', domain: 'epicgames.com' },
        { name: 'Spotify', domain: 'spotify.com' },
        { name: 'Netflix', domain: 'netflix.com' },
        { name: 'YouTube', domain: 'youtube.com' },
        { name: 'Twitch', domain: 'twitch.tv' }
      ]
    };

    return sources[category];
  }

  private getContentTemplates(category: 'news' | 'shop' | 'entertainment') {
    const templates = {
      news: [
        {
          title: 'Revolutionary AI Breakthrough Transforms Healthcare Industry',
          description: 'New machine learning model achieves 99% accuracy in early disease detection',
          content: 'Researchers at leading institutions have developed an AI system that can detect early-stage diseases with unprecedented accuracy. The breakthrough technology uses advanced neural networks to analyze medical imaging data, potentially saving millions of lives through early intervention.',
          author: 'Dr. Sarah Chen',
          tags: ['AI', 'Healthcare', 'Technology', 'Innovation'],
          path: 'article'
        },
        {
          title: 'Quantum Computing Milestone Achieved by Tech Giants',
          description: 'Major advancement in quantum error correction brings practical quantum computing closer',
          content: 'A consortium of technology companies has achieved a significant milestone in quantum computing, demonstrating reliable quantum error correction at scale. This advancement brings us closer to practical quantum computers that could revolutionize cryptography, drug discovery, and optimization problems.',
          author: 'Michael Rodriguez',
          tags: ['Quantum Computing', 'Technology', 'Innovation', 'Science'],
          path: 'tech'
        }
      ],
      shop: [
        {
          title: 'Professional AI Development Workstation - RTX 4090 Setup',
          description: 'Complete AI/ML development setup with cutting-edge GPU and optimized cooling',
          content: 'Pre-configured workstation designed specifically for AI development and machine learning tasks. Features RTX 4090 GPU, 128GB RAM, and optimized cooling system for sustained high-performance computing.',
          author: 'TechPro Solutions',
          tags: ['AI Hardware', 'Workstation', 'GPU', 'Development'],
          path: 'product',
          metadata: { price: 4999.99, productCategory: 'Health Tech', features: ['RTX 4090', '128GB DDR5 RAM', 'Liquid Cooling'] }
        },
        {
          title: 'Advanced Neural Network Training Software Suite',
          description: 'Professional-grade ML training platform with visual workflow builder',
          content: 'Comprehensive software suite for training and deploying neural networks. Includes visual workflow builder, automated hyperparameter tuning, and production deployment tools.',
          author: 'AI Solutions Inc.',
          tags: ['Software', 'AI Training', 'Machine Learning', 'Tools'],
          path: 'software',
          metadata: { price: 999, productCategory: 'Health Tech', features: ['Visual Workflow Builder', 'AutoML', 'Deployment Tools'] }
        },
        {
          title: 'Allbirds Wool Runners',
          description: 'Sustainable and comfortable everyday sneakers made from merino wool and recycled materials.',
          content: 'Sustainable and comfortable everyday sneakers made from merino wool and recycled materials.',
          author: 'Allbirds',
          tags: ['Sustainable', 'Footwear', 'Comfort'],
          path: 'products/womens-wool-runners',
          metadata: { price: 95.00, productCategory: 'Fashion', link: 'https://www.allbirds.com/products/womens-wool-runners', features: ['Merino Wool Upper', 'SweetFoam™ midsole', 'Machine washable'], discount: 0 }
        },
        {
          title: 'Click & Grow Smart Garden 3',
          description: 'An innovative indoor garden that cares for itself. Grow fresh, flavourful herbs, fruits or vegetables in your home.',
          content: 'An innovative indoor garden that cares for itself. Grow fresh, flavourful herbs, fruits or vegetables in your home.',
          author: 'Click & Grow',
          tags: ['Gardening', 'Smart Home', 'Wellness'],
          path: 'products/the-smart-garden-3',
          metadata: { price: 99.95, productCategory: 'Home Goods', link: 'https://www.clickandgrow.com/products/the-smart-garden-3', features: ['Automated watering', 'Perfect amount of light', '3 complimentary plant pods'], discount: 0 }
        },
        {
          title: 'Fellow Stagg EKG Electric Kettle',
          description: 'A beautifully designed electric kettle for the perfect pour-over coffee. Variable temperature control and a precision spout.',
          content: 'A beautifully designed electric kettle for the perfect pour-over coffee. Variable temperature control and a precision spout.',
          author: 'Fellow',
          tags: ['Coffee', 'Kitchen', 'Design'],
          path: 'products/stagg-ekg-electric-pour-over-kettle',
          metadata: { price: 165.00, originalPrice: 195.00, productCategory: 'Home Goods', link: 'https://fellowproducts.com/products/stagg-ekg-electric-pour-over-kettle', features: ['Variable temperature control', 'LCD screen', 'Precision pour spout'], discount: 15 }
        },
        {
          title: 'Patagonia Nano Puff Jacket',
          description: 'Warm, windproof, water-resistant—the Nano Puff® Jacket uses incredibly lightweight and highly compressible 60-g PrimaLoft® Gold Insulation Eco.',
          content: 'Warm, windproof, water-resistant—the Nano Puff® Jacket uses incredibly lightweight and highly compressible 60-g PrimaLoft® Gold Insulation Eco.',
          author: 'Patagonia',
          tags: ['Outdoor', 'Sustainable', 'Recycled'],
          path: 'product/mens-nano-puff-jacket/84212.html',
          metadata: { price: 239.00, productCategory: 'Fashion', link: 'https://www.patagonia.com/product/mens-nano-puff-jacket/84212.html', features: ['100% recycled shell', 'PrimaLoft® Gold Insulation', 'Fair Trade Certified™'], discount: 0 }
        },
        {
          title: 'Theragun Mini',
          description: 'A portable, powerful percussive therapy device. Compact but powerful, the Theragun mini is the most agile massage device that goes wherever you do.',
          content: 'A portable, powerful percussive therapy device. Compact but powerful, the Theragun mini is the most agile massage device that goes wherever you do.',
          author: 'Therabody',
          tags: ['Fitness', 'Recovery', 'Health'],
          path: 'us/en-us/mini-us.html',
          metadata: { price: 199.00, productCategory: 'Wellness', link: 'https://www.therabody.com/us/en-us/mini-us.html', features: ['QuietForce Technology', '3 Speed Settings', '150-minute battery life'], discount: 0 }
        },
        {
          title: 'Blueland The Clean Essentials Kit',
          description: 'A revolutionary way to clean your home without plastic waste. Reusable bottles and tablet refills for hand soap, multi-surface cleaner, and more.',
          content: 'A revolutionary way to clean your home without plastic waste. Reusable bottles and tablet refills for hand soap, multi-surface cleaner, and more.',
          author: 'Blueland',
          tags: ['Sustainable', 'Cleaning', 'Zero Waste'],
          path: 'products/the-clean-essentials',
          metadata: { price: 39.00, productCategory: 'Eco-Friendly', link: 'https://www.blueland.com/products/the-clean-essentials', features: ['Reduces plastic waste', 'Non-toxic formulas', 'Reusable bottles'], discount: 0 }
        },
        {
          title: 'Oura Ring Gen3',
          description: 'A smart ring that tracks your sleep, activity, recovery, temperature, heart rate, stress, and more.',
          content: 'A smart ring that tracks your sleep, activity, recovery, temperature, heart rate, stress, and more.',
          author: 'Oura',
          tags: ['Wearable', 'Health', 'Sleep Tracking'],
          path: '/',
          metadata: { price: 299.00, productCategory: 'Health Tech', link: 'https://ouraring.com/', features: ['24/7 heart rate monitoring', 'Advanced sleep analysis', '7-day battery life'], discount: 0 }
        },
        {
          title: 'LARQ Bottle PureVis™',
          description: 'The world’s first self-cleaning water bottle and water purification system. It uses PureVis technology to eliminate up to 99% of bio-contaminants.',
          content: 'The world’s first self-cleaning water bottle and water purification system. It uses PureVis technology to eliminate up to 99% of bio-contaminants.',
          author: 'LARQ',
          tags: ['Health', 'Outdoors', 'Tech', 'Sustainable'],
          path: 'product/larq-bottle-purevis',
          metadata: { price: 99.00, productCategory: 'Wellness', link: 'https://www.livelarq.com/product/larq-bottle-purevis', features: ['Self-cleaning mode', 'Eliminates bacteria & viruses', 'Keeps water cold 24h'], discount: 0 }
        }
      ],
      entertainment: [
        {
          title: 'Neural Symphony: AI-Generated Interactive Music Experience',
          description: 'Revolutionary music platform that creates personalized symphonies in real-time',
          content: 'Experience music like never before with Neural Symphony, an AI-powered platform that generates unique musical compositions based on your emotions, activities, and preferences. Each listening session creates a completely original piece of music.',
          author: 'Harmonic AI Studios',
          tags: ['AI Music', 'Interactive', 'Generative', 'Audio'],
          path: 'experience'
        },
        {
          title: 'Quantum Dreamscape: Procedural Reality Simulation',
          description: 'Infinite virtual worlds generated by quantum-inspired algorithms',
          content: 'Explore limitless virtual environments in Quantum Dreamscape, where every world is procedurally generated using quantum-inspired algorithms. No two experiences are ever the same.',
          author: 'Virtual Worlds Studio',
          tags: ['Virtual Reality', 'Procedural', 'Gaming', 'Simulation'],
          path: 'game'
        }
      ]
    };

    return templates[category];
  }

  async getTrendingKeywords(category?: 'news' | 'shop' | 'entertainment'): Promise<string[]> {
    // In production, this would analyze scraped content for trending keywords
    const trending = {
      news: [
        'artificial intelligence breakthroughs',
        'quantum computing advances',
        'renewable energy innovation',
        'biotechnology developments',
        'space exploration missions',
        'autonomous vehicle progress'
      ],
      shop: [
        'AI development hardware',
        'professional workstations',
        'machine learning software',
        'automation tools',
        'productivity suites',
        'creative design platforms'
      ],
      entertainment: [
        'interactive AI experiences',
        'procedural content generation',
        'virtual reality platforms',
        'AI-generated music',
        'collaborative gaming',
        'educational simulations'
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
