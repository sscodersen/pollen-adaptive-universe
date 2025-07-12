import { pollenAI } from './pollenAI';
import { significanceAlgorithm } from './significanceAlgorithm';
import { rankItems } from './generalRanker';
import { storageService } from './storageService';

// Universal content types
export type ContentType = 'social' | 'shop' | 'entertainment' | 'games' | 'music' | 'news' | 'explore';

export interface BaseContent {
  id: string;
  type: ContentType;
  title: string;
  description: string;
  timestamp: string;
  significance: number;
  trending: boolean;
  quality: number;
  views: number;
  engagement: number;
  impact: 'low' | 'medium' | 'high' | 'critical' | 'premium';
  tags: string[];
  category: string;
  rank?: number;
}

export interface SocialContent extends BaseContent {
  type: 'social';
  user: {
    name: string;
    username: string;
    avatar: string;
    verified: boolean;
    badges: string[];
    rank: number;
  };
  content: string;
  contentType: 'social' | 'news' | 'discussion';
  readTime: string;
}

export interface ShopContent extends BaseContent {
  type: 'shop';
  name: string;
  price: string;
  originalPrice?: string;
  discount: number;
  rating: number;
  reviews: number;
  brand: string;
  features: string[];
  seller: string;
  inStock: boolean;
  link: string;
}

export interface EntertainmentContent extends BaseContent {
  type: 'entertainment';
  genre: string;
  duration: string;
  rating: number;
  thumbnail: string;
  contentType: 'movie' | 'series' | 'documentary' | 'music_video';
}

export interface GameContent extends BaseContent {
  type: 'games';
  genre: string;
  rating: number;
  players: string;
  price: string;
  featured: boolean;
  thumbnail: string;
}

export interface MusicContent extends BaseContent {
  type: 'music';
  artist: string;
  album: string;
  duration: string;
  plays: string;
  genre: string;
  thumbnail: string;
}

export interface NewsContent extends BaseContent {
  type: 'news';
  source: string;
  category: string;
  snippet: string;
  readTime: string;
}

export type UnifiedContent = SocialContent | ShopContent | EntertainmentContent | GameContent | MusicContent | NewsContent;

// Advanced content generation strategies
export interface GenerationStrategy {
  diversity: number; // 0-1, how diverse the content should be
  freshness: number; // 0-1, how much to prioritize new vs proven content
  personalization: number; // 0-1, how much to personalize (future feature)
  qualityThreshold: number; // minimum quality score (0-10)
  trendingBoost: number; // multiplier for trending content
}

// Content templates with dynamic generation
class ContentTemplateEngine {
  private realUsers = [
    { name: 'Dr. Sarah Chen', username: 'sarahchen_ai', avatar: 'bg-gradient-to-r from-blue-500 to-purple-500', verified: true, badges: ['AI Expert', 'Researcher'], rank: 98 },
    { name: 'Marcus Rodriguez', username: 'marcus_dev', avatar: 'bg-gradient-to-r from-green-500 to-blue-500', verified: true, badges: ['Developer', 'Open Source'], rank: 95 },
    { name: 'Elena Kowalski', username: 'elena_design', avatar: 'bg-gradient-to-r from-pink-500 to-red-500', verified: false, badges: ['Designer', 'UX'], rank: 87 },
    { name: 'Alex Chen', username: 'alex_crypto', avatar: 'bg-gradient-to-r from-yellow-500 to-orange-500', verified: false, badges: ['Blockchain', 'DeFi'], rank: 92 },
    { name: 'Maya Thompson', username: 'maya_startup', avatar: 'bg-gradient-to-r from-purple-500 to-pink-500', verified: true, badges: ['Entrepreneur', 'VC'], rank: 94 },
    { name: 'Global News Network', username: 'gnn_official', avatar: 'bg-gradient-to-r from-red-600 to-red-400', verified: true, badges: ['News', 'Breaking'], rank: 96 },
    { name: 'TechCrunch', username: 'techcrunch', avatar: 'bg-gradient-to-r from-emerald-500 to-cyan-500', verified: true, badges: ['Tech News', 'Verified'], rank: 97 },
    { name: 'The Economist', username: 'theeconomist', avatar: 'bg-gradient-to-r from-indigo-500 to-purple-500', verified: true, badges: ['Economics', 'Analysis'], rank: 99 },
    { name: 'Nature Journal', username: 'nature_journal', avatar: 'bg-gradient-to-r from-green-600 to-emerald-500', verified: true, badges: ['Science', 'Research'], rank: 99 },
    { name: 'World Health Org', username: 'who_official', avatar: 'bg-gradient-to-r from-blue-600 to-cyan-500', verified: true, badges: ['Health', 'Global'], rank: 98 }
  ];

  private trendingTopics = [
    'AI Consciousness', 'Quantum Computing', 'Climate Technology', 'Space Colonization', 
    'Digital Currency', 'Gene Therapy', 'Virtual Reality', 'Sustainable Energy',
    'Neural Interfaces', 'Smart Cities', 'Fusion Energy', 'Biotech Revolution'
  ];

  private generateThumbnail(category: string): string {
    const thumbnails = {
      'Technology': 'bg-gradient-to-br from-blue-500 to-cyan-500',
      'Science': 'bg-gradient-to-br from-purple-600 to-blue-600',
      'Entertainment': 'bg-gradient-to-br from-pink-500 to-purple-500',
      'Gaming': 'bg-gradient-to-br from-green-500 to-emerald-500',
      'Music': 'bg-gradient-to-br from-orange-500 to-red-500',
      'Health': 'bg-gradient-to-br from-emerald-500 to-cyan-500',
      'Business': 'bg-gradient-to-br from-yellow-500 to-orange-500',
      'News': 'bg-gradient-to-br from-red-500 to-pink-500'
    };
    return thumbnails[category as keyof typeof thumbnails] || 'bg-gradient-to-br from-gray-500 to-gray-700';
  }

  async generateSocialContent(strategy: GenerationStrategy): Promise<SocialContent> {
    const user = this.realUsers[Math.floor(Math.random() * this.realUsers.length)];
    const topic = this.trendingTopics[Math.floor(Math.random() * this.trendingTopics.length)];
    
    // Generate intelligent content
    const pollenResponse = await pollenAI.generate(
      `Create an engaging social media post about ${topic}. Make it thought-provoking, 
       informative, and relevant to current trends. Keep it conversational but insightful.
       Focus on: technology, innovation, future implications, human impact.`,
      'social'
    );
    
    // Use Pollen AI response directly (no more dummy responses)
    const content = pollenResponse.content;
    
    const scored = significanceAlgorithm.scoreContent(pollenResponse.content, 'social', user.name);
    const timestamp = `${Math.floor(Math.random() * 240) + 1}m`;
    
    return {
      id: `social-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'social',
      title: `${user.name} shared insights about ${topic}`,
      description: content,
      content: content,
      user,
      timestamp,
      contentType: scored.significanceScore > 8.5 ? 'news' : scored.significanceScore > 7 ? 'discussion' : 'social',
      significance: scored.significanceScore,
      trending: scored.significanceScore > 7.5 || Math.random() > 0.7,
      quality: Math.floor(scored.significanceScore * 10),
      views: Math.floor(Math.random() * 50000) + 1000,
      engagement: Math.floor(Math.random() * 100) + 50,
      impact: scored.significanceScore > 9 ? 'critical' : 
              scored.significanceScore > 8 ? 'high' : 
              scored.significanceScore > 6.5 ? 'medium' : 'low',
      tags: [topic, scored.significanceScore > 8 ? 'High Impact' : 'Trending'],
      category: this.getCategoryFromTopic(topic),
      readTime: `${Math.floor(Math.random() * 5) + 1} min read`
    };
  }

  async generateShopContent(strategy: GenerationStrategy): Promise<ShopContent> {
    const categories = ['Smart Home', 'Gaming', 'Audio', 'Health', 'Photography', 'Fitness', 'Office', 'Wearables'];
    const category = categories[Math.floor(Math.random() * categories.length)];
    
    const pollenResponse = await pollenAI.generate(
      `Create an innovative product in the ${category} category. Include name, description, 
       key features, and compelling selling points. Make it cutting-edge and desirable.`,
      'shop'
    );
    
    const scored = significanceAlgorithm.scoreContent(pollenResponse.content, 'shop', category);
    const price = Math.floor(Math.random() * 800 + 50);
    const hasDiscount = Math.random() > 0.4;
    const discount = hasDiscount ? Math.floor(Math.random() * 40 + 10) : 0;
    
    // Extract product details from Pollen AI response (simplified)
    const lines = pollenResponse.content.split('\n').filter(l => l.trim());
    const name = lines[0]?.replace(/^.*?:/, '').trim() || `AI-Powered ${category} Device`;
    const description = lines.slice(1).join(' ').substring(0, 200) + '...';
    
    return {
      id: `shop-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'shop',
      title: name,
      name,
      description,
      price: `$${price}`,
      originalPrice: hasDiscount ? `$${Math.floor(price / (1 - discount/100))}` : undefined,
      discount,
      rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
      reviews: Math.floor(Math.random() * 5000) + 100,
      brand: this.generateBrandName(category),
      features: this.extractFeatures(pollenResponse.content),
      seller: this.generateBrandName(category),
      inStock: Math.random() > 0.1,
      link: `https://example.com/product/${name.toLowerCase().replace(/\s+/g, '-')}`,
      timestamp: `${Math.floor(Math.random() * 24)}h ago`,
      significance: scored.significanceScore,
      trending: scored.significanceScore > 7.5 || Math.random() > 0.7,
      quality: Math.floor(scored.significanceScore * 10),
      views: Math.floor(Math.random() * 25000) + 500,
      engagement: Math.floor(Math.random() * 100) + 30,
      impact: scored.significanceScore > 9 ? 'premium' : 
              scored.significanceScore > 8 ? 'high' : 
              scored.significanceScore > 6.5 ? 'medium' : 'low',
      tags: ['AI', 'Smart', 'Premium', 'Innovative'],
      category
    };
  }

  async generateEntertainmentContent(strategy: GenerationStrategy): Promise<EntertainmentContent> {
    const types = ['movie', 'series', 'documentary', 'music_video'] as const;
    const genres = ['Sci-Fi', 'Thriller', 'Drama', 'Action', 'Technology', 'Future', 'AI'];
    
    const contentType = types[Math.floor(Math.random() * types.length)];
    const genre = genres[Math.floor(Math.random() * genres.length)];
    
    const pollenResponse = await pollenAI.generate(
      `Create a compelling ${contentType} concept in the ${genre} genre. 
       Include title, plot summary, and what makes it unique. 
       Focus on futuristic, thought-provoking themes.`,
      'entertainment'
    );
    
    const scored = significanceAlgorithm.scoreContent(pollenResponse.content, 'entertainment', genre);
    const lines = pollenResponse.content.split('\n').filter(l => l.trim());
    const title = lines[0]?.replace(/^.*?:/, '').trim() || `The ${genre} ${contentType}`;
    
    return {
      id: `entertainment-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'entertainment',
      title,
      description: pollenResponse.content,
      contentType,
      genre,
      duration: this.generateDuration(contentType),
      rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
      thumbnail: this.generateThumbnail('Entertainment'),
      timestamp: `${Math.floor(Math.random() * 24)}h ago`,
      significance: scored.significanceScore,
      trending: scored.significanceScore > 7.5 || Math.random() > 0.7,
      quality: Math.floor(scored.significanceScore * 10),
      views: Math.floor(Math.random() * 5000000) + 100000,
      engagement: Math.floor(Math.random() * 100) + 40,
      impact: scored.significanceScore > 9 ? 'critical' : 
              scored.significanceScore > 8 ? 'high' : 
              scored.significanceScore > 6.5 ? 'medium' : 'low',
      tags: [genre, contentType, 'Trending'],
      category: 'Entertainment'
    };
  }

  async generateGameContent(strategy: GenerationStrategy): Promise<GameContent> {
    const genres = ['Action RPG', 'Strategy', 'FPS', 'Simulation', 'Puzzle', 'Adventure'];
    const genre = genres[Math.floor(Math.random() * genres.length)];
    
    const pollenResponse = await pollenAI.generate(
      `Create an innovative ${genre} game concept. Include title, gameplay mechanics, 
       and what makes it revolutionary. Focus on cutting-edge technology and engaging gameplay.`,
      'entertainment'
    );
    
    const scored = significanceAlgorithm.scoreContent(pollenResponse.content, 'entertainment', genre);
    const lines = pollenResponse.content.split('\n').filter(l => l.trim());
    const title = lines[0]?.replace(/^.*?:/, '').trim() || `${genre} Revolution`;
    const isFree = Math.random() > 0.6;
    
    return {
      id: `game-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'games',
      title,
      description: pollenResponse.content,
      genre,
      rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
      players: `${Math.floor(Math.random() * 5000 + 500)}K`,
      price: isFree ? 'Free' : `$${Math.floor(Math.random() * 60 + 20)}`,
      featured: scored.significanceScore > 8 || Math.random() > 0.7,
      thumbnail: this.generateThumbnail('Gaming'),
      timestamp: `${Math.floor(Math.random() * 24)}h ago`,
      significance: scored.significanceScore,
      trending: scored.significanceScore > 7.5 || Math.random() > 0.7,
      quality: Math.floor(scored.significanceScore * 10),
      views: Math.floor(Math.random() * 1000000) + 50000,
      engagement: Math.floor(Math.random() * 100) + 30,
      impact: scored.significanceScore > 9 ? 'critical' : 
              scored.significanceScore > 8 ? 'high' : 
              scored.significanceScore > 6.5 ? 'medium' : 'low',
      tags: [genre, 'Gaming', scored.significanceScore > 8 ? 'Featured' : 'New'],
      category: 'Gaming'
    };
  }

  async generateMusicContent(strategy: GenerationStrategy): Promise<MusicContent> {
    const genres = ['Electronic', 'Ambient', 'Synthwave', 'Chiptune', 'Future Bass', 'Lo-Fi'];
    const genre = genres[Math.floor(Math.random() * genres.length)];
    
    const pollenResponse = await pollenAI.generate(
      `Create a compelling ${genre} music track concept. Include title, artist name, 
       and description of the sound/vibe. Make it futuristic and emotionally engaging.`,
      'entertainment'
    );
    
    const scored = significanceAlgorithm.scoreContent(pollenResponse.content, 'entertainment', genre);
    const lines = pollenResponse.content.split('\n').filter(l => l.trim());
    const title = lines[0]?.replace(/^.*?:/, '').trim() || `${genre} Dreams`;
    const artist = this.generateArtistName(genre);
    
    return {
      id: `music-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'music',
      title,
      description: pollenResponse.content,
      artist,
      album: this.generateAlbumName(genre),
      duration: `${Math.floor(Math.random() * 3 + 2)}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`,
      plays: `${(Math.random() * 5 + 0.5).toFixed(1)}M`,
      genre,
      thumbnail: this.generateThumbnail('Music'),
      timestamp: `${Math.floor(Math.random() * 24)}h ago`,
      significance: scored.significanceScore,
      trending: scored.significanceScore > 7.5 || Math.random() > 0.7,
      quality: Math.floor(scored.significanceScore * 10),
      views: Math.floor(Math.random() * 10000000) + 100000,
      engagement: Math.floor(Math.random() * 100) + 40,
      impact: scored.significanceScore > 9 ? 'critical' : 
              scored.significanceScore > 8 ? 'high' : 
              scored.significanceScore > 6.5 ? 'medium' : 'low',
      tags: [genre, 'Music', 'AI Generated'],
      category: 'Music'
    };
  }

  private getCategoryFromTopic(topic: string): string {
    const categoryMap: { [key: string]: string } = {
      'AI Consciousness': 'Technology',
      'Quantum Computing': 'Science',
      'Climate Technology': 'Environment',
      'Space Colonization': 'Science',
      'Digital Currency': 'Business',
      'Gene Therapy': 'Health',
      'Virtual Reality': 'Technology',
      'Sustainable Energy': 'Environment',
      'Neural Interfaces': 'Technology',
      'Smart Cities': 'Technology',
      'Fusion Energy': 'Science',
      'Biotech Revolution': 'Health'
    };
    return categoryMap[topic] || 'General';
  }

  private generateBrandName(category: string): string {
    const prefixes = ['Tech', 'Smart', 'Neo', 'Quantum', 'Digital', 'AI', 'Cyber', 'Future'];
    const suffixes = ['Labs', 'Systems', 'Tech', 'Works', 'Solutions', 'Dynamics', 'Pro', 'Gen'];
    
    const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
    const suffix = suffixes[Math.floor(Math.random() * suffixes.length)];
    
    return `${prefix}${suffix}`;
  }

  private generateArtistName(genre: string): string {
    const electronic = ['Synthwave Collective', 'Digital Echo', 'Neural Symphony', 'Code Harmony', 'Cyber Beats'];
    const ambient = ['Ethereal Sounds', 'Cosmic Drift', 'Deep Space', 'Mindful Waves', 'Zen Circuit'];
    
    if (genre === 'Electronic' || genre === 'Synthwave') {
      return electronic[Math.floor(Math.random() * electronic.length)];
    } else if (genre === 'Ambient') {
      return ambient[Math.floor(Math.random() * ambient.length)];
    }
    
    return 'AI Composer';
  }

  private generateAlbumName(genre: string): string {
    const albums = ['Digital Dreams', 'Future Sounds', 'AI Compositions', 'Neural Networks', 'Cosmic Journey'];
    return albums[Math.floor(Math.random() * albums.length)];
  }

  private generateDuration(type: string): string {
    if (type === 'movie') return `${Math.floor(Math.random() * 60 + 90)}m`;
    if (type === 'series') return `${Math.floor(Math.random() * 12 + 6)} episodes`;
    if (type === 'documentary') return `${Math.floor(Math.random() * 30 + 45)}m`;
    if (type === 'music_video') return `${Math.floor(Math.random() * 3 + 2)}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`;
    return '2h 15m';
  }

  private generateFallbackSocialContent(topic: string): string {
    const fallbacks = {
      'AI Consciousness': 'The question of AI consciousness isn\'t just philosophical anymore - it\'s becoming practical. As our AI systems become more sophisticated, we need frameworks for understanding their cognitive processes. What does it mean for an AI to "think" vs simply process?',
      'Quantum Computing': 'Quantum computing is reaching practical applications faster than expected. IBM\'s latest quantum processor achieved quantum advantage in optimization problems, potentially revolutionizing logistics, finance, and drug discovery within the next decade.',
      'Climate Technology': 'Carbon capture technology just hit a major milestone - direct air capture facilities can now remove CO2 at $150/ton. At scale, this could make carbon removal economically viable for offsetting emissions.',
      'Space Colonization': 'SpaceX\'s latest Starship test brings Mars colonization closer to reality. The successful orbital refueling demonstration proves we can transport the massive payloads needed for sustainable off-world settlements.',
      'Digital Currency': 'Central Bank Digital Currencies (CBDCs) are reshaping global finance. 14 countries have already launched digital versions of their currencies, potentially eliminating traditional banking intermediaries.',
      'Gene Therapy': 'CRISPR 3.0 just entered clinical trials with unprecedented precision. The new base editing technique can correct genetic mutations without double-strand DNA breaks, dramatically reducing side effects.',
      'Virtual Reality': 'Apple\'s Vision Pro is just the beginning. The next wave of AR/VR will feature neural interfaces that read brain signals directly, eliminating the need for hand controllers entirely.',
      'Sustainable Energy': 'Perovskite solar cells achieved 30% efficiency in lab tests - a breakthrough that could make solar power cheaper than coal worldwide. Mass production begins in 2025.',
      'Neural Interfaces': 'Neuralink\'s first human patient can now control computers with thought alone. This technology could restore mobility to paralyzed patients and eventually augment human cognitive abilities.',
      'Smart Cities': 'Singapore\'s AI-powered traffic system reduced congestion by 40% using predictive analytics. The system anticipates traffic patterns and adjusts signals in real-time across the entire city.',
      'Fusion Energy': 'The National Ignition Facility achieved net energy gain from fusion for the third time this year. Private fusion companies are now targeting commercial power generation by 2035.',
      'Biotech Revolution': 'AI-designed proteins are revolutionizing medicine. DeepMind\'s AlphaFold predictions helped create new treatments for Alzheimer\'s, cutting drug development time from decades to years.'
    };
    
    return fallbacks[topic as keyof typeof fallbacks] || `Revolutionary advances in ${topic} are reshaping our understanding of what's possible. The implications for society, technology, and human potential are profound.`;
  }

  private extractFeatures(content: string): string[] {
    // Simple feature extraction from AI-generated content
    const features = content.match(/\b[\w\s]{10,30}(?=\.|,|$)/g) || [];
    return features.slice(0, 3).map(f => f.trim());
  }
}

class UnifiedContentEngine {
  private templateEngine = new ContentTemplateEngine();
  private cache = new Map<string, { content: UnifiedContent[], timestamp: number }>();
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes

  private defaultStrategy: GenerationStrategy = {
    diversity: 0.8,
    freshness: 0.7,
    personalization: 0.5,
    qualityThreshold: 6.0,
    trendingBoost: 1.5
  };

  async generateContent(
    type: ContentType, 
    count: number = 10, 
    strategy: Partial<GenerationStrategy> = {}
  ): Promise<UnifiedContent[]> {
    const finalStrategy = { ...this.defaultStrategy, ...strategy };
    const cacheKey = `${type}-${count}-${JSON.stringify(finalStrategy)}`;
    
    // Check cache first
    const cached = this.cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.content;
    }

    console.log(`🎯 Generating ${count} ${type} content items with strategy:`, finalStrategy);
    
    const content: UnifiedContent[] = [];
    
    // Generate content based on type
    for (let i = 0; i < count; i++) {
      try {
        let item: UnifiedContent;
        
        switch (type) {
          case 'social':
            item = await this.templateEngine.generateSocialContent(finalStrategy);
            break;
          case 'shop':
            item = await this.templateEngine.generateShopContent(finalStrategy);
            break;
          case 'entertainment':
            item = await this.templateEngine.generateEntertainmentContent(finalStrategy);
            break;
          case 'games':
            item = await this.templateEngine.generateGameContent(finalStrategy);
            break;
          case 'music':
            item = await this.templateEngine.generateMusicContent(finalStrategy);
            break;
          default:
            continue;
        }
        
        // Quality filtering
        if (item.quality >= finalStrategy.qualityThreshold * 10) {
          content.push(item);
        }
      } catch (error) {
        console.error(`Error generating ${type} content:`, error);
      }
    }
    
    // Apply ranking and optimization
    const rankedContent = this.optimizeContent(content, finalStrategy);
    
    // Cache the results
    this.cache.set(cacheKey, { content: rankedContent, timestamp: Date.now() });
    
    // Persist to storage for analytics
    await this.persistContentToStorage(type, rankedContent);
    
    return rankedContent;
  }

  private optimizeContent(content: UnifiedContent[], strategy: GenerationStrategy): UnifiedContent[] {
    // Apply trending boost
    const boostedContent = content.map(item => ({
      ...item,
      significance: item.trending ? item.significance * strategy.trendingBoost : item.significance
    }));

    // Rank using the general ranker
    const ranked = rankItems(boostedContent, { type: this.getRankType(content[0]?.type) });
    
    // Add rank to each item
    return ranked.map((item, index) => ({ ...item, rank: index + 1 }));
  }

  private getRankType(contentType: ContentType): "shop" | "news" | "entertainment" | "social" {
    switch (contentType) {
      case 'shop': return 'shop';
      case 'social': return 'social';
      case 'entertainment':
      case 'games':
      case 'music': return 'entertainment';
      default: return 'news';
    }
  }

  private async persistContentToStorage(type: ContentType, content: UnifiedContent[]): Promise<void> {
    try {
      const key = `generated_content_${type}_${Date.now()}`;
      await storageService.setCache(key, {
        type,
        content: content.slice(0, 5), // Store only top 5 for analytics
        timestamp: Date.now(),
        metrics: {
          averageSignificance: content.reduce((sum, item) => sum + item.significance, 0) / content.length,
          trendingCount: content.filter(item => item.trending).length,
          highQualityCount: content.filter(item => item.quality >= 80).length
        }
      });
    } catch (error) {
      console.error('Failed to persist content to storage:', error);
    }
  }

  // Real-time content streaming simulation
  async streamContent(type: ContentType, callback: (content: UnifiedContent) => void): Promise<void> {
    const interval = setInterval(async () => {
      try {
        const newContent = await this.generateContent(type, 1);
        if (newContent.length > 0) {
          callback(newContent[0]);
        }
      } catch (error) {
        console.error('Error in content streaming:', error);
      }
    }, 30000); // Generate new content every 30 seconds

    // Auto-cleanup after 10 minutes
    setTimeout(() => clearInterval(interval), 10 * 60 * 1000);
  }

  // Clear cache manually
  clearCache(): void {
    this.cache.clear();
  }

  // Get cache statistics
  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys())
    };
  }
}

export const unifiedContentEngine = new UnifiedContentEngine();