// Mock content engine for fallback when real generation fails
import { UnifiedContent, ContentType } from './unifiedContentEngine';

export class MockContentEngine {
  private static templates = {
    social: [
      'Breaking: Revolutionary AI breakthrough changes everything',
      'Scientists discover new quantum computing possibilities',
      'Tech industry leaders discuss future innovations',
      'Climate change solutions through technology advancement',
      'Space exploration reaches new milestones today',
      'Medical AI shows promising results in trials',
      'Renewable energy adoption accelerates globally',
      'Cybersecurity threats evolve with new countermeasures'
    ],
    entertainment: [
      'Quantum Chronicles', 'Digital Dreams', 'Neural Networks', 'Cyber Reality',
      'Future Echoes', 'Virtual Dimensions', 'AI Awakening', 'Space Odyssey 2025'
    ],
    games: [
      'Quantum Quest', 'Neural Combat', 'Digital Realms', 'Cyber Warriors',
      'Space Commander', 'AI Revolution', 'Virtual Battles', 'Future Legends'
    ],
    music: [
      'Digital Symphony', 'Quantum Beats', 'Neural Harmonies', 'Cyber Rhythms',
      'Space Melodies', 'AI Compositions', 'Virtual Sounds', 'Future Vibes'
    ],
    shop: [
      'Smart AI Assistant Device', 'Quantum Computing Module', 'Neural Interface Headset',
      'Holographic Display Unit', 'Biometric Security System', 'Smart Home Hub Pro',
      'Wireless Charging Station', 'VR Gaming Controller'
    ]
  };

  static generateMockContent(type: ContentType, count: number): UnifiedContent[] {
    const templates = this.templates[type] || this.templates.social;
    const content: UnifiedContent[] = [];

    for (let i = 0; i < count; i++) {
      const baseId = `mock_${type}_${Date.now()}_${i}`;
      const title = templates[i % templates.length];
      const timestamp = new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString();
      
      const baseContent = {
        id: baseId,
        type: type as any,
        title,
        description: this.generateDescription(title, type),
        timestamp,
        significance: Math.random() * 10,
        trending: Math.random() > 0.6,
        quality: Math.random() * 5 + 5,
        views: Math.floor(Math.random() * 1000000),
        engagement: Math.random() * 100,
        impact: this.getRandomImpact(),
        tags: this.generateTags(type),
        category: this.getCategory(type),
        rank: i + 1
      };

      if (type === 'social') {
        content.push({
          ...baseContent,
          type: 'social',
          user: {
            name: this.generateUserName(),
            username: `@${this.generateUsername()}`,
            avatar: this.generateAvatar(),
            verified: Math.random() > 0.7,
            badges: this.generateBadges(),
            rank: Math.floor(Math.random() * 1000) + 1
          },
          content: this.generateSocialContent(title),
          contentType: Math.random() > 0.5 ? 'news' : 'discussion',
          readTime: `${Math.floor(Math.random() * 5) + 1} min`
        } as any);
      } else if (type === 'entertainment') {
        content.push({
          ...baseContent,
          type: 'entertainment',
          genre: this.getRandomGenre(),
          duration: this.getRandomDuration(),
          rating: Math.random() * 2 + 3,
          thumbnail: this.getRandomThumbnail(),
          contentType: this.getRandomContentType()
        } as any);
      } else if (type === 'games') {
        content.push({
          ...baseContent,
          type: 'games',
          genre: this.getRandomGameGenre(),
          rating: Math.random() * 2 + 3,
          players: `${(Math.random() * 5 + 0.5).toFixed(1)}M`,
          price: Math.random() > 0.4 ? `$${(Math.random() * 50 + 10).toFixed(2)}` : 'Free',
          featured: Math.random() > 0.7,
          thumbnail: this.getRandomThumbnail()
        } as any);
      } else if (type === 'music') {
        content.push({
          ...baseContent,
          type: 'music',
          artist: this.generateArtistName(),
          album: this.generateAlbumName(),
          duration: this.getRandomMusicDuration(),
          plays: `${(Math.random() * 10 + 0.5).toFixed(1)}M`,
          genre: this.getRandomMusicGenre(),
          thumbnail: this.getRandomThumbnail()
        } as any);
      } else if (type === 'shop') {
        content.push({
          ...baseContent,
          type: 'shop',
          name: title,
          price: `$${(Math.random() * 500 + 50).toFixed(2)}`,
          originalPrice: Math.random() > 0.6 ? `$${(Math.random() * 100 + 600).toFixed(2)}` : undefined,
          discount: Math.floor(Math.random() * 40),
          rating: Math.random() * 2 + 3,
          reviews: Math.floor(Math.random() * 10000),
          brand: this.generateBrandName(),
          features: this.generateFeatures(),
          seller: this.generateSellerName(),
          inStock: Math.random() > 0.1,
          link: `#product-${baseId}`
        } as any);
      }
    }

    return content;
  }

  private static generateDescription(title: string, type: ContentType): string {
    const descriptions = {
      social: [
        'This breakthrough could revolutionize how we interact with technology.',
        'Industry experts are calling this a game-changing development.',
        'The implications for the future are truly exciting.',
        'This innovation opens up countless new possibilities.'
      ],
      entertainment: [
        'A thrilling journey through space and time.',
        'Stunning visuals and compelling storytelling.',
        'An epic adventure that will keep you on the edge.',
        'Revolutionary filmmaking meets incredible narrative.'
      ],
      games: [
        'Immersive gameplay with cutting-edge graphics.',
        'Strategic depth meets intense action sequences.',
        'Build, explore, and conquer in this vast universe.',
        'Team up with friends for epic multiplayer battles.'
      ],
      music: [
        'A perfect blend of electronic and orchestral elements.',
        'Innovative soundscapes that transport you to another world.',
        'Rhythmic excellence with futuristic production.',
        'Melodic genius combined with experimental techniques.'
      ],
      shop: [
        'State-of-the-art technology for the modern home.',
        'Premium quality with innovative design features.',
        'Professional-grade performance in a sleek package.',
        'The perfect blend of functionality and style.'
      ]
    };
    
    const typeDescriptions = descriptions[type] || descriptions.social;
    return typeDescriptions[Math.floor(Math.random() * typeDescriptions.length)];
  }

  private static generateUserName(): string {
    const names = ['Dr. Sarah Chen', 'Alex Rivera', 'Maya Patel', 'Jordan Kim', 'Sam Morgan', 'Riley Taylor'];
    return names[Math.floor(Math.random() * names.length)];
  }

  private static generateUsername(): string {
    const usernames = ['techguru2024', 'futuretech', 'innovator_ai', 'quantumdev', 'cyberpunk_coder', 'neural_net'];
    return usernames[Math.floor(Math.random() * usernames.length)];
  }

  private static generateAvatar(): string {
    const avatars = [
      'bg-gradient-to-br from-purple-400 to-pink-400',
      'bg-gradient-to-br from-blue-400 to-cyan-400',
      'bg-gradient-to-br from-green-400 to-emerald-400',
      'bg-gradient-to-br from-orange-400 to-red-400'
    ];
    return avatars[Math.floor(Math.random() * avatars.length)];
  }

  private static generateBadges(): string[] {
    const badges = ['Verified', 'Expert', 'Pioneer', 'Innovator'];
    return badges.slice(0, Math.floor(Math.random() * 3));
  }

  private static generateSocialContent(title: string): string {
    return `${title} - This is an exciting development that could change everything we know about the field. The research team has been working tirelessly to bring this innovation to life.`;
  }

  private static getRandomGenre(): string {
    const genres = ['Sci-Fi', 'Drama', 'Thriller', 'Action', 'Comedy', 'Documentary'];
    return genres[Math.floor(Math.random() * genres.length)];
  }

  private static getRandomGameGenre(): string {
    const genres = ['Action RPG', 'Strategy', 'FPS', 'Simulation', 'Adventure', 'Puzzle'];
    return genres[Math.floor(Math.random() * genres.length)];
  }

  private static getRandomMusicGenre(): string {
    const genres = ['Electronic', 'Ambient', 'Synthwave', 'Chillstep', 'Future Bass', 'Cyberpunk'];
    return genres[Math.floor(Math.random() * genres.length)];
  }

  private static getRandomDuration(): string {
    const hours = Math.floor(Math.random() * 3) + 1;
    const minutes = Math.floor(Math.random() * 60);
    return `${hours}h ${minutes}m`;
  }

  private static getRandomMusicDuration(): string {
    const minutes = Math.floor(Math.random() * 5) + 2;
    const seconds = Math.floor(Math.random() * 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  private static getRandomContentType(): 'movie' | 'series' | 'documentary' | 'music_video' {
    const types: ('movie' | 'series' | 'documentary' | 'music_video')[] = ['movie', 'series', 'documentary', 'music_video'];
    return types[Math.floor(Math.random() * types.length)];
  }

  private static getRandomThumbnail(): string {
    const thumbnails = [
      'bg-gradient-to-br from-purple-600 to-blue-600',
      'bg-gradient-to-br from-emerald-500 to-cyan-500',
      'bg-gradient-to-br from-orange-500 to-red-500',
      'bg-gradient-to-br from-pink-500 to-purple-500',
      'bg-gradient-to-br from-blue-500 to-purple-500',
      'bg-gradient-to-br from-green-500 to-blue-500'
    ];
    return thumbnails[Math.floor(Math.random() * thumbnails.length)];
  }

  private static getRandomImpact(): 'low' | 'medium' | 'high' | 'critical' | 'premium' {
    const impacts: ('low' | 'medium' | 'high' | 'critical' | 'premium')[] = ['low', 'medium', 'high', 'critical', 'premium'];
    return impacts[Math.floor(Math.random() * impacts.length)];
  }

  private static generateTags(type: ContentType): string[] {
    const tagSets = {
      social: ['technology', 'innovation', 'AI', 'future', 'science'],
      entertainment: ['movie', 'series', 'streaming', 'drama', 'action'],
      games: ['gaming', 'multiplayer', 'strategy', 'action', 'adventure'],
      music: ['electronic', 'ambient', 'synthwave', 'chill', 'beats'],
      shop: ['technology', 'gadgets', 'smart home', 'electronics', 'premium']
    };
    
    const tags = tagSets[type] || tagSets.social;
    return tags.slice(0, Math.floor(Math.random() * 3) + 2);
  }

  private static getCategory(type: ContentType): string {
    const categories = {
      social: 'technology',
      entertainment: 'media',
      games: 'gaming', 
      music: 'audio',
      shop: 'electronics'
    };
    return categories[type] || 'general';
  }

  private static generateArtistName(): string {
    const artists = ['Neural Symphony', 'Quantum Beats', 'Digital Dreams', 'Cyber Harmony', 'AI Collective', 'Future Sound'];
    return artists[Math.floor(Math.random() * artists.length)];
  }

  private static generateAlbumName(): string {
    const albums = ['Digital Horizons', 'Quantum Realms', 'Neural Networks', 'Cyber Futures', 'AI Chronicles', 'Virtual Worlds'];
    return albums[Math.floor(Math.random() * albums.length)];
  }

  private static generateBrandName(): string {
    const brands = ['TechCorp', 'FutureNext', 'QuantumTech', 'NeuralSoft', 'CyberDyne', 'InnovateLab'];
    return brands[Math.floor(Math.random() * brands.length)];
  }

  private static generateSellerName(): string {
    const sellers = ['TechHub Pro', 'Future Electronics', 'Smart Gadgets Co', 'Innovation Store', 'Quantum Retail', 'Digital Marketplace'];
    return sellers[Math.floor(Math.random() * sellers.length)];
  }

  private static generateFeatures(): string[] {
    const features = [
      'AI-powered intelligence', 'Wireless connectivity', 'Voice control',
      'Smart automation', 'Energy efficient', 'Premium materials',
      'Advanced security', 'Cloud integration', 'Mobile app control'
    ];
    return features.slice(0, Math.floor(Math.random() * 4) + 2);
  }
}