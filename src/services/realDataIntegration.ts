import { storageService } from './storageService';

// Free data sources we can integrate with
export interface DataSource {
  id: string;
  name: string;
  type: 'news' | 'music' | 'content' | 'apps' | 'educational';
  endpoint: string;
  apiKey?: string;
  params?: Record<string, any>;
  lastFetch?: string;
  isActive: boolean;
}

export interface ExternalContent {
  id: string;
  title: string;
  description: string;
  content?: string;
  url: string;
  imageUrl?: string;
  category: string;
  source: string;
  publishedAt: string;
  metadata: Record<string, any>;
}

class RealDataIntegrationService {
  private static instance: RealDataIntegrationService;
  private dataSources: Map<string, DataSource> = new Map();
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map();

  static getInstance(): RealDataIntegrationService {
    if (!RealDataIntegrationService.instance) {
      RealDataIntegrationService.instance = new RealDataIntegrationService();
    }
    return RealDataIntegrationService.instance;
  }

  async initialize(): Promise<void> {
    // Initialize with free data sources
    const defaultSources: DataSource[] = [
      {
        id: 'hackernews',
        name: 'Hacker News',
        type: 'news',
        endpoint: 'https://hacker-news.firebaseio.com/v0',
        isActive: true,
      },
      {
        id: 'reddit_programming',
        name: 'Reddit Programming',
        type: 'content',
        endpoint: 'https://www.reddit.com/r/programming.json',
        isActive: true,
      },
      {
        id: 'github_trending',
        name: 'GitHub Trending',
        type: 'apps',
        endpoint: 'https://api.github.com/search/repositories',
        params: { q: 'created:>2024-01-01', sort: 'stars', order: 'desc' },
        isActive: true,
      },
      {
        id: 'dev_to',
        name: 'Dev.to Articles',
        type: 'educational',
        endpoint: 'https://dev.to/api/articles',
        params: { per_page: 50, top: 7 },
        isActive: true,
      },
      {
        id: 'freecodecamp',
        name: 'FreeCodeCamp News',
        type: 'educational',
        endpoint: 'https://www.freecodecamp.org/news/ghost/api/v3/content/posts',
        params: { key: 'public', limit: 50 },
        isActive: true,
      },
    ];

    // Load saved sources or use defaults
    const savedSources = await storageService.getData<DataSource[]>('data_sources');
    const sources = savedSources || defaultSources;

    sources.forEach(source => {
      this.dataSources.set(source.id, source);
    });
  }

  // Fetch content from Hacker News
  async fetchHackerNews(limit: number = 30): Promise<ExternalContent[]> {
    const cacheKey = `hackernews_${limit}`;
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    try {
      // Get top stories
      const topStoriesResponse = await fetch('https://hacker-news.firebaseio.com/v0/topstories.json');
      const topStories = await topStoriesResponse.json();
      
      // Get details for first N stories
      const storyPromises = topStories.slice(0, limit).map(async (id: number) => {
        const response = await fetch(`https://hacker-news.firebaseio.com/v0/item/${id}.json`);
        return response.json();
      });

      const stories = await Promise.all(storyPromises);
      
      const content: ExternalContent[] = stories
        .filter(story => story.title && story.url)
        .map(story => ({
          id: `hn_${story.id}`,
          title: story.title,
          description: story.text || '',
          url: story.url,
          category: 'technology',
          source: 'Hacker News',
          publishedAt: new Date(story.time * 1000).toISOString(),
          metadata: {
            score: story.score,
            comments: story.descendants || 0,
            author: story.by,
          },
        }));

      this.setCache(cacheKey, content, 600000); // 10 minutes
      return content;
    } catch (error) {
      console.error('Failed to fetch Hacker News:', error);
      return [];
    }
  }

  // Fetch content from Reddit
  async fetchRedditContent(subreddit: string = 'programming', limit: number = 25): Promise<ExternalContent[]> {
    const cacheKey = `reddit_${subreddit}_${limit}`;
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    try {
      // Use CORS proxy for Reddit API
      const response = await fetch(`https://api.allorigins.win/get?url=${encodeURIComponent(`https://www.reddit.com/r/${subreddit}.json?limit=${limit}`)}`);
      const proxyData = await response.json();
      const data = JSON.parse(proxyData.contents);
      
      const content: ExternalContent[] = data.data.children
        .filter((post: any) => post.data.title && !post.data.is_self)
        .map((post: any) => ({
          id: `reddit_${post.data.id}`,
          title: post.data.title,
          description: post.data.selftext || '',
          url: post.data.url,
          imageUrl: post.data.thumbnail !== 'self' ? post.data.thumbnail : undefined,
          category: subreddit,
          source: `Reddit r/${subreddit}`,
          publishedAt: new Date(post.data.created_utc * 1000).toISOString(),
          metadata: {
            score: post.data.score,
            comments: post.data.num_comments,
            author: post.data.author,
            upvoteRatio: post.data.upvote_ratio,
          },
        }));

      this.setCache(cacheKey, content, 600000); // 10 minutes
      return content;
    } catch (error) {
      console.error(`Failed to fetch Reddit ${subreddit}:`, error);
      return [];
    }
  }

  // Fetch GitHub trending repositories
  async fetchGitHubTrending(timeframe: 'daily' | 'weekly' | 'monthly' = 'daily'): Promise<ExternalContent[]> {
    const cacheKey = `github_trending_${timeframe}`;
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    try {
      const dateMap = {
        daily: 1,
        weekly: 7,
        monthly: 30,
      };
      
      const since = new Date();
      since.setDate(since.getDate() - dateMap[timeframe]);
      
      const query = `created:>${since.toISOString().split('T')[0]}`;
      const response = await fetch(
        `https://api.github.com/search/repositories?q=${encodeURIComponent(query)}&sort=stars&order=desc&per_page=30`
      );
      const data = await response.json();
      
      const content: ExternalContent[] = data.items.map((repo: any) => ({
        id: `github_${repo.id}`,
        title: repo.full_name,
        description: repo.description || '',
        url: repo.html_url,
        category: repo.language || 'Unknown',
        source: 'GitHub',
        publishedAt: repo.created_at,
        metadata: {
          stars: repo.stargazers_count,
          forks: repo.forks_count,
          language: repo.language,
          license: repo.license?.name,
          size: repo.size,
        },
      }));

      this.setCache(cacheKey, content, 1800000); // 30 minutes
      return content;
    } catch (error) {
      console.error('Failed to fetch GitHub trending:', error);
      return [];
    }
  }

  // Fetch Dev.to articles
  async fetchDevToArticles(tag?: string, limit: number = 30): Promise<ExternalContent[]> {
    const cacheKey = `devto_${tag || 'all'}_${limit}`;
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    try {
      let url = `https://dev.to/api/articles?per_page=${limit}&top=7`;
      if (tag) url += `&tag=${tag}`;

      const response = await fetch(url);
      const articles = await response.json();
      
      const content: ExternalContent[] = articles.map((article: any) => ({
        id: `devto_${article.id}`,
        title: article.title,
        description: article.description || '',
        url: article.url,
        imageUrl: article.cover_image,
        category: article.tags?.[0] || 'development',
        source: 'Dev.to',
        publishedAt: article.published_at,
        metadata: {
          reactions: article.public_reactions_count,
          comments: article.comments_count,
          author: article.user.name,
          tags: article.tags,
          readingTime: article.reading_time_minutes,
        },
      }));

      this.setCache(cacheKey, content, 1800000); // 30 minutes
      return content;
    } catch (error) {
      console.error('Failed to fetch Dev.to articles:', error);
      return [];
    }
  }

  // Aggregate content from all sources
  async fetchAggregatedContent(): Promise<ExternalContent[]> {
    const cacheKey = 'aggregated_content';
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    try {
      const [hackerNews, reddit, github, devto] = await Promise.all([
        this.fetchHackerNews(15),
        this.fetchRedditContent('programming', 15),
        this.fetchGitHubTrending('weekly'),
        this.fetchDevToArticles(undefined, 20),
      ]);

      const allContent = [...hackerNews, ...reddit, ...github, ...devto];
      
      // Sort by publication date (most recent first)
      allContent.sort((a, b) => new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime());

      this.setCache(cacheKey, allContent, 600000); // 10 minutes
      return allContent;
    } catch (error) {
      console.error('Failed to fetch aggregated content:', error);
      return [];
    }
  }

  // RSS feed parser for additional sources
  async fetchRSSFeed(url: string): Promise<ExternalContent[]> {
    const cacheKey = `rss_${url}`;
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    try {
      // Use RSS2JSON service for RSS parsing
      const proxyUrl = `https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(url)}`;
      const response = await fetch(proxyUrl);
      const data = await response.json();
      
      if (data.status !== 'ok') {
        throw new Error(`RSS parsing failed: ${data.message}`);
      }

      const content: ExternalContent[] = data.items.map((item: any, index: number) => ({
        id: `rss_${url}_${index}`,
        title: item.title,
        description: item.description?.replace(/<[^>]*>/g, '').slice(0, 200) || '',
        content: item.content,
        url: item.link,
        imageUrl: item.thumbnail,
        category: 'rss',
        source: data.feed.title || 'RSS Feed',
        publishedAt: item.pubDate,
        metadata: {
          author: item.author,
          categories: item.categories,
        },
      }));

      this.setCache(cacheKey, content, 1800000); // 30 minutes
      return content;
    } catch (error) {
      console.error(`Failed to fetch RSS feed ${url}:`, error);
      return [];
    }
  }

  // Cache management
  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < cached.ttl) {
      return cached.data;
    }
    this.cache.delete(key);
    return null;
  }

  private setCache(key: string, data: any, ttl: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  clearCache(): void {
    this.cache.clear();
  }

  // Add custom data source
  async addDataSource(source: DataSource): Promise<void> {
    this.dataSources.set(source.id, source);
    await this.saveDataSources();
  }

  // Remove data source
  async removeDataSource(sourceId: string): Promise<void> {
    this.dataSources.delete(sourceId);
    await this.saveDataSources();
  }

  private async saveDataSources(): Promise<void> {
    const sources = Array.from(this.dataSources.values());
    await storageService.setData('data_sources', sources);
  }

  getDataSources(): DataSource[] {
    return Array.from(this.dataSources.values());
  }
}

export const realDataIntegration = RealDataIntegrationService.getInstance();