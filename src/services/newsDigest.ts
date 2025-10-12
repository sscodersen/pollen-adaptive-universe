import pollenAIUnified from './pollenAIUnified';
import { personalizationEngine } from './personalizationEngine';
import { NewsContent } from './unifiedContentEngine';

export interface NewsDigest {
  id: string;
  date: string;
  frequency: 'daily' | 'weekly';
  summaries: DigestSummary[];
  topStories: NewsContent[];
  personalizedCategories: string[];
  totalArticles: number;
  readTime: string;
}

export interface DigestSummary {
  category: string;
  summary: string;
  articleCount: number;
  topKeywords: string[];
}

export interface BookmarkedArticle extends NewsContent {
  bookmarkedAt: string;
  userCategory?: string;
  userNotes?: string;
}

class NewsDigestService {
  private readonly STORAGE_KEY = 'news_bookmarks';
  private readonly DIGEST_KEY = 'news_digest_prefs';

  async generateDigest(articles: NewsContent[], frequency: 'daily' | 'weekly'): Promise<NewsDigest> {
    const categorizedArticles = this.categorizeArticles(articles);
    const summaries: DigestSummary[] = [];

    for (const [category, categoryArticles] of Object.entries(categorizedArticles)) {
      const keywords = this.extractKeywords(categoryArticles);
      const summaryText = await this.generateCategorySummary(category, categoryArticles);
      
      summaries.push({
        category,
        summary: summaryText,
        articleCount: categoryArticles.length,
        topKeywords: keywords.slice(0, 5)
      });
    }

    const topStories = this.selectTopStories(articles, 5);

    return {
      id: `digest_${Date.now()}`,
      date: new Date().toISOString(),
      frequency,
      summaries,
      topStories,
      personalizedCategories: Object.keys(categorizedArticles),
      totalArticles: articles.length,
      readTime: `${Math.ceil(summaries.length * 2)} min`
    };
  }

  private categorizeArticles(articles: NewsContent[]): Record<string, NewsContent[]> {
    const categorized: Record<string, NewsContent[]> = {};
    
    articles.forEach(article => {
      const category = article.category || 'general';
      if (!categorized[category]) {
        categorized[category] = [];
      }
      categorized[category].push(article);
    });

    return categorized;
  }

  private extractKeywords(articles: NewsContent[]): string[] {
    const allTags: string[] = [];
    articles.forEach(article => {
      if (article.tags) {
        allTags.push(...article.tags);
      }
    });

    const tagFrequency = allTags.reduce((acc, tag) => {
      acc[tag] = (acc[tag] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(tagFrequency)
      .sort(([, a], [, b]) => b - a)
      .map(([tag]) => tag);
  }

  private async generateCategorySummary(category: string, articles: NewsContent[]): Promise<string> {
    const topArticles = articles.slice(0, 3);
    const titles = topArticles.map(a => a.title).join('; ');
    
    try {
      const response = await pollenAIUnified.generate({
        prompt: `Create a brief 2-sentence summary of the main ${category} news themes from these headlines: ${titles}`,
        mode: 'news',
        type: 'summary'
      });
      return response.content;
    } catch (error) {
      return `${category} news covers ${articles.length} important developments including ${topArticles[0].title}.`;
    }
  }

  private selectTopStories(articles: NewsContent[], count: number): NewsContent[] {
    return articles
      .sort((a, b) => {
        const scoreA = (a.views || 0) + (a.engagement || 0) * 10;
        const scoreB = (b.views || 0) + (b.engagement || 0) * 10;
        return scoreB - scoreA;
      })
      .slice(0, count);
  }

  saveBookmark(article: NewsContent, userCategory?: string, userNotes?: string): void {
    const bookmarks = this.getBookmarks();
    const bookmarked: BookmarkedArticle = {
      ...article,
      bookmarkedAt: new Date().toISOString(),
      userCategory,
      userNotes
    };

    bookmarks.push(bookmarked);
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(bookmarks));
    }

    personalizationEngine.trackBehavior({
      action: 'save',
      contentId: article.id,
      contentType: 'educational',
      metadata: { 
        category: article.category,
        userCategory,
        hasNotes: !!userNotes
      }
    });
  }

  removeBookmark(articleId: string): void {
    const bookmarks = this.getBookmarks();
    const filtered = bookmarks.filter(b => b.id !== articleId);
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(filtered));
    }
  }

  getBookmarks(): BookmarkedArticle[] {
    if (typeof window === 'undefined' || !window.localStorage) {
      return [];
    }
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  getBookmarksByCategory(category?: string): BookmarkedArticle[] {
    const bookmarks = this.getBookmarks();
    if (!category) return bookmarks;
    return bookmarks.filter(b => 
      b.category === category || b.userCategory === category
    );
  }

  getUserCategories(): string[] {
    const bookmarks = this.getBookmarks();
    const categories = new Set<string>();
    bookmarks.forEach(b => {
      if (b.userCategory) categories.add(b.userCategory);
    });
    return Array.from(categories);
  }

  async createCustomFeed(categories: string[]): Promise<NewsContent[]> {
    const bookmarks = this.getBookmarks();
    const categoryArticles = bookmarks.filter(b => 
      categories.includes(b.category) || 
      (b.userCategory && categories.includes(b.userCategory))
    );

    return categoryArticles.sort((a, b) => 
      new Date(b.bookmarkedAt).getTime() - new Date(a.bookmarkedAt).getTime()
    );
  }

  getSuggestedCategories(bookmarks: BookmarkedArticle[]): string[] {
    const categoryCounts: Record<string, number> = {};
    
    bookmarks.forEach(b => {
      const cat = b.userCategory || b.category;
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
    });

    return Object.entries(categoryCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([cat]) => cat);
  }

  saveDigestPreferences(frequency: 'daily' | 'weekly', categories: string[]): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.DIGEST_KEY, JSON.stringify({ frequency, categories }));
    }
  }

  getDigestPreferences(): { frequency: 'daily' | 'weekly'; categories: string[] } {
    if (typeof window === 'undefined' || !window.localStorage) {
      return { frequency: 'daily', categories: [] };
    }
    try {
      const stored = localStorage.getItem(this.DIGEST_KEY);
      return stored ? JSON.parse(stored) : { frequency: 'daily', categories: [] };
    } catch {
      return { frequency: 'daily', categories: [] };
    }
  }
}

export const newsDigestService = new NewsDigestService();
