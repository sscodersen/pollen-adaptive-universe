import type { TrendData } from './enhancedTrendEngine';

// Simple multi-source trend fetcher with no API keys. Uses CORS-friendly proxies.
// Sources: Hacker News RSS, Techmeme RSS, Google News RSS (technology), GitHub Trending (HTML), ExplodingTopics (HTML)
// Note: HTML sources are fetched via r.jina.ai proxy which allows cross-origin reads.

export interface AggregatedHeadline {
  title: string;
  source: string;
  category: string;
  snippet: string;
  link?: string;
}

const proxiedFetchText = async (url: string): Promise<string> => {
  try {
    // Prefer r.jina.ai proxy for HTML; for RSS we can fetch directly, fallback to allorigins
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed ${res.status}`);
    return await res.text();
  } catch (e) {
    // allorigins fallback
    const alt = await fetch(`https://api.allorigins.win/get?url=${encodeURIComponent(url)}`);
    const data = await alt.json().catch(() => ({} as any));
    return data?.contents || '';
  }
};

const parseRSSItems = (xml: string): { title: string; link?: string }[] => {
  const items: { title: string; link?: string }[] = [];
  // Very light XML parsing via regex (sufficient for titles)
  const itemRegex = /<item[\s\S]*?<\/item>/g;
  const titleRegex = /<title>([\s\S]*?)<\/title>/i;
  const linkRegex = /<link>([\s\S]*?)<\/link>/i;
  const matches = xml.match(itemRegex) || [];
  for (const item of matches) {
    const titleMatch = item.match(titleRegex);
    const linkMatch = item.match(linkRegex);
    const title = titleMatch ? titleMatch[1].trim() : '';
    const link = linkMatch ? linkMatch[1].trim() : undefined;
    if (title) items.push({ title, link });
  }
  return items;
};

const humanCategory = (text: string): string => {
  const t = text.toLowerCase();
  if (/(ai|ml|model|neural|llm|gemini|openai|anthropic|mistral)/.test(t)) return 'AI/ML';
  if (/(crypto|blockchain|web3|defi|bitcoin|ethereum)/.test(t)) return 'Crypto/Web3';
  if (/(security|hack|breach|vuln|cve)/.test(t)) return 'Security';
  if (/(startup|funding|ipo|acquires|deal|market)/.test(t)) return 'Business';
  if (/(react|javascript|typescript|python|rust|framework)/.test(t)) return 'Web Development';
  if (/(science|research|paper|arxiv|quantum|physics|biology)/.test(t)) return 'Science';
  return 'Technology';
};

const extractKeywords = (text: string): string[] => {
  const words = text
    .replace(/[^a-zA-Z0-9\s#]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
  const important = new Set<string>();
  for (const w of words) {
    if (w.length > 3 && /[A-Z]/.test(w[0])) important.add(w);
    if (/^#[\w-]+$/.test(w)) important.add(w.replace('#',''));
  }
  return Array.from(important).slice(0, 8);
};

const scoreFromHeuristics = (title: string, source: string): number => {
  let score = 50;
  const caps = (title.match(/[A-Z][a-z]+/g) || []).length;
  score += Math.min(20, caps * 1.5);
  const srcMul: Record<string, number> = {
    HackerNews: 1.15,
    Techmeme: 1.2,
    GoogleNews: 1.05,
    GitHub: 1.1,
    ExplodingTopics: 1.25,
  };
  score *= srcMul[source] || 1.0;
  // small randomness to avoid ties
  score += Math.random() * 15;
  return Math.min(100, score);
};

export const trendAggregator = {
  // Fetch multiple sources and unify to TrendData
  async fetchTrends(): Promise<TrendData[]> {
    const now = Date.now();
    try {
      const [hnXML, techmemeXML, gnewsXML, ghHTML, etHTML] = await Promise.all([
        proxiedFetchText('https://news.ycombinator.com/rss'),
        proxiedFetchText('https://www.techmeme.com/feed.xml'),
        proxiedFetchText('https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en'),
        proxiedFetchText('https://r.jina.ai/http://github.com/trending'),
        proxiedFetchText('https://r.jina.ai/https://explodingtopics.com/topics')
      ]);

      const trends: TrendData[] = [];

      // HN
      for (const item of parseRSSItems(hnXML).slice(0, 30)) {
        const category = humanCategory(item.title);
        const keywords = extractKeywords(item.title);
        trends.push({
          id: `agg-hn-${item.title.slice(0,40)}-${Math.random()}`,
          topic: item.title,
          score: scoreFromHeuristics(item.title, 'HackerNews'),
          sentiment: 0,
          source: 'HackerNews',
          timestamp: now,
          category,
          keywords,
          momentum: Math.random() * 100,
          reach: Math.floor(Math.random() * 20000),
          engagement: Math.floor(Math.random() * 2000),
        });
      }

      // Techmeme
      for (const item of parseRSSItems(techmemeXML).slice(0, 30)) {
        const category = humanCategory(item.title);
        const keywords = extractKeywords(item.title);
        trends.push({
          id: `agg-tm-${item.title.slice(0,40)}-${Math.random()}`,
          topic: item.title,
          score: scoreFromHeuristics(item.title, 'Techmeme'),
          sentiment: 0,
          source: 'Techmeme',
          timestamp: now,
          category,
          keywords,
          momentum: Math.random() * 100,
          reach: Math.floor(Math.random() * 30000),
          engagement: Math.floor(Math.random() * 3000),
        });
      }

      // Google News (Tech)
      for (const item of parseRSSItems(gnewsXML).slice(0, 30)) {
        const category = humanCategory(item.title);
        const keywords = extractKeywords(item.title);
        trends.push({
          id: `agg-gn-${item.title.slice(0,40)}-${Math.random()}`,
          topic: item.title,
          score: scoreFromHeuristics(item.title, 'GoogleNews'),
          sentiment: 0,
          source: 'GoogleNews',
          timestamp: now,
          category,
          keywords,
          momentum: Math.random() * 100,
          reach: Math.floor(Math.random() * 25000),
          engagement: Math.floor(Math.random() * 2500),
        });
      }

      // GitHub trending (rough extraction of repo titles)
      const ghItems = Array.from(ghHTML.matchAll(/<h2 class=\"h3 lh-condensed\">[\s\S]*?<a [^>]*>([\s\S]*?)<\/a>/g));
      for (const m of ghItems.slice(0, 20)) {
        const raw = (m[1] || '').replace(/\s+/g, ' ').trim();
        if (!raw) continue;
        const title = `GitHub: ${raw}`;
        const category = 'Web Development';
        const keywords = extractKeywords(title);
        trends.push({
          id: `agg-gh-${raw.slice(0,40)}-${Math.random()}`,
          topic: title,
          score: scoreFromHeuristics(title, 'GitHub'),
          sentiment: 0,
          source: 'GitHub',
          timestamp: now,
          category,
          keywords,
          momentum: Math.random() * 100,
          reach: Math.floor(Math.random() * 12000),
          engagement: Math.floor(Math.random() * 1200),
        });
      }

      // ExplodingTopics (extract list items headings if present)
      const etMatches = Array.from(etHTML.matchAll(/<h[23][^>]*>(.*?)<\/h[23]>/g));
      for (const m of etMatches.slice(0, 20)) {
        const title = (m[1] || '').replace(/<[^>]+>/g, '').replace(/&[^;]+;/g, ' ').trim();
        if (!title || title.length < 6) continue;
        const category = humanCategory(title);
        const keywords = extractKeywords(title);
        trends.push({
          id: `agg-et-${title.slice(0,40)}-${Math.random()}`,
          topic: title,
          score: scoreFromHeuristics(title, 'ExplodingTopics'),
          sentiment: 0,
          source: 'ExplodingTopics',
          timestamp: now,
          category,
          keywords,
          momentum: Math.random() * 100,
          reach: Math.floor(Math.random() * 18000),
          engagement: Math.floor(Math.random() * 1800),
        });
      }

      // Deduplicate by topic lowercase
      const map = new Map<string, TrendData>();
      for (const t of trends) {
        const key = t.topic.toLowerCase();
        if (!map.has(key) || (map.get(key)!.score < t.score)) {
          map.set(key, t);
        }
      }

      return Array.from(map.values()).sort((a, b) => b.score - a.score).slice(0, 80);
    } catch (e) {
      console.warn('trendAggregator.fetchTrends failed:', e);
      return [];
    }
  },

  async fetchHeadlines(): Promise<AggregatedHeadline[]> {
    const trends = await this.fetchTrends();
    return trends.slice(0, 20).map(t => ({
      title: t.topic,
      source: t.source,
      category: t.category,
      snippet: t.topic.length > 120 ? t.topic.slice(0, 117) + '...' : t.topic,
    }));
  }
};
