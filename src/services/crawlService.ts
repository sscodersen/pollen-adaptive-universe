// Service to integrate with external FastAPI crawler (scrape/extract/stream)
// Uses SSE for real-time streaming of scraped content

export type CrawlServiceConfig = {
  baseUrl: string; // e.g. http://localhost:8000
};

export type StreamHandlers = {
  onChunk?: (text: string) => void;
  onError?: (err: any) => void;
  onDone?: () => void;
  // Optional rate limit in ms between chunks flushed to UI
  throttleMs?: number;
};

class CrawlService {
  private baseUrl: string | null = null;

  setEndpoint(url: string) {
    this.baseUrl = url.replace(/\/$/, '');
  }

  private requireBase(): string {
    if (!this.baseUrl) throw new Error('CrawlService endpoint not set. Call setEndpoint(baseUrl).');
    return this.baseUrl;
  }

  async scrape(url: string): Promise<string> {
    const base = this.requireBase();
    const res = await fetch(`${base}/scrape?url=${encodeURIComponent(url)}`);
    if (!res.ok) throw new Error(`Scrape failed: ${res.status}`);
    // Backend returns plain text (markdown). Try text fallback.
    const ct = res.headers.get('content-type') || '';
    return ct.includes('application/json') ? JSON.stringify(await res.json()) : await res.text();
  }

  async extract(url: string, query: string, contextSize = 300): Promise<string> {
    const base = this.requireBase();
    const res = await fetch(`${base}/extract?url=${encodeURIComponent(url)}&query=${encodeURIComponent(query)}&context_size=${contextSize}`);
    if (!res.ok) throw new Error(`Extract failed: ${res.status}`);
    const ct = res.headers.get('content-type') || '';
    return ct.includes('application/json') ? JSON.stringify(await res.json()) : await res.text();
  }

  async smartExtract(url: string, instruction: string): Promise<string> {
    const base = this.requireBase();
    const res = await fetch(`${base}/smart_extract?url=${encodeURIComponent(url)}&instruction=${encodeURIComponent(instruction)}`);
    if (!res.ok) throw new Error(`Smart extract failed: ${res.status}`);
    const ct = res.headers.get('content-type') || '';
    return ct.includes('application/json') ? JSON.stringify(await res.json()) : await res.text();
  }

  // Returns a function to stop the stream
  stream(url: string, handlers: StreamHandlers = {}): () => void {
    const base = this.requireBase();
    const streamUrl = `${base}/stream?url=${encodeURIComponent(url)}`;

    const throttleMs = handlers.throttleMs ?? 250;
    let buffer = '';
    let lastFlush = 0;
    let closed = false;

    const flush = () => {
      if (!buffer) return;
      handlers.onChunk?.(buffer);
      buffer = '';
      lastFlush = Date.now();
    };

    const es = new EventSource(streamUrl);

    es.onmessage = (ev) => {
      if (closed) return;
      try {
        const now = Date.now();
        buffer += (ev.data || '') + '\n\n';
        if (now - lastFlush >= throttleMs) flush();
      } catch (err) {
        handlers.onError?.(err);
      }
    };

    es.onerror = (err) => {
      handlers.onError?.(err);
      // Auto-close on error; caller can decide to retry
      es.close();
      handlers.onDone?.();
    };

    es.onopen = () => {
      // no-op
    };

    const stop = () => {
      closed = true;
      flush();
      es.close();
      handlers.onDone?.();
    };

    return stop;
  }
}

export const crawlService = new CrawlService();
