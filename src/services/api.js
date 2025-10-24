export const API = {
  shopping: {
    search: '/api/shopping/search',
    categories: '/api/shopping/categories',
  },
  travel: {
    plan: '/api/travel/plan',
    destinations: '/api/travel/destinations',
  },
  news: {
    fetch: '/api/news/fetch',
    categories: '/api/news/categories',
  },
  content: {
    generate: '/api/content/generate',
    types: '/api/content/types',
  },
  scraper: {
    search: '/api/scraper/search',
  },
  ai: {
    chat: '/api/ai/chat',
    models: '/api/ai/models',
  },
};

export const fetchAPI = async (url, options = {}) => {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
};
