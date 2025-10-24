import { API_BASE_URL } from '@utils/constants';

const getApiUrl = () => {
  if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
    const currentHost = window.location.hostname;
    const port = '8000';
    return `https://${currentHost}:${port}`;
  }
  return API_BASE_URL;
};

export const API = {
  shopping: {
    search: `${getApiUrl()}/api/shopping/search`,
    categories: `${getApiUrl()}/api/shopping/categories`,
  },
  travel: {
    plan: `${getApiUrl()}/api/travel/plan`,
    destinations: `${getApiUrl()}/api/travel/destinations`,
  },
  news: {
    fetch: `${getApiUrl()}/api/news/fetch`,
    categories: `${getApiUrl()}/api/news/categories`,
  },
  content: {
    generate: `${getApiUrl()}/api/content/generate`,
    types: `${getApiUrl()}/api/content/types`,
  },
  scraper: {
    search: `${getApiUrl()}/api/scraper/search`,
  },
  ai: {
    chat: `${getApiUrl()}/api/ai/chat`,
    models: `${getApiUrl()}/api/ai/models`,
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
