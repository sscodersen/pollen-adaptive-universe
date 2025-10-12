// Pollen AI Edge Computing Service Worker
// Provides offline caching and edge optimization

const CACHE_VERSION = 'pollen-ai-v3.0.0';
const CACHE_ASSETS = 'pollen-assets';
const CACHE_AI_RESPONSES = 'pollen-ai-responses';
const CACHE_DURATION = 15 * 60 * 1000; // 15 minutes

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing Pollen AI Edge Worker');
  
  event.waitUntil(
    caches.open(CACHE_ASSETS).then((cache) => {
      return cache.addAll([
        '/',
        '/index.html',
        '/manifest.json'
      ]).catch(err => {
        console.warn('[Service Worker] Failed to cache some assets:', err);
      });
    })
  );
  
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating Pollen AI Edge Worker');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name.startsWith('pollen-') && name !== CACHE_VERSION)
          .map((name) => caches.delete(name))
      );
    })
  );
  
  self.clients.claim();
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Cache AI responses from Pollen AI backend
  if (url.pathname.includes('/generate') || url.pathname.includes('/api/ai')) {
    event.respondWith(handleAIRequest(request));
    return;
  }
  
  // Cache static assets
  if (request.method === 'GET' && !url.pathname.includes('/api/')) {
    event.respondWith(handleStaticAsset(request));
    return;
  }
  
  // Default: network only
  event.respondWith(fetch(request));
});

// Handle AI requests with intelligent caching
async function handleAIRequest(request) {
  const cache = await caches.open(CACHE_AI_RESPONSES);
  
  // Create cache key for POST requests (based on body content)
  let cacheKey = request.url;
  if (request.method === 'POST') {
    const requestBody = await request.clone().text();
    const bodyHash = await hashString(requestBody);
    cacheKey = `${request.url}?body=${bodyHash}`;
  }
  
  // Try cache first
  const cacheRequest = new Request(cacheKey);
  const cached = await cache.match(cacheRequest);
  if (cached) {
    const cacheTime = new Date(cached.headers.get('sw-cache-time'));
    const age = Date.now() - cacheTime.getTime();
    
    if (age < CACHE_DURATION) {
      console.log('[Service Worker] Serving AI response from cache');
      return cached.clone();
    }
  }
  
  // Fetch from network
  try {
    const response = await fetch(request);
    
    // Cache successful responses (GET or POST)
    if (response.ok) {
      const clonedResponse = response.clone();
      const headers = new Headers(clonedResponse.headers);
      headers.set('sw-cache-time', new Date().toISOString());
      
      const cachedResponse = new Response(clonedResponse.body, {
        status: clonedResponse.status,
        statusText: clonedResponse.statusText,
        headers: headers
      });
      
      cache.put(cacheRequest, cachedResponse);
    }
    
    return response;
  } catch (error) {
    console.error('[Service Worker] Network request failed:', error);
    
    // Return cached response if available (even if stale)
    if (cached) {
      console.log('[Service Worker] Network failed, serving stale cache');
      return cached.clone();
    }
    
    // Return error response
    return new Response(
      JSON.stringify({
        error: 'Network unavailable',
        message: 'AI service is temporarily offline',
        cached: false
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

// Simple hash function for cache keys
async function hashString(str) {
  const encoder = new TextEncoder();
  const data = encoder.encode(str);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 16);
}

// Handle static assets with cache-first strategy
async function handleStaticAsset(request) {
  const cache = await caches.open(CACHE_ASSETS);
  
  // Try cache first
  const cached = await cache.match(request);
  if (cached) {
    return cached;
  }
  
  // Fetch from network and cache
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error('[Service Worker] Failed to fetch asset:', error);
    throw error;
  }
}

// Message handler for cache control
self.addEventListener('message', (event) => {
  if (event.data === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((name) => caches.delete(name))
        );
      }).then(() => {
        console.log('[Service Worker] All caches cleared');
        event.ports[0].postMessage({ success: true });
      })
    );
  }
  
  if (event.data === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

console.log('[Service Worker] Pollen AI Edge Worker loaded');
