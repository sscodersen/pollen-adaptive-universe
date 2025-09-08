const http = require('http');
const httpProxy = require('http-proxy');

// Create a proxy server with custom application logic
const proxy = httpProxy.createProxyServer({});

const server = http.createServer((req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // Route API calls to the backend server, everything else to Vite
  const target = req.url.startsWith('/api/') ? 
    'http://localhost:3001' : 
    'http://localhost:8080';

  console.log(`Proxying ${req.method} ${req.url} to ${target}`);

  proxy.web(req, res, {
    target: target,
    changeOrigin: true,
    timeout: 30000,
    proxyTimeout: 30000
  }, (error) => {
    console.error(`Proxy error for ${req.url}:`, error.message);
    if (!res.headersSent) {
      res.writeHead(500);
      res.end('Proxy error');
    }
  });
});

server.listen(5000, '0.0.0.0', () => {
  console.log('Proxy server running on port 5000, forwarding to port 8080');
});

// Handle proxy errors
proxy.on('error', (err, req, res) => {
  console.error('Proxy error:', err);
  if (!res.headersSent) {
    res.writeHead(500, {
      'Content-Type': 'text/plain'
    });
    res.end('Something went wrong. And we are reporting a custom error message.');
  }
});