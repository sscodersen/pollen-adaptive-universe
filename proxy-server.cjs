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

  // Proxy to Vite dev server on port 8080
  proxy.web(req, res, {
    target: 'http://localhost:8080',
    changeOrigin: true,
    timeout: 30000,
    proxyTimeout: 30000
  }, (error) => {
    console.error('Proxy error:', error);
    res.writeHead(500);
    res.end('Proxy error');
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