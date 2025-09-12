#!/usr/bin/env node

const { spawn } = require('child_process');
const http = require('http');
const httpProxy = require('http-proxy');

// Create proxy server
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

  // Route API calls to backend, everything else to frontend
  const target = req.url.startsWith('/api/') ? 
    'http://localhost:3001' : 
    'http://localhost:8080';

  proxy.web(req, res, {
    target: target,
    changeOrigin: true,
    timeout: 30000,
    proxyTimeout: 30000
  }, (error) => {
    console.error(`Proxy error for ${req.url}:`, error.message);
    if (!res.headersSent) {
      res.writeHead(500);
      res.end('Service temporarily unavailable');
    }
  });
});

// Start backend server
const backendProcess = spawn('node', ['local-backend.cjs'], {
  stdio: 'inherit',
  env: { ...process.env }
});

// Start Vite dev server
const viteProcess = spawn('npm', ['run', 'dev'], {
  stdio: 'inherit',
  env: { ...process.env }
});

// Use PORT environment variable for deployment compatibility
const PORT = process.env.PORT || 5000;

// Start proxy on the specified port
server.listen(PORT, '0.0.0.0', () => {
  console.log('🚀 Pollen Adaptive Universe is starting up...');
  console.log('   Frontend: http://localhost:8080 (Vite dev server)');
  console.log('   Backend: http://localhost:3001 (Local API server)');
  console.log(`   Proxy: http://localhost:${PORT} (Main application URL)`);
  console.log('   ✨ Application ready!');
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down services...');
  backendProcess.kill();
  viteProcess.kill();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down services...');
  backendProcess.kill();
  viteProcess.kill();
  process.exit(0);
});

// Handle WebSocket upgrades for Vite HMR
server.on('upgrade', (req, socket, head) => {
  const target = req.url.startsWith('/api/') ? 
    'http://localhost:3001' : 
    'http://localhost:8080';
  
  proxy.ws(req, socket, head, { target });
});

// Handle proxy errors
proxy.on('error', (err, req, res) => {
  console.error('Proxy error:', err.message);
  if (!res.headersSent) {
    res.writeHead(500, {
      'Content-Type': 'text/plain'
    });
    res.end('Service unavailable');
  }
});