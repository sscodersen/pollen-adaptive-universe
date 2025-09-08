const { spawn } = require('child_process');
const http = require('http');

// Function to check if a port is available
function checkPort(port, callback) {
  const server = http.createServer();
  server.listen(port, (err) => {
    if (err) {
      callback(false);
    } else {
      server.once('close', () => callback(true));
      server.close();
    }
  });
  server.on('error', () => callback(false));
}

// Function to wait for a port to be ready
function waitForPort(port, timeout = 30000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    
    function check() {
      const server = http.createServer();
      server.listen(port, (err) => {
        if (!err) {
          server.close();
          resolve();
          return;
        }
        
        if (Date.now() - start < timeout) {
          setTimeout(check, 1000);
        } else {
          reject(new Error(`Port ${port} not ready after ${timeout}ms`));
        }
      });
      server.on('error', () => {
        if (Date.now() - start < timeout) {
          setTimeout(check, 1000);
        } else {
          reject(new Error(`Port ${port} not ready after ${timeout}ms`));
        }
      });
    }
    
    check();
  });
}

async function startApp() {
  console.log('Starting Pollen Adaptive Universe...');
  
  // Start the backend server first
  console.log('Starting Pollen AI backend server on port 3001...');
  const backendProcess = spawn('node', ['local-backend.cjs'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env }
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[Backend]: ${data}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[Backend Error]: ${data}`);
  });
  
  // Start the Vite dev server
  console.log('Starting Vite dev server on port 8080...');
  const viteProcess = spawn('npm', ['run', 'dev'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env }
  });

  viteProcess.stdout.on('data', (data) => {
    console.log(`[Vite]: ${data}`);
  });

  viteProcess.stderr.on('data', (data) => {
    console.error(`[Vite Error]: ${data}`);
  });

  // Wait for both servers to be ready
  try {
    console.log('Waiting for backend server to start...');
    await waitForPort(3001, 30000);
    console.log('Backend server is ready!');
    
    console.log('Waiting for Vite server to start...');
    await waitForPort(8080, 60000);
    console.log('Vite server is ready!');
    
    // Start the proxy server
    console.log('Starting proxy server...');
    require('./proxy-server.cjs');
    
  } catch (error) {
    console.error('Failed to start application:', error);
    process.exit(1);
  }

  // Handle cleanup
  process.on('SIGINT', () => {
    console.log('Shutting down...');
    viteProcess.kill();
    backendProcess.kill();
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    console.log('Shutting down...');
    viteProcess.kill();
    backendProcess.kill();
    process.exit(0);
  });
}

startApp();