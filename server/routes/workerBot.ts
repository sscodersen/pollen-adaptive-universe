import express from 'express';
import { workerBot } from '../workerBot';

const router = express.Router();

// SSE endpoint for real-time updates
router.get('/stream', (req, res) => {
  const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const userId = req.query.userId as string | undefined;

  // Set headers for SSE
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering

  // Add client to SSE clients
  workerBot.addSSEClient(clientId, res, userId);

  // Handle client disconnect
  req.on('close', () => {
    workerBot.removeSSEClient(clientId);
  });
});

// Submit a new task
router.post('/tasks', (req, res) => {
  try {
    const { type, payload, priority = 5 } = req.body;

    if (!type || !payload) {
      return res.status(400).json({ error: 'Type and payload are required' });
    }

    const taskId = workerBot.addTask({ type, payload, priority });

    res.json({ 
      success: true, 
      taskId,
      message: 'Task queued successfully'
    });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Get task status
router.get('/tasks/:taskId', (req, res) => {
  const { taskId } = req.params;
  const task = workerBot.getTaskStatus(taskId);

  if (!task) {
    return res.status(404).json({ error: 'Task not found' });
  }

  res.json({ task });
});

// Get worker statistics
router.get('/stats', (req, res) => {
  const stats = workerBot.getStats();
  res.json({ stats });
});

// Quick action endpoints
router.post('/generate-content', (req, res) => {
  const { prompt, type, userId } = req.body;
  
  const taskId = workerBot.addTask({
    type: 'content',
    payload: { prompt, type, userId },
    priority: 7
  });

  res.json({ taskId, message: 'Content generation queued' });
});

router.post('/generate-music', (req, res) => {
  const { mood, genre, occasion } = req.body;
  
  const taskId = workerBot.addTask({
    type: 'music',
    payload: { mood, genre, occasion },
    priority: 6
  });

  res.json({ taskId, message: 'Music generation queued' });
});

router.post('/generate-ads', (req, res) => {
  const { targetAudience, product, goals } = req.body;
  
  const taskId = workerBot.addTask({
    type: 'ads',
    payload: { targetAudience, product, goals },
    priority: 5
  });

  res.json({ taskId, message: 'Ad generation queued' });
});

router.post('/analyze-trends', (req, res) => {
  const { data, timeRange, category } = req.body;
  
  const taskId = workerBot.addTask({
    type: 'trends',
    payload: { data, timeRange, category },
    priority: 8
  });

  res.json({ taskId, message: 'Trend analysis queued' });
});

router.post('/perform-analytics', (req, res) => {
  const { userData, metrics, insights } = req.body;
  
  const taskId = workerBot.addTask({
    type: 'analytics',
    payload: { userData, metrics, insights },
    priority: 7
  });

  res.json({ taskId, message: 'Analytics queued' });
});

router.post('/personalize-content', (req, res) => {
  const { userProfile, contentPool, preferences } = req.body;
  
  const taskId = workerBot.addTask({
    type: 'personalization',
    payload: { userProfile, contentPool, preferences },
    priority: 9
  });

  res.json({ taskId, message: 'Personalization queued' });
});

export default router;
