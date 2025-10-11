import express from 'express';
import cors from 'cors';
import apiRouter from './api.js';
import { ChatServer } from './chatServer.js';
import { feedbackAnalyzer } from './services/feedbackAnalyzer.js';
import { feedbackStorage } from './storage.js';

const app = express();
const port = process.env.API_PORT || 3002;

app.use(cors());
app.use(express.json());

app.use('/api', apiRouter);

app.post('/api/feedback/submit-with-analysis', async (req, res) => {
  try {
    const analysis = await feedbackAnalyzer.analyzeFeedback(req.body);
    
    const feedback = await feedbackStorage.submitFeedback({
      ...req.body,
      sentiment: analysis.sentiment,
      topics: analysis.topics,
      category: analysis.category,
      priority: analysis.priority
    });
    
    res.json({ success: true, feedback, analysis });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

const chatServer = new ChatServer(8080);

app.listen(port, () => {
  console.log(`🚀 Pollen API server running on port ${port}`);
  console.log(`💬 WebSocket chat server running on port 8080`);
});

export { app, chatServer };
