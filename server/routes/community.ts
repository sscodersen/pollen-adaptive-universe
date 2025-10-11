import { Router } from 'express';
import { 
  chatStorage, 
  gamificationStorage, 
  eventsStorage, 
  feedbackStorage,
  curatedContentStorage,
  notificationStorage
} from '../storage.js';

export const communityRouter = Router();

// Chat Rooms API
communityRouter.post('/chat/rooms', async (req, res) => {
  try {
    const room = await chatStorage.createChatRoom(req.body);
    res.json(room);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/chat/rooms', async (req, res) => {
  try {
    const { roomType, communityId } = req.query;
    const rooms = await chatStorage.getChatRooms({ 
      roomType: roomType as string, 
      communityId: communityId as string 
    });
    res.json(rooms);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/chat/rooms/:roomId', async (req, res) => {
  try {
    const room = await chatStorage.getChatRoomById(req.params.roomId);
    res.json(room);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/chat/messages', async (req, res) => {
  try {
    const message = await chatStorage.createMessage(req.body);
    res.json(message);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/chat/messages/:roomId', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const messages = await chatStorage.getMessages(req.params.roomId, limit);
    res.json(messages);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/chat/join', async (req, res) => {
  try {
    const participant = await chatStorage.joinRoom(req.body);
    res.json(participant);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/chat/leave', async (req, res) => {
  try {
    const { roomId, userId } = req.body;
    await chatStorage.leaveRoom(roomId, userId);
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/chat/participants/:roomId', async (req, res) => {
  try {
    const participants = await chatStorage.getRoomParticipants(req.params.roomId);
    res.json(participants);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Gamification API
communityRouter.post('/gamification/badges', async (req, res) => {
  try {
    const badge = await gamificationStorage.createBadge(req.body);
    res.json(badge);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/gamification/badges', async (req, res) => {
  try {
    const { badgeType, category } = req.query;
    const badges = await gamificationStorage.getBadges({ 
      badgeType: badgeType as string, 
      category: category as string 
    });
    res.json(badges);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/gamification/award-badge', async (req, res) => {
  try {
    const award = await gamificationStorage.awardBadge(req.body);
    res.json(award);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/gamification/user-badges/:userId', async (req, res) => {
  try {
    const badges = await gamificationStorage.getUserBadges(req.params.userId);
    res.json(badges);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/gamification/points/:userId', async (req, res) => {
  try {
    const points = await gamificationStorage.getUserPoints(req.params.userId);
    res.json(points);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/gamification/add-points', async (req, res) => {
  try {
    const transaction = await gamificationStorage.addPoints(req.body);
    res.json(transaction);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/gamification/leaderboard/:leaderboardId', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const leaderboard = await gamificationStorage.getLeaderboard(req.params.leaderboardId, limit);
    res.json(leaderboard);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Events API
communityRouter.post('/events', async (req, res) => {
  try {
    const event = await eventsStorage.createEvent(req.body);
    res.json(event);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/events', async (req, res) => {
  try {
    const { eventType, category, status } = req.query;
    const events = await eventsStorage.getEvents({ 
      eventType: eventType as string, 
      category: category as string,
      status: status as string
    });
    res.json(events);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/events/:eventId', async (req, res) => {
  try {
    const event = await eventsStorage.getEventById(req.params.eventId);
    res.json(event);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/events/register', async (req, res) => {
  try {
    const registration = await eventsStorage.registerForEvent(req.body);
    res.json(registration);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/events/:eventId/registrations', async (req, res) => {
  try {
    const registrations = await eventsStorage.getEventRegistrations(req.params.eventId);
    res.json(registrations);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/events/user/:userId/registrations', async (req, res) => {
  try {
    const registrations = await eventsStorage.getUserRegistrations(req.params.userId);
    res.json(registrations);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/events/speakers', async (req, res) => {
  try {
    const speaker = await eventsStorage.addSpeaker(req.body);
    res.json(speaker);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/events/:eventId/speakers', async (req, res) => {
  try {
    const speakers = await eventsStorage.getEventSpeakers(req.params.eventId);
    res.json(speakers);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Feedback API
communityRouter.post('/feedback', async (req, res) => {
  try {
    const feedback = await feedbackStorage.submitFeedback(req.body);
    res.json(feedback);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/feedback', async (req, res) => {
  try {
    const { feedbackType, category, status, sentiment } = req.query;
    const feedback = await feedbackStorage.getFeedback({ 
      feedbackType: feedbackType as string,
      category: category as string,
      status: status as string,
      sentiment: sentiment as string
    });
    res.json(feedback);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/feedback/:feedbackId', async (req, res) => {
  try {
    const feedback = await feedbackStorage.getFeedbackById(req.params.feedbackId);
    res.json(feedback);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/feedback/respond', async (req, res) => {
  try {
    const response = await feedbackStorage.respondToFeedback(req.body);
    res.json(response);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/feedback/:feedbackId/responses', async (req, res) => {
  try {
    const responses = await feedbackStorage.getFeedbackResponses(req.params.feedbackId);
    res.json(responses);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.put('/feedback/:feedbackId/status', async (req, res) => {
  try {
    await feedbackStorage.updateFeedbackStatus(req.params.feedbackId, req.body.status);
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/feedback/:feedbackId/vote', async (req, res) => {
  try {
    const increment = req.body.increment || 1;
    await feedbackStorage.voteFeedback(req.params.feedbackId, increment);
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Curated Content API
communityRouter.post('/curated/sections', async (req, res) => {
  try {
    const section = await curatedContentStorage.createContentSection(req.body);
    res.json(section);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/curated/sections', async (req, res) => {
  try {
    const { sectionType, isActive } = req.query;
    const sections = await curatedContentStorage.getContentSections({ 
      sectionType: sectionType as string,
      isActive: isActive === 'true' ? true : isActive === 'false' ? false : undefined
    });
    res.json(sections);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/curated/content', async (req, res) => {
  try {
    const content = await curatedContentStorage.addCuratedContent(req.body);
    res.json(content);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/curated/content/:sectionId', async (req, res) => {
  try {
    const userId = req.query.userId as string;
    const content = await curatedContentStorage.getCuratedContent(req.params.sectionId, userId);
    res.json(content);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/curated/track/impression/:curatedId', async (req, res) => {
  try {
    await curatedContentStorage.trackContentImpression(req.params.curatedId);
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/curated/track/click/:curatedId', async (req, res) => {
  try {
    await curatedContentStorage.trackContentClick(req.params.curatedId);
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.post('/curated/recommendations', async (req, res) => {
  try {
    const recommendation = await curatedContentStorage.addRecommendation(req.body);
    res.json(recommendation);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/curated/recommendations/:userId', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 20;
    const recommendations = await curatedContentStorage.getUserRecommendations(req.params.userId, limit);
    res.json(recommendations);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Notifications API
communityRouter.post('/notifications', async (req, res) => {
  try {
    const notification = await notificationStorage.createNotification(req.body);
    res.json(notification);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/notifications/:userId', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const notifications = await notificationStorage.getUserNotifications(req.params.userId, limit);
    res.json(notifications);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.put('/notifications/:notificationId/read', async (req, res) => {
  try {
    await notificationStorage.markAsRead(req.params.notificationId);
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

communityRouter.get('/notifications/:userId/unread-count', async (req, res) => {
  try {
    const count = await notificationStorage.getUnreadCount(req.params.userId);
    res.json({ count });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});
