import { Router } from 'express';
import { healthResearchStorage, forumStorage } from './storage.js';
import { communityRouter } from './routes/community.js';

const router = Router();

router.use('/community', communityRouter);

router.post('/health-research/data', async (req, res) => {
  try {
    const { userId, dataType, category, metrics, demographics, tags, isPublic } = req.body;
    
    if (!userId || !dataType || !category || !metrics) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await healthResearchStorage.submitHealthData({
      userId,
      dataType,
      category,
      metrics,
      demographics,
      tags,
      isPublic: isPublic ?? true
    });
    
    res.json({ success: true, data: result });
  } catch (error) {
    console.error('Error submitting health data:', error);
    res.status(500).json({ error: 'Failed to submit health data' });
  }
});

router.get('/health-research/data', async (req, res) => {
  try {
    const { dataType, category, isPublic } = req.query;
    const filters: any = {};
    
    if (dataType) filters.dataType = dataType as string;
    if (category) filters.category = category as string;
    if (isPublic !== undefined) filters.isPublic = isPublic === 'true';
    
    const data = await healthResearchStorage.getHealthData(filters);
    res.json({ success: true, data });
  } catch (error) {
    console.error('Error fetching health data:', error);
    res.status(500).json({ error: 'Failed to fetch health data' });
  }
});

router.post('/health-research/journeys', async (req, res) => {
  try {
    const { userId, journeyType, startDate, endDate, milestones, outcomes, challenges, insights, isActive, isPublic } = req.body;
    
    if (!userId || !journeyType || !startDate) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await healthResearchStorage.submitWellnessJourney({
      userId,
      journeyType,
      startDate: new Date(startDate),
      endDate: endDate ? new Date(endDate) : undefined,
      milestones,
      outcomes,
      challenges,
      insights,
      isActive: isActive ?? true,
      isPublic: isPublic ?? true
    });
    
    res.json({ success: true, journey: result });
  } catch (error) {
    console.error('Error submitting wellness journey:', error);
    res.status(500).json({ error: 'Failed to submit wellness journey' });
  }
});

router.get('/health-research/journeys', async (req, res) => {
  try {
    const { journeyType, isActive, isPublic } = req.query;
    const filters: any = {};
    
    if (journeyType) filters.journeyType = journeyType as string;
    if (isActive !== undefined) filters.isActive = isActive === 'true';
    if (isPublic !== undefined) filters.isPublic = isPublic === 'true';
    
    const journeys = await healthResearchStorage.getWellnessJourneys(filters);
    res.json({ success: true, journeys });
  } catch (error) {
    console.error('Error fetching wellness journeys:', error);
    res.status(500).json({ error: 'Failed to fetch wellness journeys' });
  }
});

router.post('/health-research/insights', async (req, res) => {
  try {
    const { insightType, category, title, description, dataPoints, confidence, significance, visualizationData, metadata } = req.body;
    
    if (!insightType || !category || !title || !description) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await healthResearchStorage.createHealthInsight({
      insightType,
      category,
      title,
      description,
      dataPoints,
      confidence,
      significance,
      visualizationData,
      metadata
    });
    
    res.json({ success: true, insight: result });
  } catch (error) {
    console.error('Error creating health insight:', error);
    res.status(500).json({ error: 'Failed to create health insight' });
  }
});

router.get('/health-research/insights', async (req, res) => {
  try {
    const { insightType, category } = req.query;
    const filters: any = {};
    
    if (insightType) filters.insightType = insightType as string;
    if (category) filters.category = category as string;
    
    const insights = await healthResearchStorage.getHealthInsights(filters);
    res.json({ success: true, insights });
  } catch (error) {
    console.error('Error fetching health insights:', error);
    res.status(500).json({ error: 'Failed to fetch health insights' });
  }
});

router.post('/health-research/findings', async (req, res) => {
  try {
    const { title, summary, fullReport, findingType, impactScore, datasetSize, categories, keyMetrics, visualizations, citations, status } = req.body;
    
    if (!title || !summary || !findingType) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await healthResearchStorage.createResearchFinding({
      title,
      summary,
      fullReport,
      findingType,
      impactScore,
      datasetSize,
      categories,
      keyMetrics,
      visualizations,
      citations,
      status: status || 'draft'
    });
    
    res.json({ success: true, finding: result });
  } catch (error) {
    console.error('Error creating research finding:', error);
    res.status(500).json({ error: 'Failed to create research finding' });
  }
});

router.get('/health-research/findings', async (req, res) => {
  try {
    const { findingType, status } = req.query;
    const filters: any = {};
    
    if (findingType) filters.findingType = findingType as string;
    if (status) filters.status = status as string;
    
    const findings = await healthResearchStorage.getResearchFindings(filters);
    res.json({ success: true, findings });
  } catch (error) {
    console.error('Error fetching research findings:', error);
    res.status(500).json({ error: 'Failed to fetch research findings' });
  }
});

router.post('/forum/topics', async (req, res) => {
  try {
    const { creatorId, title, description, category, tags } = req.body;
    
    if (!creatorId || !title || !description || !category) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await forumStorage.createTopic({
      creatorId,
      title,
      description,
      category,
      tags
    });
    
    res.json({ success: true, topic: result });
  } catch (error) {
    console.error('Error creating forum topic:', error);
    res.status(500).json({ error: 'Failed to create forum topic' });
  }
});

router.get('/forum/topics', async (req, res) => {
  try {
    const { category, status } = req.query;
    const filters: any = {};
    
    if (category) filters.category = category as string;
    if (status) filters.status = status as string;
    
    const topics = await forumStorage.getTopics(filters);
    res.json({ success: true, topics });
  } catch (error) {
    console.error('Error fetching forum topics:', error);
    res.status(500).json({ error: 'Failed to fetch forum topics' });
  }
});

router.get('/forum/topics/:topicId', async (req, res) => {
  try {
    const { topicId } = req.params;
    const topic = await forumStorage.getTopicById(topicId);
    
    if (!topic) {
      return res.status(404).json({ error: 'Topic not found' });
    }
    
    await forumStorage.incrementTopicViews(topicId);
    res.json({ success: true, topic });
  } catch (error) {
    console.error('Error fetching forum topic:', error);
    res.status(500).json({ error: 'Failed to fetch forum topic' });
  }
});

router.post('/forum/topics/:topicId/posts', async (req, res) => {
  try {
    const { topicId } = req.params;
    const { userId, content, parentPostId, postType, isExpertPost, metadata } = req.body;
    
    if (!userId || !content) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await forumStorage.createPost({
      topicId,
      userId,
      content,
      parentPostId,
      postType: postType || 'reply',
      isExpertPost: isExpertPost || false,
      metadata
    });
    
    res.json({ success: true, post: result });
  } catch (error) {
    console.error('Error creating forum post:', error);
    res.status(500).json({ error: 'Failed to create forum post' });
  }
});

router.get('/forum/topics/:topicId/posts', async (req, res) => {
  try {
    const { topicId } = req.params;
    const posts = await forumStorage.getPostsByTopic(topicId);
    res.json({ success: true, posts });
  } catch (error) {
    console.error('Error fetching forum posts:', error);
    res.status(500).json({ error: 'Failed to fetch forum posts' });
  }
});

router.post('/forum/vote', async (req, res) => {
  try {
    const { userId, targetType, targetId, voteType } = req.body;
    
    if (!userId || !targetType || !targetId || !voteType) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await forumStorage.voteOnTarget({
      userId,
      targetType,
      targetId,
      voteType
    });
    
    res.json({ success: true, ...result });
  } catch (error) {
    console.error('Error processing vote:', error);
    res.status(500).json({ error: 'Failed to process vote' });
  }
});

router.post('/forum/guidelines', async (req, res) => {
  try {
    const { title, content, category, version, metadata } = req.body;
    
    if (!title || !content || !category || !version) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await forumStorage.createGuideline({
      title,
      content,
      category,
      version,
      metadata
    });
    
    res.json({ success: true, guideline: result });
  } catch (error) {
    console.error('Error creating guideline:', error);
    res.status(500).json({ error: 'Failed to create guideline' });
  }
});

router.get('/forum/guidelines', async (req, res) => {
  try {
    const { category, approvalStatus } = req.query;
    const filters: any = {};
    
    if (category) filters.category = category as string;
    if (approvalStatus) filters.approvalStatus = approvalStatus as string;
    
    const guidelines = await forumStorage.getGuidelines(filters);
    res.json({ success: true, guidelines });
  } catch (error) {
    console.error('Error fetching guidelines:', error);
    res.status(500).json({ error: 'Failed to fetch guidelines' });
  }
});

router.post('/forum/expert-contributions', async (req, res) => {
  try {
    const { expertId, contributionType, relatedTopicId, relatedGuidelineId, content, expertise, citations, impactScore } = req.body;
    
    if (!expertId || !contributionType || !content) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await forumStorage.createExpertContribution({
      expertId,
      contributionType,
      relatedTopicId,
      relatedGuidelineId,
      content,
      expertise,
      citations,
      impactScore
    });
    
    res.json({ success: true, contribution: result });
  } catch (error) {
    console.error('Error creating expert contribution:', error);
    res.status(500).json({ error: 'Failed to create expert contribution' });
  }
});

router.get('/forum/expert-contributions', async (req, res) => {
  try {
    const { expertId, contributionType } = req.query;
    const filters: any = {};
    
    if (expertId) filters.expertId = expertId as string;
    if (contributionType) filters.contributionType = contributionType as string;
    
    const contributions = await forumStorage.getExpertContributions(filters);
    res.json({ success: true, contributions });
  } catch (error) {
    console.error('Error fetching expert contributions:', error);
    res.status(500).json({ error: 'Failed to fetch expert contributions' });
  }
});

router.post('/forum/moderation', async (req, res) => {
  try {
    const { moderatorId, targetType, targetId, actionType, reason, automated, metadata } = req.body;
    
    if (!moderatorId || !targetType || !targetId || !actionType || !reason) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const result = await forumStorage.createModerationAction({
      moderatorId,
      targetType,
      targetId,
      actionType,
      reason,
      automated: automated || false,
      metadata
    });
    
    res.json({ success: true, action: result });
  } catch (error) {
    console.error('Error creating moderation action:', error);
    res.status(500).json({ error: 'Failed to create moderation action' });
  }
});

router.get('/forum/moderation', async (req, res) => {
  try {
    const { targetType, actionType } = req.query;
    const filters: any = {};
    
    if (targetType) filters.targetType = targetType as string;
    if (actionType) filters.actionType = actionType as string;
    
    const actions = await forumStorage.getModerationActions(filters);
    res.json({ success: true, actions });
  } catch (error) {
    console.error('Error fetching moderation actions:', error);
    res.status(500).json({ error: 'Failed to fetch moderation actions' });
  }
});

export default router;
