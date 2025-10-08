const crypto = require('crypto');

const HASH_SALT = process.env.HASH_SALT || 'pollen-health-research-salt-2024';

class HealthResearchStorage {
  constructor(db) {
    this.db = db;
    this.initSchema();
  }

  async initSchema() {
    if (!this.schema) {
      this.schema = await import('../shared/schema.js');
    }
  }

  async submitHealthData(data) {
    await this.initSchema();
    const anonymousUserId = this.hashUserIdWithSalt(data.userId);
    const dataId = `health_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const { userId, ...healthDataWithoutUserId } = data;
    
    const [result] = await this.db.insert(this.schema.healthData).values({
      ...healthDataWithoutUserId,
      dataId,
      anonymousUserId
    }).returning();
    
    return result;
  }

  async getHealthData(filters) {
    await this.initSchema();
    const conditions = [];
    
    if (filters?.dataType) {
      conditions.push(this.schema.healthData.dataType.eq(filters.dataType));
    }
    if (filters?.category) {
      conditions.push(this.schema.healthData.category.eq(filters.category));
    }
    if (filters?.isPublic !== undefined) {
      conditions.push(this.schema.healthData.isPublic.eq(filters.isPublic));
    }
    
    let query = this.db.select().from(this.schema.healthData);
    
    if (conditions.length > 0) {
      const { and } = require('drizzle-orm');
      query = query.where(and(...conditions));
    }
    
    const { desc } = require('drizzle-orm');
    return await query.orderBy(desc(this.schema.healthData.createdAt));
  }

  async submitWellnessJourney(journey) {
    await this.initSchema();
    const anonymousUserId = this.hashUserIdWithSalt(journey.userId);
    const journeyId = `journey_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const { userId, ...journeyWithoutUserId } = journey;
    
    const [result] = await this.db.insert(this.schema.wellnessJourneys).values({
      ...journeyWithoutUserId,
      journeyId,
      anonymousUserId
    }).returning();
    
    return result;
  }

  async getWellnessJourneys(filters) {
    await this.initSchema();
    const conditions = [];
    
    if (filters?.journeyType) {
      conditions.push(this.schema.wellnessJourneys.journeyType.eq(filters.journeyType));
    }
    if (filters?.isActive !== undefined) {
      conditions.push(this.schema.wellnessJourneys.isActive.eq(filters.isActive));
    }
    if (filters?.isPublic !== undefined) {
      conditions.push(this.schema.wellnessJourneys.isPublic.eq(filters.isPublic));
    }
    
    let query = this.db.select().from(this.schema.wellnessJourneys);
    
    if (conditions.length > 0) {
      const { and } = require('drizzle-orm');
      query = query.where(and(...conditions));
    }
    
    const { desc } = require('drizzle-orm');
    return await query.orderBy(desc(this.schema.wellnessJourneys.createdAt));
  }

  async createHealthInsight(insight) {
    await this.initSchema();
    const insightId = `insight_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await this.db.insert(this.schema.healthInsights).values({
      ...insight,
      insightId
    }).returning();
    
    return result;
  }

  async getHealthInsights(filters) {
    await this.initSchema();
    const conditions = [];
    
    if (filters?.insightType) {
      conditions.push(this.schema.healthInsights.insightType.eq(filters.insightType));
    }
    if (filters?.category) {
      conditions.push(this.schema.healthInsights.category.eq(filters.category));
    }
    
    let query = this.db.select().from(this.schema.healthInsights);
    
    if (conditions.length > 0) {
      const { and } = require('drizzle-orm');
      query = query.where(and(...conditions));
    }
    
    const { desc } = require('drizzle-orm');
    return await query.orderBy(desc(this.schema.healthInsights.createdAt));
  }

  async createResearchFinding(finding) {
    await this.initSchema();
    const findingId = `finding_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await this.db.insert(this.schema.researchFindings).values({
      ...finding,
      findingId
    }).returning();
    
    return result;
  }

  async getResearchFindings(filters) {
    await this.initSchema();
    const conditions = [];
    
    if (filters?.findingType) {
      conditions.push(this.schema.researchFindings.findingType.eq(filters.findingType));
    }
    if (filters?.status) {
      conditions.push(this.schema.researchFindings.status.eq(filters.status));
    }
    
    let query = this.db.select().from(this.schema.researchFindings);
    
    if (conditions.length > 0) {
      const { and } = require('drizzle-orm');
      query = query.where(and(...conditions));
    }
    
    const { desc } = require('drizzle-orm');
    return await query.orderBy(desc(this.schema.researchFindings.createdAt));
  }

  hashUserIdWithSalt(userId) {
    return crypto.createHmac('sha256', HASH_SALT).update(userId).digest('hex');
  }
}

class ForumStorage {
  constructor(db) {
    this.db = db;
    this.initSchema();
  }

  async initSchema() {
    if (!this.schema) {
      this.schema = await import('../shared/schema.js');
    }
  }

  async createTopic(topic) {
    await this.initSchema();
    const topicId = `topic_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await this.db.insert(this.schema.forumTopics).values({
      ...topic,
      topicId
    }).returning();
    
    return result;
  }

  async getTopics(filters) {
    await this.initSchema();
    const conditions = [];
    
    if (filters?.category) {
      conditions.push(this.schema.forumTopics.category.eq(filters.category));
    }
    if (filters?.status) {
      conditions.push(this.schema.forumTopics.status.eq(filters.status));
    }
    
    let query = this.db.select().from(this.schema.forumTopics);
    
    if (conditions.length > 0) {
      const { and } = require('drizzle-orm');
      query = query.where(and(...conditions));
    }
    
    const { desc } = require('drizzle-orm');
    return await query.orderBy(desc(this.schema.forumTopics.isPinned), desc(this.schema.forumTopics.createdAt));
  }

  async getTopicById(topicId) {
    await this.initSchema();
    const { eq } = require('drizzle-orm');
    const [result] = await this.db.select().from(this.schema.forumTopics).where(eq(this.schema.forumTopics.topicId, topicId));
    return result;
  }

  async incrementTopicViews(topicId) {
    await this.initSchema();
    const { eq, sql } = require('drizzle-orm');
    await this.db.update(this.schema.forumTopics)
      .set({ viewCount: sql`${this.schema.forumTopics.viewCount} + 1` })
      .where(eq(this.schema.forumTopics.topicId, topicId));
  }

  async createPost(post) {
    await this.initSchema();
    const postId = `post_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await this.db.insert(this.schema.forumPosts).values({
      ...post,
      postId
    }).returning();
    
    const { eq, sql } = require('drizzle-orm');
    await this.db.update(this.schema.forumTopics)
      .set({ postCount: sql`${this.schema.forumTopics.postCount} + 1` })
      .where(eq(this.schema.forumTopics.topicId, post.topicId));
    
    return result;
  }

  async getPostsByTopic(topicId) {
    await this.initSchema();
    const { eq, desc } = require('drizzle-orm');
    return await this.db.select().from(this.schema.forumPosts)
      .where(eq(this.schema.forumPosts.topicId, topicId))
      .orderBy(desc(this.schema.forumPosts.createdAt));
  }

  async voteOnTarget(vote) {
    await this.initSchema();
    const { and, eq } = require('drizzle-orm');
    const existingVote = await this.db.select().from(this.schema.forumVotes)
      .where(
        and(
          eq(this.schema.forumVotes.userId, vote.userId),
          eq(this.schema.forumVotes.targetId, vote.targetId)
        )
      );
    
    if (existingVote.length > 0) {
      await this.db.delete(this.schema.forumVotes).where(eq(this.schema.forumVotes.voteId, existingVote[0].voteId));
      
      if (existingVote[0].voteType === vote.voteType) {
        return { action: 'removed', vote: existingVote[0] };
      }
    }
    
    const voteId = `vote_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await this.db.insert(this.schema.forumVotes).values({
      ...vote,
      voteId
    }).returning();
    
    const { sql } = require('drizzle-orm');
    if (vote.targetType === 'topic') {
      const increment = vote.voteType === 'upvote' ? 1 : -1;
      await this.db.update(this.schema.forumTopics)
        .set({ upvotes: sql`${this.schema.forumTopics.upvotes} + ${increment}` })
        .where(eq(this.schema.forumTopics.topicId, vote.targetId));
    } else if (vote.targetType === 'post') {
      const field = vote.voteType === 'upvote' ? this.schema.forumPosts.upvotes : this.schema.forumPosts.downvotes;
      const updateObj = vote.voteType === 'upvote' 
        ? { upvotes: sql`${this.schema.forumPosts.upvotes} + 1` }
        : { downvotes: sql`${this.schema.forumPosts.downvotes} + 1` };
      await this.db.update(this.schema.forumPosts)
        .set(updateObj)
        .where(eq(this.schema.forumPosts.postId, vote.targetId));
    }
    
    return { action: 'added', vote: result };
  }

  async createGuideline(guideline) {
    await this.initSchema();
    const guidelineId = `guideline_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await this.db.insert(this.schema.ethicalGuidelines).values({
      ...guideline,
      guidelineId
    }).returning();
    
    return result;
  }

  async getGuidelines(filters) {
    await this.initSchema();
    const conditions = [];
    
    if (filters?.category) {
      conditions.push(this.schema.ethicalGuidelines.category.eq(filters.category));
    }
    if (filters?.approvalStatus) {
      conditions.push(this.schema.ethicalGuidelines.approvalStatus.eq(filters.approvalStatus));
    }
    
    let query = this.db.select().from(this.schema.ethicalGuidelines);
    
    if (conditions.length > 0) {
      const { and } = require('drizzle-orm');
      query = query.where(and(...conditions));
    }
    
    const { desc } = require('drizzle-orm');
    return await query.orderBy(desc(this.schema.ethicalGuidelines.updatedAt));
  }

  async createExpertContribution(contribution) {
    await this.initSchema();
    const contributionId = `contrib_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await this.db.insert(this.schema.expertContributions).values({
      ...contribution,
      contributionId
    }).returning();
    
    return result;
  }

  async getExpertContributions(filters) {
    await this.initSchema();
    const conditions = [];
    
    if (filters?.expertId) {
      conditions.push(this.schema.expertContributions.expertId.eq(filters.expertId));
    }
    if (filters?.contributionType) {
      conditions.push(this.schema.expertContributions.contributionType.eq(filters.contributionType));
    }
    
    let query = this.db.select().from(this.schema.expertContributions);
    
    if (conditions.length > 0) {
      const { and } = require('drizzle-orm');
      query = query.where(and(...conditions));
    }
    
    const { desc } = require('drizzle-orm');
    return await query.orderBy(desc(this.schema.expertContributions.createdAt));
  }

  async createModerationAction(action) {
    await this.initSchema();
    const actionId = `modaction_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await this.db.insert(this.schema.forumModerationActions).values({
      ...action,
      actionId
    }).returning();
    
    return result;
  }

  async getModerationActions(filters) {
    await this.initSchema();
    const conditions = [];
    
    if (filters?.targetType) {
      conditions.push(this.schema.forumModerationActions.targetType.eq(filters.targetType));
    }
    if (filters?.actionType) {
      conditions.push(this.schema.forumModerationActions.actionType.eq(filters.actionType));
    }
    
    let query = this.db.select().from(this.schema.forumModerationActions);
    
    if (conditions.length > 0) {
      const { and } = require('drizzle-orm');
      query = query.where(and(...conditions));
    }
    
    const { desc } = require('drizzle-orm');
    return await query.orderBy(desc(this.schema.forumModerationActions.createdAt));
  }
}

module.exports = { HealthResearchStorage, ForumStorage };
