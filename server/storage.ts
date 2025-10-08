import { db } from './db.js';
import { 
  healthData, 
  wellnessJourneys, 
  healthInsights, 
  researchFindings,
  forumTopics,
  forumPosts,
  forumVotes,
  ethicalGuidelines,
  expertContributions,
  forumModerationActions,
  type InsertHealthData,
  type InsertWellnessJourney,
  type InsertHealthInsight,
  type InsertResearchFinding,
  type InsertForumTopic,
  type InsertForumPost,
  type InsertForumVote,
  type InsertEthicalGuideline,
  type InsertExpertContribution,
  type InsertForumModerationAction
} from '../shared/schema.js';
import { eq, desc, and, or, sql } from 'drizzle-orm';
import { createHash } from 'crypto';

export class HealthResearchStorage {
  async submitHealthData(data: Omit<InsertHealthData, 'dataId' | 'anonymousUserId'> & { userId: string }) {
    const anonymousUserId = this.hashUserId(data.userId);
    const dataId = `health_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const { userId, ...healthDataWithoutUserId } = data;
    
    const [result] = await db.insert(healthData).values({
      ...healthDataWithoutUserId,
      dataId,
      anonymousUserId
    }).returning();
    
    return result;
  }

  async getHealthData(filters?: { dataType?: string; category?: string; isPublic?: boolean }) {
    let query = db.select().from(healthData);
    
    if (filters?.dataType) {
      query = query.where(eq(healthData.dataType, filters.dataType)) as any;
    }
    if (filters?.category) {
      query = query.where(eq(healthData.category, filters.category)) as any;
    }
    if (filters?.isPublic !== undefined) {
      query = query.where(eq(healthData.isPublic, filters.isPublic)) as any;
    }
    
    return await query.orderBy(desc(healthData.createdAt));
  }

  async submitWellnessJourney(journey: Omit<InsertWellnessJourney, 'journeyId' | 'anonymousUserId'> & { userId: string }) {
    const anonymousUserId = this.hashUserId(journey.userId);
    const journeyId = `journey_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const { userId, ...journeyWithoutUserId } = journey;
    
    const [result] = await db.insert(wellnessJourneys).values({
      ...journeyWithoutUserId,
      journeyId,
      anonymousUserId
    }).returning();
    
    return result;
  }

  async getWellnessJourneys(filters?: { journeyType?: string; isActive?: boolean; isPublic?: boolean }) {
    let query = db.select().from(wellnessJourneys);
    
    if (filters?.journeyType) {
      query = query.where(eq(wellnessJourneys.journeyType, filters.journeyType)) as any;
    }
    if (filters?.isActive !== undefined) {
      query = query.where(eq(wellnessJourneys.isActive, filters.isActive)) as any;
    }
    if (filters?.isPublic !== undefined) {
      query = query.where(eq(wellnessJourneys.isPublic, filters.isPublic)) as any;
    }
    
    return await query.orderBy(desc(wellnessJourneys.createdAt));
  }

  async createHealthInsight(insight: Omit<InsertHealthInsight, 'insightId'>) {
    const insightId = `insight_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await db.insert(healthInsights).values({
      ...insight,
      insightId
    }).returning();
    
    return result;
  }

  async getHealthInsights(filters?: { insightType?: string; category?: string }) {
    let query = db.select().from(healthInsights);
    
    if (filters?.insightType) {
      query = query.where(eq(healthInsights.insightType, filters.insightType)) as any;
    }
    if (filters?.category) {
      query = query.where(eq(healthInsights.category, filters.category)) as any;
    }
    
    return await query.orderBy(desc(healthInsights.createdAt));
  }

  async createResearchFinding(finding: Omit<InsertResearchFinding, 'findingId'>) {
    const findingId = `finding_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await db.insert(researchFindings).values({
      ...finding,
      findingId
    }).returning();
    
    return result;
  }

  async getResearchFindings(filters?: { findingType?: string; status?: string }) {
    let query = db.select().from(researchFindings);
    
    if (filters?.findingType) {
      query = query.where(eq(researchFindings.findingType, filters.findingType)) as any;
    }
    if (filters?.status) {
      query = query.where(eq(researchFindings.status, filters.status)) as any;
    }
    
    return await query.orderBy(desc(researchFindings.createdAt));
  }

  private hashUserId(userId: string): string {
    return createHash('sha256').update(userId).digest('hex');
  }
}

export class ForumStorage {
  async createTopic(topic: Omit<InsertForumTopic, 'topicId'>) {
    const topicId = `topic_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await db.insert(forumTopics).values({
      ...topic,
      topicId
    }).returning();
    
    return result;
  }

  async getTopics(filters?: { category?: string; status?: string }) {
    let query = db.select().from(forumTopics);
    
    if (filters?.category) {
      query = query.where(eq(forumTopics.category, filters.category)) as any;
    }
    if (filters?.status) {
      query = query.where(eq(forumTopics.status, filters.status)) as any;
    }
    
    return await query.orderBy(desc(forumTopics.isPinned), desc(forumTopics.createdAt));
  }

  async getTopicById(topicId: string) {
    const [result] = await db.select().from(forumTopics).where(eq(forumTopics.topicId, topicId));
    return result;
  }

  async incrementTopicViews(topicId: string) {
    await db.update(forumTopics)
      .set({ viewCount: sql`${forumTopics.viewCount} + 1` })
      .where(eq(forumTopics.topicId, topicId));
  }

  async createPost(post: Omit<InsertForumPost, 'postId'>) {
    const postId = `post_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await db.insert(forumPosts).values({
      ...post,
      postId
    }).returning();
    
    await db.update(forumTopics)
      .set({ postCount: sql`${forumTopics.postCount} + 1` })
      .where(eq(forumTopics.topicId, post.topicId));
    
    return result;
  }

  async getPostsByTopic(topicId: string) {
    return await db.select().from(forumPosts)
      .where(eq(forumPosts.topicId, topicId))
      .orderBy(desc(forumPosts.createdAt));
  }

  async voteOnTarget(vote: Omit<InsertForumVote, 'voteId'>) {
    const existingVote = await db.select().from(forumVotes)
      .where(
        and(
          eq(forumVotes.userId, vote.userId),
          eq(forumVotes.targetId, vote.targetId)
        )
      );
    
    if (existingVote.length > 0) {
      await db.delete(forumVotes).where(eq(forumVotes.voteId, existingVote[0].voteId));
      
      if (existingVote[0].voteType === vote.voteType) {
        return { action: 'removed', vote: existingVote[0] };
      }
    }
    
    const voteId = `vote_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(forumVotes).values({
      ...vote,
      voteId
    }).returning();
    
    if (vote.targetType === 'topic') {
      const increment = vote.voteType === 'upvote' ? 1 : -1;
      await db.update(forumTopics)
        .set({ upvotes: sql`${forumTopics.upvotes} + ${increment}` })
        .where(eq(forumTopics.topicId, vote.targetId));
    } else if (vote.targetType === 'post') {
      const field = vote.voteType === 'upvote' ? forumPosts.upvotes : forumPosts.downvotes;
      await db.update(forumPosts)
        .set({ [vote.voteType === 'upvote' ? 'upvotes' : 'downvotes']: sql`${field} + 1` })
        .where(eq(forumPosts.postId, vote.targetId));
    }
    
    return { action: 'added', vote: result };
  }

  async createGuideline(guideline: Omit<InsertEthicalGuideline, 'guidelineId'>) {
    const guidelineId = `guideline_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await db.insert(ethicalGuidelines).values({
      ...guideline,
      guidelineId
    }).returning();
    
    return result;
  }

  async getGuidelines(filters?: { category?: string; approvalStatus?: string }) {
    let query = db.select().from(ethicalGuidelines);
    
    if (filters?.category) {
      query = query.where(eq(ethicalGuidelines.category, filters.category)) as any;
    }
    if (filters?.approvalStatus) {
      query = query.where(eq(ethicalGuidelines.approvalStatus, filters.approvalStatus)) as any;
    }
    
    return await query.orderBy(desc(ethicalGuidelines.updatedAt));
  }

  async createExpertContribution(contribution: Omit<InsertExpertContribution, 'contributionId'>) {
    const contributionId = `contrib_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await db.insert(expertContributions).values({
      ...contribution,
      contributionId
    }).returning();
    
    return result;
  }

  async getExpertContributions(filters?: { expertId?: string; contributionType?: string }) {
    let query = db.select().from(expertContributions);
    
    if (filters?.expertId) {
      query = query.where(eq(expertContributions.expertId, filters.expertId)) as any;
    }
    if (filters?.contributionType) {
      query = query.where(eq(expertContributions.contributionType, filters.contributionType)) as any;
    }
    
    return await query.orderBy(desc(expertContributions.createdAt));
  }

  async createModerationAction(action: Omit<InsertForumModerationAction, 'actionId'>) {
    const actionId = `modaction_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const [result] = await db.insert(forumModerationActions).values({
      ...action,
      actionId
    }).returning();
    
    return result;
  }

  async getModerationActions(filters?: { targetType?: string; actionType?: string }) {
    let query = db.select().from(forumModerationActions);
    
    if (filters?.targetType) {
      query = query.where(eq(forumModerationActions.targetType, filters.targetType)) as any;
    }
    if (filters?.actionType) {
      query = query.where(eq(forumModerationActions.actionType, filters.actionType)) as any;
    }
    
    return await query.orderBy(desc(forumModerationActions.createdAt));
  }
}

export const healthResearchStorage = new HealthResearchStorage();
export const forumStorage = new ForumStorage();
