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

import {
  chatRooms,
  chatMessages,
  chatParticipants,
  badges,
  userBadges,
  userPoints,
  pointTransactions,
  leaderboards,
  events,
  eventRegistrations,
  eventSpeakers,
  feedbackSubmissions,
  feedbackResponses,
  feedbackCategories,
  feedbackAnalytics,
  curatedContentSections,
  curatedContent,
  contentRecommendations,
  notifications,
  type InsertChatRoom,
  type InsertChatMessage,
  type InsertChatParticipant,
  type InsertBadge,
  type InsertUserBadge,
  type InsertUserPoints,
  type InsertPointTransaction,
  type InsertLeaderboard,
  type InsertEvent,
  type InsertEventRegistration,
  type InsertEventSpeaker,
  type InsertFeedbackSubmission,
  type InsertFeedbackResponse,
  type InsertFeedbackCategory,
  type InsertFeedbackAnalytics,
  type InsertCuratedContentSection,
  type InsertCuratedContent,
  type InsertContentRecommendation,
  type InsertNotification
} from '../shared/schema.js';

export class ChatStorage {
  async createChatRoom(room: Omit<InsertChatRoom, 'roomId'>) {
    const roomId = `room_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(chatRooms).values({ ...room, roomId }).returning();
    return result;
  }

  async getChatRooms(filters?: { roomType?: string; communityId?: string }) {
    let query = db.select().from(chatRooms);
    if (filters?.roomType) query = query.where(eq(chatRooms.roomType, filters.roomType)) as any;
    if (filters?.communityId) query = query.where(eq(chatRooms.communityId, filters.communityId)) as any;
    return await query.orderBy(desc(chatRooms.createdAt));
  }

  async getChatRoomById(roomId: string) {
    const [result] = await db.select().from(chatRooms).where(eq(chatRooms.roomId, roomId));
    return result;
  }

  async createMessage(message: Omit<InsertChatMessage, 'messageId'>) {
    const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(chatMessages).values({ ...message, messageId }).returning();
    return result;
  }

  async getMessages(roomId: string, limit = 100) {
    return await db.select().from(chatMessages)
      .where(and(eq(chatMessages.roomId, roomId), eq(chatMessages.isDeleted, false)))
      .orderBy(desc(chatMessages.createdAt))
      .limit(limit);
  }

  async joinRoom(participant: Omit<InsertChatParticipant, 'id'>) {
    const [result] = await db.insert(chatParticipants).values(participant).returning();
    await db.update(chatRooms)
      .set({ activeParticipants: sql`${chatRooms.activeParticipants} + 1` })
      .where(eq(chatRooms.roomId, participant.roomId));
    return result;
  }

  async leaveRoom(roomId: string, userId: string) {
    await db.delete(chatParticipants)
      .where(and(eq(chatParticipants.roomId, roomId), eq(chatParticipants.userId, userId)));
    await db.update(chatRooms)
      .set({ activeParticipants: sql`${chatRooms.activeParticipants} - 1` })
      .where(eq(chatRooms.roomId, roomId));
  }

  async getRoomParticipants(roomId: string) {
    return await db.select().from(chatParticipants).where(eq(chatParticipants.roomId, roomId));
  }
}

export class GamificationStorage {
  async createBadge(badge: Omit<InsertBadge, 'badgeId'>) {
    const badgeId = `badge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(badges).values({ ...badge, badgeId }).returning();
    return result;
  }

  async getBadges(filters?: { badgeType?: string; category?: string }) {
    let query = db.select().from(badges);
    if (filters?.badgeType) query = query.where(eq(badges.badgeType, filters.badgeType)) as any;
    if (filters?.category) query = query.where(eq(badges.category, filters.category)) as any;
    return await query.orderBy(desc(badges.createdAt));
  }

  async awardBadge(award: Omit<InsertUserBadge, 'id'>) {
    const [result] = await db.insert(userBadges).values(award).returning();
    return result;
  }

  async getUserBadges(userId: string) {
    return await db.select().from(userBadges).where(eq(userBadges.userId, userId));
  }

  async getUserPoints(userId: string) {
    const [result] = await db.select().from(userPoints).where(eq(userPoints.userId, userId));
    return result;
  }

  async addPoints(transaction: Omit<InsertPointTransaction, 'transactionId'>) {
    const transactionId = `trans_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(pointTransactions).values({ ...transaction, transactionId }).returning();
    
    const existing = await this.getUserPoints(transaction.userId);
    if (existing) {
      await db.update(userPoints)
        .set({ 
          totalPoints: sql`${userPoints.totalPoints} + ${transaction.points}`,
          levelPoints: sql`${userPoints.levelPoints} + ${transaction.points}`,
          updatedAt: new Date()
        })
        .where(eq(userPoints.userId, transaction.userId));
    } else {
      await db.insert(userPoints).values({
        userId: transaction.userId,
        totalPoints: transaction.points,
        levelPoints: transaction.points,
        level: 1,
        streak: 1,
        lastActivityDate: new Date()
      });
    }
    
    return result;
  }

  async getLeaderboard(leaderboardId: string, limit = 100) {
    return await db.select().from(userPoints).orderBy(desc(userPoints.totalPoints)).limit(limit);
  }
}

export class EventsStorage {
  async createEvent(event: Omit<InsertEvent, 'eventId'>) {
    const eventId = `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(events).values({ ...event, eventId }).returning();
    return result;
  }

  async getEvents(filters?: { eventType?: string; category?: string; status?: string }) {
    let query = db.select().from(events);
    if (filters?.eventType) query = query.where(eq(events.eventType, filters.eventType)) as any;
    if (filters?.category) query = query.where(eq(events.category, filters.category)) as any;
    if (filters?.status) query = query.where(eq(events.status, filters.status)) as any;
    return await query.orderBy(events.startTime);
  }

  async getEventById(eventId: string) {
    const [result] = await db.select().from(events).where(eq(events.eventId, eventId));
    return result;
  }

  async registerForEvent(registration: Omit<InsertEventRegistration, 'registrationId'>) {
    const registrationId = `reg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(eventRegistrations).values({ ...registration, registrationId }).returning();
    
    await db.update(events)
      .set({ registeredCount: sql`${events.registeredCount} + 1` })
      .where(eq(events.eventId, registration.eventId));
    
    return result;
  }

  async getEventRegistrations(eventId: string) {
    return await db.select().from(eventRegistrations).where(eq(eventRegistrations.eventId, eventId));
  }

  async getUserRegistrations(userId: string) {
    return await db.select().from(eventRegistrations).where(eq(eventRegistrations.userId, userId));
  }

  async addSpeaker(speaker: Omit<InsertEventSpeaker, 'id'>) {
    const [result] = await db.insert(eventSpeakers).values(speaker).returning();
    return result;
  }

  async getEventSpeakers(eventId: string) {
    return await db.select().from(eventSpeakers).where(eq(eventSpeakers.eventId, eventId));
  }
}

export class FeedbackStorage {
  async submitFeedback(feedback: Omit<InsertFeedbackSubmission, 'feedbackId'>) {
    const feedbackId = `feedback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(feedbackSubmissions).values({ ...feedback, feedbackId }).returning();
    return result;
  }

  async getFeedback(filters?: { feedbackType?: string; category?: string; status?: string; sentiment?: string }) {
    let query = db.select().from(feedbackSubmissions);
    if (filters?.feedbackType) query = query.where(eq(feedbackSubmissions.feedbackType, filters.feedbackType)) as any;
    if (filters?.category) query = query.where(eq(feedbackSubmissions.category, filters.category)) as any;
    if (filters?.status) query = query.where(eq(feedbackSubmissions.status, filters.status)) as any;
    if (filters?.sentiment) query = query.where(eq(feedbackSubmissions.sentiment, filters.sentiment)) as any;
    return await query.orderBy(desc(feedbackSubmissions.createdAt));
  }

  async getFeedbackById(feedbackId: string) {
    const [result] = await db.select().from(feedbackSubmissions).where(eq(feedbackSubmissions.feedbackId, feedbackId));
    return result;
  }

  async respondToFeedback(response: Omit<InsertFeedbackResponse, 'responseId'>) {
    const responseId = `response_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(feedbackResponses).values({ ...response, responseId }).returning();
    
    if (response.responseType === 'resolution') {
      await db.update(feedbackSubmissions)
        .set({ status: 'completed', updatedAt: new Date() })
        .where(eq(feedbackSubmissions.feedbackId, response.feedbackId));
    }
    
    return result;
  }

  async getFeedbackResponses(feedbackId: string) {
    return await db.select().from(feedbackResponses)
      .where(eq(feedbackResponses.feedbackId, feedbackId))
      .orderBy(desc(feedbackResponses.createdAt));
  }

  async updateFeedbackStatus(feedbackId: string, status: string) {
    await db.update(feedbackSubmissions)
      .set({ status, updatedAt: new Date() })
      .where(eq(feedbackSubmissions.feedbackId, feedbackId));
  }

  async voteFeedback(feedbackId: string, increment: number) {
    await db.update(feedbackSubmissions)
      .set({ votes: sql`${feedbackSubmissions.votes} + ${increment}` })
      .where(eq(feedbackSubmissions.feedbackId, feedbackId));
  }
}

export class CuratedContentStorage {
  async createContentSection(section: Omit<InsertCuratedContentSection, 'sectionId'>) {
    const sectionId = `section_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(curatedContentSections).values({ ...section, sectionId }).returning();
    return result;
  }

  async getContentSections(filters?: { sectionType?: string; isActive?: boolean }) {
    let query = db.select().from(curatedContentSections);
    if (filters?.sectionType) query = query.where(eq(curatedContentSections.sectionType, filters.sectionType)) as any;
    if (filters?.isActive !== undefined) query = query.where(eq(curatedContentSections.isActive, filters.isActive)) as any;
    return await query.orderBy(curatedContentSections.displayOrder);
  }

  async addCuratedContent(content: Omit<InsertCuratedContent, 'curatedId'>) {
    const curatedId = `curated_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(curatedContent).values({ ...content, curatedId }).returning();
    return result;
  }

  async getCuratedContent(sectionId: string, userId?: string) {
    if (userId) {
      return await db.select().from(curatedContent)
        .where(
          and(
            eq(curatedContent.sectionId, sectionId),
            or(
              eq(curatedContent.personalizedFor, userId),
              eq(curatedContent.personalizedFor, null as any)
            )
          )
        )
        .orderBy(curatedContent.displayOrder, desc(curatedContent.createdAt));
    }
    return await db.select().from(curatedContent)
      .where(eq(curatedContent.sectionId, sectionId))
      .orderBy(curatedContent.displayOrder, desc(curatedContent.createdAt));
  }

  async trackContentImpression(curatedId: string) {
    await db.update(curatedContent)
      .set({ impressions: sql`${curatedContent.impressions} + 1` })
      .where(eq(curatedContent.curatedId, curatedId));
  }

  async trackContentClick(curatedId: string) {
    await db.update(curatedContent)
      .set({ clicks: sql`${curatedContent.clicks} + 1` })
      .where(eq(curatedContent.curatedId, curatedId));
  }

  async addRecommendation(recommendation: Omit<InsertContentRecommendation, 'recommendationId'>) {
    const recommendationId = `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(contentRecommendations).values({ ...recommendation, recommendationId }).returning();
    return result;
  }

  async getUserRecommendations(userId: string, limit = 20) {
    return await db.select().from(contentRecommendations)
      .where(eq(contentRecommendations.userId, userId))
      .orderBy(desc(contentRecommendations.score))
      .limit(limit);
  }
}

export class NotificationStorage {
  async createNotification(notification: Omit<InsertNotification, 'notificationId'>) {
    const notificationId = `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const [result] = await db.insert(notifications).values({ ...notification, notificationId }).returning();
    return result;
  }

  async getUserNotifications(userId: string, limit = 50) {
    return await db.select().from(notifications)
      .where(eq(notifications.userId, userId))
      .orderBy(desc(notifications.createdAt))
      .limit(limit);
  }

  async markAsRead(notificationId: string) {
    await db.update(notifications)
      .set({ isRead: true, readAt: new Date() })
      .where(eq(notifications.notificationId, notificationId));
  }

  async getUnreadCount(userId: string) {
    const result = await db.select({ count: sql<number>`count(*)` })
      .from(notifications)
      .where(and(eq(notifications.userId, userId), eq(notifications.isRead, false)));
    return result[0]?.count || 0;
  }
}

export const chatStorage = new ChatStorage();
export const gamificationStorage = new GamificationStorage();
export const eventsStorage = new EventsStorage();
export const feedbackStorage = new FeedbackStorage();
export const curatedContentStorage = new CuratedContentStorage();
export const notificationStorage = new NotificationStorage();
