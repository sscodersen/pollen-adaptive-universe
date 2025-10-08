import { pgTable, serial, text, timestamp, real, integer, jsonb, boolean } from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";

// Content storage table for Pollen AI generated content
export const content = pgTable("content", {
  id: serial("id").primaryKey(),
  contentId: text("content_id").notNull().unique(), // Original content_${timestamp}_${random} format
  content: text("content").notNull(),
  confidence: real("confidence").default(0.8),
  reasoning: text("reasoning"),
  type: text("type").notNull(), // 'general', 'music', 'product', 'feed_post'
  mode: text("mode").notNull(), // 'chat', 'analysis'
  prompt: text("prompt").notNull(),
  views: integer("views").default(0),
  metadata: jsonb("metadata"), // Additional data like title, industry, etc.
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Feed aggregation table for main feed visibility
export const feedItems = pgTable("feed_items", {
  id: serial("id").primaryKey(),
  contentId: text("content_id").notNull(), // Reference to content.contentId
  feedId: text("feed_id").notNull().unique(), // feed_${content_id} format
  featured: integer("featured").default(0), // 0 = normal, 1 = featured
  createdAt: timestamp("created_at").defaultNow()
});

// User interactions for analytics (future use)
export const interactions = pgTable("interactions", {
  id: serial("id").primaryKey(),
  contentId: text("content_id").notNull(),
  action: text("action").notNull(), // 'view', 'like', 'share'
  timestamp: timestamp("timestamp").defaultNow(),
  metadata: jsonb("metadata") // Additional interaction data
});

// User profiles for community building and personalization
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  userId: text("user_id").notNull().unique(),
  username: text("username").notNull(),
  email: text("email"),
  profileData: jsonb("profile_data"), // interests, preferences, demographics
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Ethical concerns and AI bias reports
export const ethicalReports = pgTable("ethical_reports", {
  id: serial("id").primaryKey(),
  reportId: text("report_id").notNull().unique(),
  userId: text("user_id").notNull(),
  contentId: text("content_id"), // Optional: related content
  concernType: text("concern_type").notNull(), // 'bias', 'fairness', 'transparency', 'harmful_content'
  description: text("description").notNull(),
  severity: text("severity").notNull(), // 'low', 'medium', 'high', 'critical'
  status: text("status").notNull().default('pending'), // 'pending', 'reviewing', 'resolved', 'dismissed'
  resolution: text("resolution"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// AI bias detection logs
export const biasDetectionLogs = pgTable("bias_detection_logs", {
  id: serial("id").primaryKey(),
  logId: text("log_id").notNull().unique(),
  contentId: text("content_id").notNull(),
  biasType: text("bias_type").notNull(), // 'gender', 'racial', 'age', 'cultural', 'political'
  detectionScore: real("detection_score").notNull(),
  mitigationApplied: boolean("mitigation_applied").default(false),
  mitigationStrategy: text("mitigation_strategy"),
  originalContent: text("original_content"),
  mitigatedContent: text("mitigated_content"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

// AI decision transparency logs
export const aiDecisionLogs = pgTable("ai_decision_logs", {
  id: serial("id").primaryKey(),
  logId: text("log_id").notNull().unique(),
  userId: text("user_id"),
  contentId: text("content_id"),
  decisionType: text("decision_type").notNull(), // 'recommendation', 'content_generation', 'filtering'
  factors: jsonb("factors").notNull(), // Array of decision factors with weights
  reasoning: text("reasoning").notNull(),
  confidence: real("confidence"),
  createdAt: timestamp("created_at").defaultNow()
});

// Communities and support groups
export const communities = pgTable("communities", {
  id: serial("id").primaryKey(),
  communityId: text("community_id").notNull().unique(),
  name: text("name").notNull(),
  description: text("description"),
  type: text("type").notNull(), // 'support_group', 'interest_group', 'topic_based'
  category: text("category"), // 'wellness', 'agriculture', 'social_impact', 'opportunities'
  isPrivate: boolean("is_private").default(false),
  memberCount: integer("member_count").default(0),
  creatorId: text("creator_id").notNull(),
  rules: jsonb("rules"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Community memberships
export const communityMembers = pgTable("community_members", {
  id: serial("id").primaryKey(),
  communityId: text("community_id").notNull(),
  userId: text("user_id").notNull(),
  role: text("role").notNull().default('member'), // 'member', 'moderator', 'admin'
  joinedAt: timestamp("joined_at").defaultNow(),
  status: text("status").notNull().default('active') // 'active', 'muted', 'banned'
});

// Community posts and discussions
export const communityPosts = pgTable("community_posts", {
  id: serial("id").primaryKey(),
  postId: text("post_id").notNull().unique(),
  communityId: text("community_id").notNull(),
  userId: text("user_id").notNull(),
  content: text("content").notNull(),
  postType: text("post_type").notNull().default('discussion'), // 'discussion', 'question', 'resource', 'announcement'
  likes: integer("likes").default(0),
  replies: integer("replies").default(0),
  isPinned: boolean("is_pinned").default(false),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Moderation actions
export const moderationActions = pgTable("moderation_actions", {
  id: serial("id").primaryKey(),
  actionId: text("action_id").notNull().unique(),
  communityId: text("community_id").notNull(),
  moderatorId: text("moderator_id").notNull(),
  targetUserId: text("target_user_id"),
  targetPostId: text("target_post_id"),
  actionType: text("action_type").notNull(), // 'warn', 'mute', 'ban', 'delete_post', 'pin_post'
  reason: text("reason").notNull(),
  duration: integer("duration"), // Duration in minutes for temporary actions
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

// User matching scores for community suggestions
export const userMatchingScores = pgTable("user_matching_scores", {
  id: serial("id").primaryKey(),
  userId1: text("user_id_1").notNull(),
  userId2: text("user_id_2").notNull(),
  matchScore: real("match_score").notNull(),
  commonInterests: jsonb("common_interests"),
  calculatedAt: timestamp("calculated_at").defaultNow()
});

// Relations
export const contentRelations = relations(content, ({ many }) => ({
  feedItems: many(feedItems),
  interactions: many(interactions),
  biasLogs: many(biasDetectionLogs),
  ethicalReports: many(ethicalReports)
}));

export const feedItemsRelations = relations(feedItems, ({ one }) => ({
  content: one(content, {
    fields: [feedItems.contentId],
    references: [content.contentId]
  })
}));

export const interactionsRelations = relations(interactions, ({ one }) => ({
  content: one(content, {
    fields: [interactions.contentId], 
    references: [content.contentId]
  })
}));

export const usersRelations = relations(users, ({ many }) => ({
  ethicalReports: many(ethicalReports),
  communityMemberships: many(communityMembers),
  communityPosts: many(communityPosts),
  createdCommunities: many(communities)
}));

export const communitiesRelations = relations(communities, ({ one, many }) => ({
  creator: one(users, {
    fields: [communities.creatorId],
    references: [users.userId]
  }),
  members: many(communityMembers),
  posts: many(communityPosts),
  moderationActions: many(moderationActions)
}));

export const communityMembersRelations = relations(communityMembers, ({ one }) => ({
  community: one(communities, {
    fields: [communityMembers.communityId],
    references: [communities.communityId]
  }),
  user: one(users, {
    fields: [communityMembers.userId],
    references: [users.userId]
  })
}));

// Types
export type Content = typeof content.$inferSelect;
export type InsertContent = typeof content.$inferInsert;
export type FeedItem = typeof feedItems.$inferSelect;
export type InsertFeedItem = typeof feedItems.$inferInsert;
export type Interaction = typeof interactions.$inferSelect;
export type InsertInteraction = typeof interactions.$inferInsert;
export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;
export type EthicalReport = typeof ethicalReports.$inferSelect;
export type InsertEthicalReport = typeof ethicalReports.$inferInsert;
export type BiasDetectionLog = typeof biasDetectionLogs.$inferSelect;
export type InsertBiasDetectionLog = typeof biasDetectionLogs.$inferInsert;
export type AiDecisionLog = typeof aiDecisionLogs.$inferSelect;
export type InsertAiDecisionLog = typeof aiDecisionLogs.$inferInsert;
export type Community = typeof communities.$inferSelect;
export type InsertCommunity = typeof communities.$inferInsert;
export type CommunityMember = typeof communityMembers.$inferSelect;
export type InsertCommunityMember = typeof communityMembers.$inferInsert;
export type CommunityPost = typeof communityPosts.$inferSelect;
export type InsertCommunityPost = typeof communityPosts.$inferInsert;
export type ModerationAction = typeof moderationActions.$inferSelect;
export type InsertModerationAction = typeof moderationActions.$inferInsert;