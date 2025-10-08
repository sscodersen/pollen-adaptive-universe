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

// Health & Wellness Research Platform Tables

// Health data submissions (anonymized)
export const healthData = pgTable("health_data", {
  id: serial("id").primaryKey(),
  dataId: text("data_id").notNull().unique(),
  anonymousUserId: text("anonymous_user_id").notNull(), // Hashed user identifier
  dataType: text("data_type").notNull(), // 'fitness', 'nutrition', 'mental_health', 'sleep', 'medical'
  category: text("category").notNull(), // Specific subcategory
  metrics: jsonb("metrics").notNull(), // Health metrics data
  demographics: jsonb("demographics"), // Age range, region (anonymized)
  tags: jsonb("tags"), // Searchable tags
  isPublic: boolean("is_public").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Wellness journeys tracking
export const wellnessJourneys = pgTable("wellness_journeys", {
  id: serial("id").primaryKey(),
  journeyId: text("journey_id").notNull().unique(),
  anonymousUserId: text("anonymous_user_id").notNull(),
  journeyType: text("journey_type").notNull(), // 'weight_loss', 'fitness', 'mental_wellness', 'recovery'
  startDate: timestamp("start_date").notNull(),
  endDate: timestamp("end_date"),
  milestones: jsonb("milestones"), // Journey milestones and progress
  outcomes: jsonb("outcomes"), // Results achieved
  challenges: jsonb("challenges"), // Challenges faced
  insights: text("insights"), // Personal insights
  isActive: boolean("is_active").default(true),
  isPublic: boolean("is_public").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// AI-generated health insights
export const healthInsights = pgTable("health_insights", {
  id: serial("id").primaryKey(),
  insightId: text("insight_id").notNull().unique(),
  insightType: text("insight_type").notNull(), // 'trend', 'correlation', 'recommendation', 'breakthrough'
  category: text("category").notNull(), // Health category
  title: text("title").notNull(),
  description: text("description").notNull(),
  dataPoints: integer("data_points"), // Number of data points analyzed
  confidence: real("confidence"), // AI confidence score
  significance: real("significance"), // Statistical significance
  visualizationData: jsonb("visualization_data"), // Data for charts/graphs
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

// Research findings and breakthroughs
export const researchFindings = pgTable("research_findings", {
  id: serial("id").primaryKey(),
  findingId: text("finding_id").notNull().unique(),
  title: text("title").notNull(),
  summary: text("summary").notNull(),
  fullReport: text("full_report"),
  findingType: text("finding_type").notNull(), // 'correlation', 'breakthrough', 'pattern', 'anomaly'
  impactScore: real("impact_score"), // Potential impact rating
  datasetSize: integer("dataset_size"),
  categories: jsonb("categories"), // Related health categories
  keyMetrics: jsonb("key_metrics"), // Important metrics
  visualizations: jsonb("visualizations"), // Visualization configs
  citations: jsonb("citations"), // Related data sources
  status: text("status").default('draft'), // 'draft', 'published', 'peer_review'
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// AI Ethics and Responsible Innovation Forum Tables

// Forum topics and discussions
export const forumTopics = pgTable("forum_topics", {
  id: serial("id").primaryKey(),
  topicId: text("topic_id").notNull().unique(),
  creatorId: text("creator_id").notNull(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  category: text("category").notNull(), // 'ai_bias', 'privacy', 'transparency', 'fairness', 'accountability', 'safety'
  tags: jsonb("tags"),
  status: text("status").default('active'), // 'active', 'closed', 'archived'
  isPinned: boolean("is_pinned").default(false),
  viewCount: integer("view_count").default(0),
  postCount: integer("post_count").default(0),
  upvotes: integer("upvotes").default(0),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Forum posts and replies
export const forumPosts = pgTable("forum_posts", {
  id: serial("id").primaryKey(),
  postId: text("post_id").notNull().unique(),
  topicId: text("topic_id").notNull(),
  userId: text("user_id").notNull(),
  parentPostId: text("parent_post_id"), // For threaded replies
  content: text("content").notNull(),
  postType: text("post_type").default('reply'), // 'reply', 'expert_opinion', 'proposal'
  upvotes: integer("upvotes").default(0),
  downvotes: integer("downvotes").default(0),
  isExpertPost: boolean("is_expert_post").default(false),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Forum voting records
export const forumVotes = pgTable("forum_votes", {
  id: serial("id").primaryKey(),
  voteId: text("vote_id").notNull().unique(),
  userId: text("user_id").notNull(),
  targetType: text("target_type").notNull(), // 'topic', 'post'
  targetId: text("target_id").notNull(),
  voteType: text("vote_type").notNull(), // 'upvote', 'downvote'
  createdAt: timestamp("created_at").defaultNow()
});

// Ethical guidelines and policies
export const ethicalGuidelines = pgTable("ethical_guidelines", {
  id: serial("id").primaryKey(),
  guidelineId: text("guideline_id").notNull().unique(),
  title: text("title").notNull(),
  content: text("content").notNull(),
  category: text("category").notNull(),
  version: text("version").notNull(),
  contributorCount: integer("contributor_count").default(0),
  approvalStatus: text("approval_status").default('draft'), // 'draft', 'review', 'approved'
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Expert contributions and invitations
export const expertContributions = pgTable("expert_contributions", {
  id: serial("id").primaryKey(),
  contributionId: text("contribution_id").notNull().unique(),
  expertId: text("expert_id").notNull(),
  contributionType: text("contribution_type").notNull(), // 'opinion', 'guideline', 'research', 'review'
  relatedTopicId: text("related_topic_id"),
  relatedGuidelineId: text("related_guideline_id"),
  content: text("content").notNull(),
  expertise: jsonb("expertise"), // Areas of expertise
  citations: jsonb("citations"),
  impactScore: real("impact_score"),
  createdAt: timestamp("created_at").defaultNow()
});

// Forum moderation for ethics discussions
export const forumModerationActions = pgTable("forum_moderation_actions", {
  id: serial("id").primaryKey(),
  actionId: text("action_id").notNull().unique(),
  moderatorId: text("moderator_id").notNull(),
  targetType: text("target_type").notNull(), // 'topic', 'post', 'user'
  targetId: text("target_id").notNull(),
  actionType: text("action_type").notNull(), // 'approve', 'flag', 'remove', 'warn', 'ban'
  reason: text("reason").notNull(),
  automated: boolean("automated").default(false), // AI-assisted moderation
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
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

export const forumTopicsRelations = relations(forumTopics, ({ one, many }) => ({
  creator: one(users, {
    fields: [forumTopics.creatorId],
    references: [users.userId]
  }),
  posts: many(forumPosts),
  votes: many(forumVotes)
}));

export const forumPostsRelations = relations(forumPosts, ({ one, many }) => ({
  topic: one(forumTopics, {
    fields: [forumPosts.topicId],
    references: [forumTopics.topicId]
  }),
  author: one(users, {
    fields: [forumPosts.userId],
    references: [users.userId]
  }),
  votes: many(forumVotes)
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
export type HealthData = typeof healthData.$inferSelect;
export type InsertHealthData = typeof healthData.$inferInsert;
export type WellnessJourney = typeof wellnessJourneys.$inferSelect;
export type InsertWellnessJourney = typeof wellnessJourneys.$inferInsert;
export type HealthInsight = typeof healthInsights.$inferSelect;
export type InsertHealthInsight = typeof healthInsights.$inferInsert;
export type ResearchFinding = typeof researchFindings.$inferSelect;
export type InsertResearchFinding = typeof researchFindings.$inferInsert;
export type ForumTopic = typeof forumTopics.$inferSelect;
export type InsertForumTopic = typeof forumTopics.$inferInsert;
export type ForumPost = typeof forumPosts.$inferSelect;
export type InsertForumPost = typeof forumPosts.$inferInsert;
export type ForumVote = typeof forumVotes.$inferSelect;
export type InsertForumVote = typeof forumVotes.$inferInsert;
export type EthicalGuideline = typeof ethicalGuidelines.$inferSelect;
export type InsertEthicalGuideline = typeof ethicalGuidelines.$inferInsert;
export type ExpertContribution = typeof expertContributions.$inferSelect;
export type InsertExpertContribution = typeof expertContributions.$inferInsert;
export type ForumModerationAction = typeof forumModerationActions.$inferSelect;
export type InsertForumModerationAction = typeof forumModerationActions.$inferInsert;
