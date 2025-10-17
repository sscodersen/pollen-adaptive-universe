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

// Real-time Chat Rooms
export const chatRooms = pgTable("chat_rooms", {
  id: serial("id").primaryKey(),
  roomId: text("room_id").notNull().unique(),
  name: text("name").notNull(),
  description: text("description"),
  roomType: text("room_type").notNull(), // 'public', 'private', 'community', 'event'
  communityId: text("community_id"), // Link to community if community chat
  creatorId: text("creator_id").notNull(),
  maxParticipants: integer("max_participants").default(100),
  activeParticipants: integer("active_participants").default(0),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

export const chatMessages = pgTable("chat_messages", {
  id: serial("id").primaryKey(),
  messageId: text("message_id").notNull().unique(),
  roomId: text("room_id").notNull(),
  userId: text("user_id").notNull(),
  content: text("content").notNull(),
  messageType: text("message_type").default('text'), // 'text', 'image', 'file', 'system'
  replyToId: text("reply_to_id"), // For threaded conversations
  reactions: jsonb("reactions"), // Emoji reactions
  isEdited: boolean("is_edited").default(false),
  isDeleted: boolean("is_deleted").default(false),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

export const chatParticipants = pgTable("chat_participants", {
  id: serial("id").primaryKey(),
  roomId: text("room_id").notNull(),
  userId: text("user_id").notNull(),
  role: text("role").default('participant'), // 'participant', 'moderator', 'admin'
  lastReadAt: timestamp("last_read_at"),
  joinedAt: timestamp("joined_at").defaultNow(),
  status: text("status").default('active') // 'active', 'away', 'offline'
});

// Gamification System
export const badges = pgTable("badges", {
  id: serial("id").primaryKey(),
  badgeId: text("badge_id").notNull().unique(),
  name: text("name").notNull(),
  description: text("description").notNull(),
  iconUrl: text("icon_url"),
  badgeType: text("badge_type").notNull(), // 'achievement', 'milestone', 'special', 'expert'
  category: text("category"), // 'contribution', 'engagement', 'expertise', 'community'
  criteria: jsonb("criteria").notNull(), // Requirements to earn
  points: integer("points").default(0),
  rarity: text("rarity").default('common'), // 'common', 'rare', 'epic', 'legendary'
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

export const userBadges = pgTable("user_badges", {
  id: serial("id").primaryKey(),
  userId: text("user_id").notNull(),
  badgeId: text("badge_id").notNull(),
  earnedAt: timestamp("earned_at").defaultNow(),
  progress: jsonb("progress"), // Progress tracking
  isDisplayed: boolean("is_displayed").default(true)
});

export const userPoints = pgTable("user_points", {
  id: serial("id").primaryKey(),
  userId: text("user_id").notNull().unique(),
  totalPoints: integer("total_points").default(0),
  levelPoints: integer("level_points").default(0), // Points in current level
  level: integer("level").default(1),
  rank: integer("rank"),
  streak: integer("streak").default(0), // Consecutive days active
  lastActivityDate: timestamp("last_activity_date"),
  metadata: jsonb("metadata"),
  updatedAt: timestamp("updated_at").defaultNow()
});

export const pointTransactions = pgTable("point_transactions", {
  id: serial("id").primaryKey(),
  transactionId: text("transaction_id").notNull().unique(),
  userId: text("user_id").notNull(),
  points: integer("points").notNull(),
  action: text("action").notNull(), // 'post', 'comment', 'like', 'share', 'achievement', 'daily_login'
  reason: text("reason"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

export const leaderboards = pgTable("leaderboards", {
  id: serial("id").primaryKey(),
  leaderboardId: text("leaderboard_id").notNull().unique(),
  name: text("name").notNull(),
  description: text("description"),
  leaderboardType: text("leaderboard_type").notNull(), // 'global', 'weekly', 'monthly', 'community'
  category: text("category"), // 'points', 'contributions', 'engagement'
  timeframe: text("timeframe"), // 'all_time', 'monthly', 'weekly', 'daily'
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// Virtual Events and Webinars
export const events = pgTable("events", {
  id: serial("id").primaryKey(),
  eventId: text("event_id").notNull().unique(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  eventType: text("event_type").notNull(), // 'webinar', 'workshop', 'conference', 'meetup', 'ama'
  category: text("category"), // 'wellness', 'agriculture', 'ai_ethics', 'community'
  organizerId: text("organizer_id").notNull(),
  startTime: timestamp("start_time").notNull(),
  endTime: timestamp("end_time").notNull(),
  timezone: text("timezone").default('UTC'),
  capacity: integer("capacity"),
  registeredCount: integer("registered_count").default(0),
  attendanceCount: integer("attendance_count").default(0),
  isVirtual: boolean("is_virtual").default(true),
  location: text("location"), // Physical location or virtual platform
  meetingUrl: text("meeting_url"), // Video conference link
  chatRoomId: text("chat_room_id"), // Associated chat room
  status: text("status").default('upcoming'), // 'upcoming', 'live', 'completed', 'cancelled'
  recordingUrl: text("recording_url"),
  materials: jsonb("materials"), // Slides, resources, etc.
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

export const eventRegistrations = pgTable("event_registrations", {
  id: serial("id").primaryKey(),
  registrationId: text("registration_id").notNull().unique(),
  eventId: text("event_id").notNull(),
  userId: text("user_id").notNull(),
  registrationStatus: text("registration_status").default('registered'), // 'registered', 'attended', 'cancelled', 'no_show'
  reminderSent: boolean("reminder_sent").default(false),
  feedback: text("feedback"),
  rating: integer("rating"), // 1-5 stars
  metadata: jsonb("metadata"),
  registeredAt: timestamp("registered_at").defaultNow()
});

export const eventSpeakers = pgTable("event_speakers", {
  id: serial("id").primaryKey(),
  eventId: text("event_id").notNull(),
  userId: text("user_id").notNull(),
  role: text("role").default('speaker'), // 'speaker', 'moderator', 'panelist', 'host'
  bio: text("bio"),
  topic: text("topic"), // What they're speaking about
  order: integer("order"), // Speaking order
  metadata: jsonb("metadata")
});

// Feedback Collection System
export const feedbackSubmissions = pgTable("feedback_submissions", {
  id: serial("id").primaryKey(),
  feedbackId: text("feedback_id").notNull().unique(),
  userId: text("user_id"),
  feedbackType: text("feedback_type").notNull(), // 'bug', 'feature_request', 'improvement', 'general', 'complaint'
  category: text("category"), // 'platform', 'ai_features', 'community', 'ui_ux', 'performance'
  subject: text("subject").notNull(),
  description: text("description").notNull(),
  priority: text("priority").default('medium'), // 'low', 'medium', 'high', 'critical'
  status: text("status").default('submitted'), // 'submitted', 'under_review', 'planned', 'in_progress', 'completed', 'rejected'
  sentiment: text("sentiment"), // 'positive', 'neutral', 'negative' (NLP-analyzed)
  topics: jsonb("topics"), // Auto-extracted topics via NLP
  relatedFeature: text("related_feature"),
  attachments: jsonb("attachments"),
  votes: integer("votes").default(0), // Community voting
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

export const feedbackResponses = pgTable("feedback_responses", {
  id: serial("id").primaryKey(),
  responseId: text("response_id").notNull().unique(),
  feedbackId: text("feedback_id").notNull(),
  responderId: text("responder_id").notNull(),
  responseType: text("response_type").default('comment'), // 'comment', 'status_update', 'resolution'
  content: text("content").notNull(),
  isOfficial: boolean("is_official").default(false), // From platform team
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

export const feedbackCategories = pgTable("feedback_categories", {
  id: serial("id").primaryKey(),
  categoryId: text("category_id").notNull().unique(),
  name: text("name").notNull(),
  description: text("description"),
  parentCategoryId: text("parent_category_id"), // For subcategories
  submissionCount: integer("submission_count").default(0),
  metadata: jsonb("metadata")
});

export const feedbackAnalytics = pgTable("feedback_analytics", {
  id: serial("id").primaryKey(),
  analyticsId: text("analytics_id").notNull().unique(),
  timeframe: text("timeframe").notNull(), // 'daily', 'weekly', 'monthly'
  date: timestamp("date").notNull(),
  totalSubmissions: integer("total_submissions").default(0),
  byType: jsonb("by_type"), // Count by feedback type
  byCategory: jsonb("by_category"), // Count by category
  bySentiment: jsonb("by_sentiment"), // Count by sentiment
  topTopics: jsonb("top_topics"), // Most common topics
  averageResponseTime: real("average_response_time"), // In hours
  resolutionRate: real("resolution_rate"), // Percentage
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

// AI-Curated Content Sections
export const curatedContentSections = pgTable("curated_content_sections", {
  id: serial("id").primaryKey(),
  sectionId: text("section_id").notNull().unique(),
  name: text("name").notNull(),
  description: text("description"),
  sectionType: text("section_type").notNull(), // 'music', 'ads', 'trending', 'news', 'opportunities', 'wellness'
  displayOrder: integer("display_order").default(0),
  isActive: boolean("is_active").default(true),
  aiCurationEnabled: boolean("ai_curation_enabled").default(true),
  refreshInterval: integer("refresh_interval").default(3600), // Seconds
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

export const curatedContent = pgTable("curated_content", {
  id: serial("id").primaryKey(),
  curatedId: text("curated_id").notNull().unique(),
  sectionId: text("section_id").notNull(),
  contentId: text("content_id"), // Link to content table
  title: text("title").notNull(),
  description: text("description"),
  contentType: text("content_type").notNull(), // 'music_playlist', 'ad', 'article', 'video', 'opportunity'
  contentData: jsonb("content_data").notNull(), // The actual content
  curatedBy: text("curated_by").default('ai'), // 'ai', 'human', 'hybrid'
  relevanceScore: real("relevance_score"), // AI-calculated relevance
  trendingScore: real("trending_score"), // Trending algorithm score
  personalizedFor: text("personalized_for"), // User ID if personalized
  displayOrder: integer("display_order").default(0),
  impressions: integer("impressions").default(0),
  clicks: integer("clicks").default(0),
  engagement: real("engagement"), // CTR or engagement metric
  expiresAt: timestamp("expires_at"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

export const contentRecommendations = pgTable("content_recommendations", {
  id: serial("id").primaryKey(),
  recommendationId: text("recommendation_id").notNull().unique(),
  userId: text("user_id").notNull(),
  contentId: text("content_id").notNull(),
  recommendationType: text("recommendation_type").notNull(), // 'personalized', 'trending', 'similar', 'collaborative'
  score: real("score").notNull(),
  reasoning: text("reasoning"), // Why this was recommended
  algorithm: text("algorithm"), // Which algorithm generated this
  wasClicked: boolean("was_clicked").default(false),
  wasLiked: boolean("was_liked"),
  userFeedback: text("user_feedback"), // 'relevant', 'not_relevant', 'offensive'
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

// Notifications System
export const notifications = pgTable("notifications", {
  id: serial("id").primaryKey(),
  notificationId: text("notification_id").notNull().unique(),
  userId: text("user_id").notNull(),
  notificationType: text("notification_type").notNull(), // 'event_reminder', 'feedback_response', 'badge_earned', 'chat_message', 'achievement'
  title: text("title").notNull(),
  message: text("message").notNull(),
  actionUrl: text("action_url"),
  priority: text("priority").default('normal'), // 'low', 'normal', 'high', 'urgent'
  isRead: boolean("is_read").default(false),
  readAt: timestamp("read_at"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

// Add relation types
export const chatRoomsRelations = relations(chatRooms, ({ one, many }) => ({
  creator: one(users, {
    fields: [chatRooms.creatorId],
    references: [users.userId]
  }),
  messages: many(chatMessages),
  participants: many(chatParticipants)
}));

export const badgesRelations = relations(badges, ({ many }) => ({
  userBadges: many(userBadges)
}));

export const eventsRelations = relations(events, ({ one, many }) => ({
  organizer: one(users, {
    fields: [events.organizerId],
    references: [users.userId]
  }),
  registrations: many(eventRegistrations),
  speakers: many(eventSpeakers)
}));

export const feedbackSubmissionsRelations = relations(feedbackSubmissions, ({ one, many }) => ({
  user: one(users, {
    fields: [feedbackSubmissions.userId],
    references: [users.userId]
  }),
  responses: many(feedbackResponses)
}));

// Add types
export type ChatRoom = typeof chatRooms.$inferSelect;
export type InsertChatRoom = typeof chatRooms.$inferInsert;
export type ChatMessage = typeof chatMessages.$inferSelect;
export type InsertChatMessage = typeof chatMessages.$inferInsert;
export type ChatParticipant = typeof chatParticipants.$inferSelect;
export type InsertChatParticipant = typeof chatParticipants.$inferInsert;
export type Badge = typeof badges.$inferSelect;
export type InsertBadge = typeof badges.$inferInsert;
export type UserBadge = typeof userBadges.$inferSelect;
export type InsertUserBadge = typeof userBadges.$inferInsert;
export type UserPoints = typeof userPoints.$inferSelect;
export type InsertUserPoints = typeof userPoints.$inferInsert;
export type PointTransaction = typeof pointTransactions.$inferSelect;
export type InsertPointTransaction = typeof pointTransactions.$inferInsert;
export type Leaderboard = typeof leaderboards.$inferSelect;
export type InsertLeaderboard = typeof leaderboards.$inferInsert;
export type Event = typeof events.$inferSelect;
export type InsertEvent = typeof events.$inferInsert;
export type EventRegistration = typeof eventRegistrations.$inferSelect;
export type InsertEventRegistration = typeof eventRegistrations.$inferInsert;
export type EventSpeaker = typeof eventSpeakers.$inferSelect;
export type InsertEventSpeaker = typeof eventSpeakers.$inferInsert;
export type FeedbackSubmission = typeof feedbackSubmissions.$inferSelect;
export type InsertFeedbackSubmission = typeof feedbackSubmissions.$inferInsert;
export type FeedbackResponse = typeof feedbackResponses.$inferSelect;
export type InsertFeedbackResponse = typeof feedbackResponses.$inferInsert;
export type FeedbackCategory = typeof feedbackCategories.$inferSelect;
export type InsertFeedbackCategory = typeof feedbackCategories.$inferInsert;
export type FeedbackAnalytics = typeof feedbackAnalytics.$inferSelect;
export type InsertFeedbackAnalytics = typeof feedbackAnalytics.$inferInsert;
export type CuratedContentSection = typeof curatedContentSections.$inferSelect;
export type InsertCuratedContentSection = typeof curatedContentSections.$inferInsert;
export type CuratedContent = typeof curatedContent.$inferSelect;
export type InsertCuratedContent = typeof curatedContent.$inferInsert;
export type ContentRecommendation = typeof contentRecommendations.$inferSelect;
export type InsertContentRecommendation = typeof contentRecommendations.$inferInsert;
export type Notification = typeof notifications.$inferSelect;
export type InsertNotification = typeof notifications.$inferInsert;

// Anonymous Session Management (Privacy-first)
export const anonymousSessions = pgTable("anonymous_sessions", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull().unique(),
  preferences: jsonb("preferences").default({}),
  interactionCount: integer("interaction_count").default(0),
  createdAt: timestamp("created_at").defaultNow(),
  lastActive: timestamp("last_active").defaultNow()
});

// AI Memory Systems - Episodic Memory
export const aiMemoryEpisodes = pgTable("ai_memory_episodes", {
  id: serial("id").primaryKey(),
  episodeId: text("episode_id").notNull().unique(),
  sessionId: text("session_id"),
  input: text("input").notNull(),
  output: text("output").notNull(),
  mode: text("mode").notNull(),
  confidence: real("confidence"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

// AI Memory Systems - Long-term Memory
export const aiMemoryLongterm = pgTable("ai_memory_longterm", {
  id: serial("id").primaryKey(),
  memoryKey: text("memory_key").notNull().unique(),
  value: jsonb("value").notNull(),
  accessCount: integer("access_count").default(0),
  lastAccessed: timestamp("last_accessed"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow()
});

// AI Memory Systems - Contextual Memory
export const aiMemoryContextual = pgTable("ai_memory_contextual", {
  id: serial("id").primaryKey(),
  contextId: text("context_id").notNull().unique(),
  content: text("content").notNull(),
  embedding: jsonb("embedding"), // Store embedding vector as JSON
  similarityScore: real("similarity_score"),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow()
});

// AI Processing Queue for Background Tasks
export const aiProcessingQueue = pgTable("ai_processing_queue", {
  id: serial("id").primaryKey(),
  jobId: text("job_id").notNull().unique(),
  taskType: text("task_type").notNull(), // 'content_generation', 'memory_consolidation', 'trend_analysis'
  payload: jsonb("payload").notNull(),
  status: text("status").notNull().default('pending'), // 'pending', 'processing', 'completed', 'failed'
  priority: integer("priority").default(0),
  attempts: integer("attempts").default(0),
  result: jsonb("result"),
  error: text("error"),
  createdAt: timestamp("created_at").defaultNow(),
  processedAt: timestamp("processed_at")
});

// Export types for TypeScript
export type AnonymousSession = typeof anonymousSessions.$inferSelect;
export type InsertAnonymousSession = typeof anonymousSessions.$inferInsert;
export type AiMemoryEpisode = typeof aiMemoryEpisodes.$inferSelect;
export type InsertAiMemoryEpisode = typeof aiMemoryEpisodes.$inferInsert;
export type AiMemoryLongterm = typeof aiMemoryLongterm.$inferSelect;
export type InsertAiMemoryLongterm = typeof aiMemoryLongterm.$inferInsert;
export type AiMemoryContextual = typeof aiMemoryContextual.$inferSelect;
export type InsertAiMemoryContextual = typeof aiMemoryContextual.$inferInsert;
export type AiProcessingQueue = typeof aiProcessingQueue.$inferSelect;
export type InsertAiProcessingQueue = typeof aiProcessingQueue.$inferInsert;
