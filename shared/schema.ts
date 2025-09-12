import { pgTable, serial, text, timestamp, real, integer, jsonb } from "drizzle-orm/pg-core";
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

// Relations
export const contentRelations = relations(content, ({ many }) => ({
  feedItems: many(feedItems),
  interactions: many(interactions)
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

// Types
export type Content = typeof content.$inferSelect;
export type InsertContent = typeof content.$inferInsert;
export type FeedItem = typeof feedItems.$inferSelect;
export type InsertFeedItem = typeof feedItems.$inferInsert;
export type Interaction = typeof interactions.$inferSelect;
export type InsertInteraction = typeof interactions.$inferInsert;