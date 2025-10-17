/**
 * Database-backed Anonymous Session Management
 * Production-ready with PostgreSQL persistence
 */

import { db } from './db';
import { anonymousSessions, type AnonymousSession, type InsertAnonymousSession } from '../shared/schema';
import { eq, and, gt } from 'drizzle-orm';
import { randomBytes } from 'crypto';

export class DatabaseSessionManager {
  private sessionTimeout = 24 * 60 * 60 * 1000; // 24 hours

  /**
   * Generate a new anonymous session ID
   */
  generateSessionId(): string {
    const timestamp = Date.now();
    const random = randomBytes(8).toString('hex');
    return `anon_${timestamp}_${random}`;
  }

  /**
   * Create or retrieve session from database
   */
  async getOrCreateSession(sessionId?: string): Promise<AnonymousSession> {
    // Clean up expired sessions periodically
    await this.cleanupExpiredSessions();

    if (sessionId) {
      try {
        const [session] = await db
          .select()
          .from(anonymousSessions)
          .where(eq(anonymousSessions.sessionId, sessionId))
          .limit(1);

        if (session) {
          // Update last active timestamp
          const [updated] = await db
            .update(anonymousSessions)
            .set({ lastActive: new Date() })
            .where(eq(anonymousSessions.sessionId, sessionId))
            .returning();
          
          return updated;
        }
      } catch (error) {
        console.error('Error fetching session:', error);
      }
    }

    // Create new session in database
    const newSessionId = sessionId || this.generateSessionId();
    const [newSession] = await db
      .insert(anonymousSessions)
      .values({
        sessionId: newSessionId,
        preferences: {},
        interactionCount: 0
      })
      .returning();

    return newSession;
  }

  /**
   * Update session activity
   */
  async touchSession(sessionId: string): Promise<void> {
    await db
      .update(anonymousSessions)
      .set({ 
        lastActive: new Date(),
        interactionCount: db.$count(anonymousSessions.interactionCount) + 1
      })
      .where(eq(anonymousSessions.sessionId, sessionId));
  }

  /**
   * Update session preferences
   */
  async updatePreferences(sessionId: string, preferences: Record<string, any>): Promise<void> {
    const [session] = await db
      .select()
      .from(anonymousSessions)
      .where(eq(anonymousSessions.sessionId, sessionId))
      .limit(1);

    if (session) {
      const updatedPrefs = { ...session.preferences as object, ...preferences };
      await db
        .update(anonymousSessions)
        .set({ preferences: updatedPrefs })
        .where(eq(anonymousSessions.sessionId, sessionId));
    }
  }

  /**
   * Get session preferences
   */
  async getPreferences(sessionId: string): Promise<Record<string, any>> {
    const [session] = await db
      .select()
      .from(anonymousSessions)
      .where(eq(anonymousSessions.sessionId, sessionId))
      .limit(1);

    return (session?.preferences as Record<string, any>) || {};
  }

  /**
   * Clean up expired sessions from database
   */
  private async cleanupExpiredSessions(): Promise<void> {
    const expiryTime = new Date(Date.now() - this.sessionTimeout);
    
    await db
      .delete(anonymousSessions)
      .where(gt(anonymousSessions.lastActive, expiryTime));
  }

  /**
   * Get session stats from database
   */
  async getStats(): Promise<{
    totalSessions: number;
    activeSessions: number;
    totalInteractions: number;
  }> {
    const oneHourAgo = new Date(Date.now() - (60 * 60 * 1000));
    
    const allSessions = await db.select().from(anonymousSessions);
    const activeSessions = await db
      .select()
      .from(anonymousSessions)
      .where(gt(anonymousSessions.lastActive, oneHourAgo));

    const totalInteractions = allSessions.reduce(
      (sum, session) => sum + (session.interactionCount || 0), 
      0
    );

    return {
      totalSessions: allSessions.length,
      activeSessions: activeSessions.length,
      totalInteractions
    };
  }

  /**
   * Validate session ID format
   */
  isValidSessionId(sessionId: string): boolean {
    return /^anon_\d+_[a-f0-9]{16}$/.test(sessionId);
  }
}

// Export singleton instance
export const dbSessionManager = new DatabaseSessionManager();
