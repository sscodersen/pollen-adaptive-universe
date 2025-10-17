/**
 * Anonymous Session Management for Pollen AI Platform
 * Privacy-first approach - no personal data, only session tracking
 */

import { randomBytes } from 'crypto';

export interface AnonymousSession {
  sessionId: string;
  createdAt: Date;
  lastActive: Date;
  preferences: Record<string, any>;
  interactionCount: number;
}

class SessionManager {
  private sessions: Map<string, AnonymousSession> = new Map();
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
   * Create or retrieve session
   */
  getOrCreateSession(sessionId?: string): AnonymousSession {
    // Clean up expired sessions periodically
    this.cleanupExpiredSessions();

    if (sessionId && this.sessions.has(sessionId)) {
      const session = this.sessions.get(sessionId)!;
      session.lastActive = new Date();
      return session;
    }

    // Create new session
    const newSessionId = sessionId || this.generateSessionId();
    const session: AnonymousSession = {
      sessionId: newSessionId,
      createdAt: new Date(),
      lastActive: new Date(),
      preferences: {},
      interactionCount: 0
    };

    this.sessions.set(newSessionId, session);
    return session;
  }

  /**
   * Update session activity
   */
  touchSession(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.lastActive = new Date();
      session.interactionCount++;
    }
  }

  /**
   * Update session preferences
   */
  updatePreferences(sessionId: string, preferences: Record<string, any>): void {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.preferences = { ...session.preferences, ...preferences };
    }
  }

  /**
   * Get session preferences
   */
  getPreferences(sessionId: string): Record<string, any> {
    const session = this.sessions.get(sessionId);
    return session?.preferences || {};
  }

  /**
   * Clean up expired sessions
   */
  private cleanupExpiredSessions(): void {
    const now = Date.now();
    for (const [sessionId, session] of this.sessions.entries()) {
      if (now - session.lastActive.getTime() > this.sessionTimeout) {
        this.sessions.delete(sessionId);
      }
    }
  }

  /**
   * Get session stats
   */
  getStats(): {
    totalSessions: number;
    activeSessions: number;
    totalInteractions: number;
  } {
    const now = Date.now();
    const oneHourAgo = now - (60 * 60 * 1000);
    
    let activeSessions = 0;
    let totalInteractions = 0;

    for (const session of this.sessions.values()) {
      if (session.lastActive.getTime() > oneHourAgo) {
        activeSessions++;
      }
      totalInteractions += session.interactionCount;
    }

    return {
      totalSessions: this.sessions.size,
      activeSessions,
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
export const sessionManager = new SessionManager();
