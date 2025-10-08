import { Request, Response } from 'express';
import { db } from '../../lib/server/db';
import { users, communities, communityMembers, userMatchingScores } from '../../../shared/schema';
import { eq, and } from 'drizzle-orm';

interface UserProfile {
  userId: string;
  interests: string[];
  preferences: any;
  challenges?: string[];
}

export class CommunityMatchingService {
  async calculateMatchScore(user1: UserProfile, user2: UserProfile): Promise<number> {
    const interests1 = new Set(user1.interests);
    const interests2 = new Set(user2.interests);
    
    const commonInterests = [...interests1].filter(interest => interests2.has(interest));
    const unionSize = new Set([...interests1, ...interests2]).size;
    
    const jaccardSimilarity = unionSize > 0 ? commonInterests.length / unionSize : 0;
    
    const baseScore = jaccardSimilarity * 0.7;
    const challengeBonus = this.calculateChallengeBonus(user1, user2) * 0.3;
    
    return Math.min(baseScore + challengeBonus, 1);
  }

  private calculateChallengeBonus(user1: UserProfile, user2: UserProfile): number {
    const challenges1 = new Set(user1.challenges || []);
    const challenges2 = new Set(user2.challenges || []);
    
    const commonChallenges = [...challenges1].filter(c => challenges2.has(c));
    
    return commonChallenges.length > 0 ? 0.5 : 0;
  }

  async findMatchingUsers(userId: string, limit: number = 10) {
    const userRecord = await db.select().from(users).where(eq(users.userId, userId)).limit(1);
    
    if (!userRecord.length) {
      return [];
    }

    const currentUser = userRecord[0];
    const profileData = currentUser.profileData as any || {};
    const userProfile: UserProfile = {
      userId: currentUser.userId,
      interests: profileData.interests || [],
      preferences: profileData.preferences || {},
      challenges: profileData.challenges || []
    };

    const allUsers = await db.select().from(users).limit(100);
    const matches: any[] = [];

    for (const otherUser of allUsers) {
      if (otherUser.userId === userId) continue;

      const otherProfileData = otherUser.profileData as any || {};
      const otherProfile: UserProfile = {
        userId: otherUser.userId,
        interests: otherProfileData.interests || [],
        preferences: otherProfileData.preferences || {},
        challenges: otherProfileData.challenges || []
      };

      const matchScore = await this.calculateMatchScore(userProfile, otherProfile);

      if (matchScore > 0.3) {
        matches.push({
          user: otherUser,
          matchScore,
          commonInterests: userProfile.interests.filter(i => otherProfile.interests.includes(i))
        });

        const existingMatch = await db.select().from(userMatchingScores)
          .where(and(
            eq(userMatchingScores.userId1, userId),
            eq(userMatchingScores.userId2, otherUser.userId)
          ))
          .limit(1);

        if (existingMatch.length === 0) {
          await db.insert(userMatchingScores).values({
            userId1: userId,
            userId2: otherUser.userId,
            matchScore,
            commonInterests: userProfile.interests.filter(i => otherProfile.interests.includes(i))
          });
        }
      }
    }

    return matches
      .sort((a, b) => b.matchScore - a.matchScore)
      .slice(0, limit);
  }

  async suggestCommunities(userId: string) {
    const userRecord = await db.select().from(users).where(eq(users.userId, userId)).limit(1);
    
    if (!userRecord.length) {
      return [];
    }

    const userProfileData = userRecord[0].profileData as any || {};
    const userInterests = userProfileData.interests || [];
    const allCommunities = await db.select().from(communities).limit(100);

    const suggestions = allCommunities.map(community => {
      const communityMetadata = community.metadata as any || {};
      const communityTags = communityMetadata.tags || [];
      const relevanceScore = this.calculateRelevance(userInterests, communityTags);

      return {
        community,
        relevanceScore,
        matchingInterests: userInterests.filter(i => communityTags.includes(i))
      };
    });

    return suggestions
      .filter(s => s.relevanceScore > 0.2)
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 10);
  }

  private calculateRelevance(userInterests: string[], communityTags: string[]): number {
    const userSet = new Set(userInterests);
    const communitySet = new Set(communityTags);
    
    const commonTags = [...userSet].filter(tag => communitySet.has(tag));
    const unionSize = new Set([...userSet, ...communitySet]).size;
    
    return unionSize > 0 ? commonTags.length / unionSize : 0;
  }
}

export const matchingService = new CommunityMatchingService();

export async function findMatches(req: Request, res: Response) {
  try {
    const { userId } = req.params;
    const { limit = 10 } = req.query;

    const matches = await matchingService.findMatchingUsers(userId, Number(limit));

    res.status(200).json({ success: true, matches });
  } catch (error) {
    console.error('Error finding matches:', error);
    res.status(500).json({ error: 'Failed to find matching users' });
  }
}

export async function suggestCommunities(req: Request, res: Response) {
  try {
    const { userId } = req.params;

    const suggestions = await matchingService.suggestCommunities(userId);

    res.status(200).json({ success: true, suggestions });
  } catch (error) {
    console.error('Error suggesting communities:', error);
    res.status(500).json({ error: 'Failed to suggest communities' });
  }
}
