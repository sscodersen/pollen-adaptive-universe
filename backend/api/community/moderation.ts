import { Request, Response } from 'express';
import { db } from '../../lib/server/db';
import { moderationActions, communityMembers, communityPosts, InsertModerationAction } from '../../../shared/schema';
import { eq, and } from 'drizzle-orm';

export async function moderateUser(req: Request, res: Response) {
  try {
    const { communityId } = req.params;
    const { moderatorId, targetUserId, actionType, reason, duration } = req.body;

    if (!moderatorId || !targetUserId || !actionType || !reason) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const moderator = await db.select().from(communityMembers)
      .where(and(
        eq(communityMembers.communityId, communityId),
        eq(communityMembers.userId, moderatorId)
      ))
      .limit(1);

    if (!moderator.length || (moderator[0].role !== 'moderator' && moderator[0].role !== 'admin')) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    const actionId = `mod_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const action: InsertModerationAction = {
      actionId,
      communityId,
      moderatorId,
      targetUserId,
      actionType,
      reason,
      duration: duration || null,
      metadata: { executedAt: new Date().toISOString() }
    };

    const [moderationAction] = await db.insert(moderationActions).values(action).returning();

    if (actionType === 'ban' || actionType === 'mute') {
      await db.update(communityMembers)
        .set({ status: actionType === 'ban' ? 'banned' : 'muted' })
        .where(and(
          eq(communityMembers.communityId, communityId),
          eq(communityMembers.userId, targetUserId)
        ));
    }

    res.status(201).json({ success: true, action: moderationAction });
  } catch (error) {
    console.error('Error moderating user:', error);
    res.status(500).json({ error: 'Failed to moderate user' });
  }
}

export async function deletePost(req: Request, res: Response) {
  try {
    const { communityId, postId } = req.params;
    const { moderatorId, reason } = req.body;

    if (!moderatorId || !reason) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const moderator = await db.select().from(communityMembers)
      .where(and(
        eq(communityMembers.communityId, communityId),
        eq(communityMembers.userId, moderatorId)
      ))
      .limit(1);

    if (!moderator.length || (moderator[0].role !== 'moderator' && moderator[0].role !== 'admin')) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    const actionId = `mod_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    await db.insert(moderationActions).values({
      actionId,
      communityId,
      moderatorId,
      targetPostId: postId,
      actionType: 'delete_post',
      reason,
      metadata: { executedAt: new Date().toISOString() }
    });

    await db.delete(communityPosts).where(eq(communityPosts.postId, postId));

    res.status(200).json({ success: true, message: 'Post deleted successfully' });
  } catch (error) {
    console.error('Error deleting post:', error);
    res.status(500).json({ error: 'Failed to delete post' });
  }
}

export async function getModerationHistory(req: Request, res: Response) {
  try {
    const { communityId } = req.params;

    const actions = await db.select()
      .from(moderationActions)
      .where(eq(moderationActions.communityId, communityId))
      .limit(100);

    res.status(200).json({ success: true, actions });
  } catch (error) {
    console.error('Error fetching moderation history:', error);
    res.status(500).json({ error: 'Failed to fetch moderation history' });
  }
}
