import { Request, Response } from 'express';
import { db } from '../../lib/server/db';
import { communities, communityMembers, communityPosts, InsertCommunity, InsertCommunityMember, InsertCommunityPost } from '../../../shared/schema';
import { eq, and } from 'drizzle-orm';

export async function createCommunity(req: Request, res: Response) {
  try {
    const { name, description, type, category, isPrivate, creatorId, rules } = req.body;

    if (!name || !type || !creatorId) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const communityId = `community_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const newCommunity: InsertCommunity = {
      communityId,
      name,
      description,
      type,
      category,
      isPrivate: isPrivate || false,
      creatorId,
      rules: rules || {},
      metadata: { createdAt: new Date().toISOString() }
    };

    const [community] = await db.insert(communities).values(newCommunity).returning();

    await db.insert(communityMembers).values({
      communityId,
      userId: creatorId,
      role: 'admin',
      status: 'active'
    });

    res.status(201).json({ success: true, community });
  } catch (error) {
    console.error('Error creating community:', error);
    res.status(500).json({ error: 'Failed to create community' });
  }
}

export async function joinCommunity(req: Request, res: Response) {
  try {
    const { communityId } = req.params;
    const { userId } = req.body;

    if (!userId) {
      return res.status(400).json({ error: 'User ID is required' });
    }

    const existing = await db.select().from(communityMembers)
      .where(and(
        eq(communityMembers.communityId, communityId),
        eq(communityMembers.userId, userId)
      ))
      .limit(1);

    if (existing.length > 0) {
      return res.status(400).json({ error: 'Already a member of this community' });
    }

    const [membership] = await db.insert(communityMembers).values({
      communityId,
      userId,
      role: 'member',
      status: 'active'
    }).returning();

    const [community] = await db.select().from(communities)
      .where(eq(communities.communityId, communityId))
      .limit(1);

    if (community) {
      await db.update(communities)
        .set({ memberCount: (community.memberCount || 0) + 1 })
        .where(eq(communities.communityId, communityId));
    }

    res.status(201).json({ success: true, membership });
  } catch (error) {
    console.error('Error joining community:', error);
    res.status(500).json({ error: 'Failed to join community' });
  }
}

export async function getCommunities(req: Request, res: Response) {
  try {
    const { type, category } = req.query;

    let query = db.select().from(communities);

    if (type) {
      query = query.where(eq(communities.type, type as string)) as any;
    }

    if (category) {
      query = query.where(eq(communities.category, category as string)) as any;
    }

    const allCommunities = await query.limit(100);

    res.status(200).json({ success: true, communities: allCommunities });
  } catch (error) {
    console.error('Error fetching communities:', error);
    res.status(500).json({ error: 'Failed to fetch communities' });
  }
}

export async function createPost(req: Request, res: Response) {
  try {
    const { communityId } = req.params;
    const { userId, content, postType } = req.body;

    if (!userId || !content) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const postId = `post_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const newPost: InsertCommunityPost = {
      postId,
      communityId,
      userId,
      content,
      postType: postType || 'discussion',
      metadata: { createdAt: new Date().toISOString() }
    };

    const [post] = await db.insert(communityPosts).values(newPost).returning();

    res.status(201).json({ success: true, post });
  } catch (error) {
    console.error('Error creating post:', error);
    res.status(500).json({ error: 'Failed to create post' });
  }
}

export async function getPosts(req: Request, res: Response) {
  try {
    const { communityId } = req.params;

    const posts = await db.select()
      .from(communityPosts)
      .where(eq(communityPosts.communityId, communityId))
      .limit(100);

    res.status(200).json({ success: true, posts });
  } catch (error) {
    console.error('Error fetching posts:', error);
    res.status(500).json({ error: 'Failed to fetch posts' });
  }
}
