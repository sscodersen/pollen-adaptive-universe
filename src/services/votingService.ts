import { storageService } from './storageService';

interface VoteData {
  postId: string;
  voteType: 'up' | 'down' | null;
  timestamp: number;
}

interface PostVotes {
  upvotes: number;
  downvotes: number;
  score: number;
}

class VotingService {
  private readonly USER_VOTES_KEY = 'pollen_user_votes';

  /**
   * Toggle a vote on a post
   * @param postId - The post ID
   * @param voteType - The type of vote (up or down)
   * @param serverVotes - The pure server vote counts (user's vote already removed by caller)
   * @returns Updated vote counts with user's new vote applied
   */
  async vote(postId: string, voteType: 'up' | 'down', serverVotes: PostVotes): Promise<PostVotes> {
    const currentUserVote = this.getUserVote(postId);

    // If clicking the same vote type, remove the vote
    if (currentUserVote === voteType) {
      return this.removeVote(postId, serverVotes);
    }

    // Start with server votes as baseline (caller already backed out previous vote)
    let upvotes = serverVotes.upvotes;
    let downvotes = serverVotes.downvotes;

    // Add new user vote (previous vote already removed by caller)
    if (voteType === 'up') {
      upvotes++;
    } else {
      downvotes++;
    }

    const score = this.calculateScore(upvotes, downvotes);

    this.saveUserVote(postId, voteType);

    return { upvotes, downvotes, score };
  }

  /**
   * Remove a user's vote from a post
   * @param postId - The post ID
   * @param serverVotes - The pure server vote counts (user's vote already removed by caller)
   * @returns Server vote counts unchanged
   */
  async removeVote(postId: string, serverVotes: PostVotes): Promise<PostVotes> {
    // The serverVotes passed in are already the pure server counts
    // (handleVote backed out the user's vote before calling this)
    // Just remove the user vote record and return server counts as-is
    
    this.removeUserVote(postId);

    const score = this.calculateScore(serverVotes.upvotes, serverVotes.downvotes);

    return { 
      upvotes: serverVotes.upvotes, 
      downvotes: serverVotes.downvotes, 
      score 
    };
  }

  /**
   * Get the user's vote for a post
   * @param postId - The post ID
   * @returns The user's vote ('up', 'down', or null)
   */
  getUserVote(postId: string): 'up' | 'down' | null {
    try {
      const userVotes = JSON.parse(localStorage.getItem(this.USER_VOTES_KEY) || '{}');
      return userVotes[postId] || null;
    } catch {
      return null;
    }
  }

  private saveUserVote(postId: string, voteType: 'up' | 'down'): void {
    try {
      const userVotes = JSON.parse(localStorage.getItem(this.USER_VOTES_KEY) || '{}');
      userVotes[postId] = voteType;
      localStorage.setItem(this.USER_VOTES_KEY, JSON.stringify(userVotes));
    } catch (error) {
      console.error('Error saving user vote:', error);
    }
  }

  private removeUserVote(postId: string): void {
    try {
      const userVotes = JSON.parse(localStorage.getItem(this.USER_VOTES_KEY) || '{}');
      delete userVotes[postId];
      localStorage.setItem(this.USER_VOTES_KEY, JSON.stringify(userVotes));
    } catch (error) {
      console.error('Error removing user vote:', error);
    }
  }

  private calculateScore(upvotes: number, downvotes: number): number {
    const total = upvotes + downvotes;
    if (total === 0) return 0;
    
    // Wilson score interval for better ranking
    const ratio = upvotes / total;
    const confidence = Math.min(total / 10, 1);
    
    return Math.round((ratio * 10 * confidence + upvotes - downvotes) * 10) / 10;
  }

  /**
   * Enrich posts with user vote data while preserving server vote counts
   * @param posts - Array of posts to enrich
   * @returns Posts with user vote information added
   */
  enrichPostsWithVotes<T extends { id: string; votes?: any }>(posts: T[]): T[] {
    return posts.map(post => {
      const serverVotes = post.votes || { upvotes: 0, downvotes: 0, score: 0 };
      const userVote = this.getUserVote(post.id);

      // Calculate current votes by applying user's vote to server totals
      let upvotes = serverVotes.upvotes;
      let downvotes = serverVotes.downvotes;

      if (userVote === 'up') {
        upvotes++;
      } else if (userVote === 'down') {
        downvotes++;
      }

      const score = this.calculateScore(upvotes, downvotes);

      return {
        ...post,
        votes: {
          upvotes,
          downvotes,
          score,
          userVote
        }
      };
    });
  }
}

export const votingService = new VotingService();
