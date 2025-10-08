import axios from 'axios';

const API_BASE = '/api/forum';

export interface ForumTopicCreation {
  creatorId: string;
  title: string;
  description: string;
  category: 'ai_bias' | 'privacy' | 'transparency' | 'fairness' | 'accountability' | 'safety';
  tags?: string[];
}

export interface ForumPostCreation {
  topicId: string;
  userId: string;
  content: string;
  parentPostId?: string;
  postType?: 'reply' | 'expert_opinion' | 'proposal';
  isExpertPost?: boolean;
  metadata?: Record<string, any>;
}

export interface ForumVote {
  userId: string;
  targetType: 'topic' | 'post';
  targetId: string;
  voteType: 'upvote' | 'downvote';
}

export interface GuidelineCreation {
  title: string;
  content: string;
  category: string;
  version: string;
  metadata?: Record<string, any>;
}

export interface ExpertContributionCreation {
  expertId: string;
  contributionType: 'opinion' | 'guideline' | 'research' | 'review';
  relatedTopicId?: string;
  relatedGuidelineId?: string;
  content: string;
  expertise?: string[];
  citations?: any[];
  impactScore?: number;
}

export interface ModerationActionCreation {
  moderatorId: string;
  targetType: 'topic' | 'post' | 'user';
  targetId: string;
  actionType: 'approve' | 'flag' | 'remove' | 'warn' | 'ban';
  reason: string;
  automated?: boolean;
  metadata?: Record<string, any>;
}

class EthicsForumService {
  async createTopic(topic: ForumTopicCreation) {
    const response = await axios.post(`${API_BASE}/topics`, topic);
    return response.data;
  }

  async getTopics(filters?: {
    category?: string;
    status?: string;
  }) {
    const response = await axios.get(`${API_BASE}/topics`, { params: filters });
    return response.data;
  }

  async getTopicById(topicId: string) {
    const response = await axios.get(`${API_BASE}/topics/${topicId}`);
    return response.data;
  }

  async createPost(post: ForumPostCreation) {
    const response = await axios.post(`${API_BASE}/topics/${post.topicId}/posts`, post);
    return response.data;
  }

  async getPostsByTopic(topicId: string) {
    const response = await axios.get(`${API_BASE}/topics/${topicId}/posts`);
    return response.data;
  }

  async vote(vote: ForumVote) {
    const response = await axios.post(`${API_BASE}/vote`, vote);
    return response.data;
  }

  async createGuideline(guideline: GuidelineCreation) {
    const response = await axios.post(`${API_BASE}/guidelines`, guideline);
    return response.data;
  }

  async getGuidelines(filters?: {
    category?: string;
    approvalStatus?: string;
  }) {
    const response = await axios.get(`${API_BASE}/guidelines`, { params: filters });
    return response.data;
  }

  async createExpertContribution(contribution: ExpertContributionCreation) {
    const response = await axios.post(`${API_BASE}/expert-contributions`, contribution);
    return response.data;
  }

  async getExpertContributions(filters?: {
    expertId?: string;
    contributionType?: string;
  }) {
    const response = await axios.get(`${API_BASE}/expert-contributions`, { params: filters });
    return response.data;
  }

  async createModerationAction(action: ModerationActionCreation) {
    const response = await axios.post(`${API_BASE}/moderation`, action);
    return response.data;
  }

  async getModerationActions(filters?: {
    targetType?: string;
    actionType?: string;
  }) {
    const response = await axios.get(`${API_BASE}/moderation`, { params: filters });
    return response.data;
  }
}

export const ethicsForumService = new EthicsForumService();
