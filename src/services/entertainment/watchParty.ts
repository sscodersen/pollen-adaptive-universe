import { personalizationEngine } from '../personalizationEngine';

export interface WatchParty {
  id: string;
  contentId: string;
  contentTitle: string;
  contentType: 'movie' | 'series' | 'documentary';
  hostId: string;
  hostName: string;
  scheduledTime: string;
  participants: PartyParticipant[];
  maxParticipants: number;
  status: 'scheduled' | 'live' | 'ended';
  chatMessages: ChatMessage[];
  currentTimestamp: number;
  createdAt: string;
}

export interface PartyParticipant {
  id: string;
  name: string;
  avatar: string;
  joinedAt: string;
  isHost: boolean;
}

export interface ChatMessage {
  id: string;
  userId: string;
  userName: string;
  message: string;
  timestamp: string;
  reactions: Reaction[];
}

export interface Reaction {
  emoji: string;
  users: string[];
}

export interface CommunityRating {
  contentId: string;
  averageRating: number;
  totalRatings: number;
  distribution: Record<number, number>;
  reviews: UserReview[];
  trending: boolean;
}

export interface UserReview {
  id: string;
  userId: string;
  userName: string;
  rating: number;
  review: string;
  helpful: number;
  createdAt: string;
  verified: boolean;
}

class WatchPartyService {
  private readonly STORAGE_KEY = 'watch_parties';
  private readonly RATINGS_KEY = 'community_ratings';

  createWatchParty(
    contentId: string,
    contentTitle: string,
    contentType: 'movie' | 'series' | 'documentary',
    hostId: string,
    hostName: string,
    scheduledTime: string,
    maxParticipants: number = 10
  ): WatchParty {
    const party: WatchParty = {
      id: `party_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      contentId,
      contentTitle,
      contentType,
      hostId,
      hostName,
      scheduledTime,
      participants: [{
        id: hostId,
        name: hostName,
        avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${hostId}`,
        joinedAt: new Date().toISOString(),
        isHost: true
      }],
      maxParticipants,
      status: 'scheduled',
      chatMessages: [],
      currentTimestamp: 0,
      createdAt: new Date().toISOString()
    };

    const parties = this.getAllParties();
    parties.push(party);
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(parties));
    }

    personalizationEngine.trackBehavior({
      action: 'generate',
      contentId: party.id,
      contentType: 'entertainment',
      metadata: { type: 'watch_party', contentType }
    });

    return party;
  }

  joinParty(partyId: string, userId: string, userName: string): boolean {
    const parties = this.getAllParties();
    const party = parties.find(p => p.id === partyId);

    if (!party) return false;
    if (party.participants.length >= party.maxParticipants) return false;
    if (party.participants.some(p => p.id === userId)) return true;

    party.participants.push({
      id: userId,
      name: userName,
      avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${userId}`,
      joinedAt: new Date().toISOString(),
      isHost: false
    });

    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(parties));
    }

    personalizationEngine.trackBehavior({
      action: 'view',
      contentId: partyId,
      contentType: 'entertainment',
      metadata: { type: 'watch_party' }
    });

    return true;
  }

  leaveParty(partyId: string, userId: string): void {
    const parties = this.getAllParties();
    const party = parties.find(p => p.id === partyId);

    if (!party) return;

    party.participants = party.participants.filter(p => p.id !== userId);

    if (party.participants.length === 0) {
      const index = parties.findIndex(p => p.id === partyId);
      parties.splice(index, 1);
    }

    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(parties));
    }
  }

  sendMessage(partyId: string, userId: string, userName: string, message: string): void {
    const parties = this.getAllParties();
    const party = parties.find(p => p.id === partyId);

    if (!party) return;

    const chatMessage: ChatMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      userId,
      userName,
      message,
      timestamp: new Date().toISOString(),
      reactions: []
    };

    party.chatMessages.push(chatMessage);
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(parties));
    }
  }

  updateTimestamp(partyId: string, timestamp: number): void {
    const parties = this.getAllParties();
    const party = parties.find(p => p.id === partyId);

    if (!party) return;

    party.currentTimestamp = timestamp;
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(parties));
    }
  }

  startParty(partyId: string): void {
    const parties = this.getAllParties();
    const party = parties.find(p => p.id === partyId);

    if (!party) return;

    party.status = 'live';
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(parties));
    }
  }

  endParty(partyId: string): void {
    const parties = this.getAllParties();
    const party = parties.find(p => p.id === partyId);

    if (!party) return;

    party.status = 'ended';
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(parties));
    }
  }

  getAllParties(): WatchParty[] {
    if (typeof window === 'undefined' || !window.localStorage) {
      return [];
    }
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  getActiveParties(): WatchParty[] {
    return this.getAllParties().filter(p => p.status === 'scheduled' || p.status === 'live');
  }

  getPartiesForContent(contentId: string): WatchParty[] {
    return this.getAllParties().filter(p => p.contentId === contentId);
  }

  submitRating(
    contentId: string,
    userId: string,
    userName: string,
    rating: number,
    review: string
  ): void {
    const ratings = this.getCommunityRatings();
    let contentRating = ratings.find(r => r.contentId === contentId);

    if (!contentRating) {
      contentRating = {
        contentId,
        averageRating: 0,
        totalRatings: 0,
        distribution: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
        reviews: [],
        trending: false
      };
      ratings.push(contentRating);
    }

    const existingReview = contentRating.reviews.find(r => r.userId === userId);
    if (existingReview) {
      contentRating.distribution[existingReview.rating]--;
      existingReview.rating = rating;
      existingReview.review = review;
      existingReview.createdAt = new Date().toISOString();
    } else {
      const userReview: UserReview = {
        id: `review_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        userId,
        userName,
        rating,
        review,
        helpful: 0,
        createdAt: new Date().toISOString(),
        verified: true
      };
      contentRating.reviews.push(userReview);
      contentRating.totalRatings++;
    }

    contentRating.distribution[rating]++;

    const totalScore = Object.entries(contentRating.distribution).reduce(
      (sum, [stars, count]) => sum + parseInt(stars) * count,
      0
    );
    contentRating.averageRating = totalScore / contentRating.totalRatings;

    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.RATINGS_KEY, JSON.stringify(ratings));
    }

    personalizationEngine.trackBehavior({
      action: 'like',
      contentId,
      contentType: 'entertainment',
      metadata: { rating, hasReview: !!review }
    });
  }

  getCommunityRating(contentId: string): CommunityRating | null {
    const ratings = this.getCommunityRatings();
    return ratings.find(r => r.contentId === contentId) || null;
  }

  getCommunityRatings(): CommunityRating[] {
    if (typeof window === 'undefined' || !window.localStorage) {
      return [];
    }
    try {
      const stored = localStorage.getItem(this.RATINGS_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  markReviewHelpful(contentId: string, reviewId: string): void {
    const ratings = this.getCommunityRatings();
    const contentRating = ratings.find(r => r.contentId === contentId);

    if (!contentRating) return;

    const review = contentRating.reviews.find(r => r.id === reviewId);
    if (review) {
      review.helpful++;
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.setItem(this.RATINGS_KEY, JSON.stringify(ratings));
      }
    }
  }
}

export const watchPartyService = new WatchPartyService();
