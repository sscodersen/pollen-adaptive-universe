import { personalizationEngine } from '../personalizationEngine';

export interface LeaderboardEntry {
  userId: string;
  username: string;
  avatar: string;
  points: number;
  rank: number;
  badges: Badge[];
  level: number;
  streak: number;
  contributions: number;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  earnedAt: string;
  category: string;
}

export interface UserStats {
  userId: string;
  totalPoints: number;
  level: number;
  xpToNextLevel: number;
  currentXP: number;
  badges: Badge[];
  achievements: Achievement[];
  streakDays: number;
  longestStreak: number;
  totalContributions: number;
  categories: Record<string, number>;
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  progress: number;
  target: number;
  reward: number;
  completed: boolean;
  category: string;
}

class LeaderboardService {
  private readonly LEADERBOARD_KEY = 'global_leaderboard';
  private readonly USER_STATS_KEY = 'user_stats';

  awardPoints(userId: string, points: number, action: string, category: string): void {
    const stats = this.getUserStats(userId);
    
    stats.totalPoints += points;
    stats.currentXP += points;
    stats.totalContributions++;
    
    if (!stats.categories[category]) {
      stats.categories[category] = 0;
    }
    stats.categories[category] += points;

    while (stats.currentXP >= stats.xpToNextLevel) {
      stats.currentXP -= stats.xpToNextLevel;
      stats.level++;
      stats.xpToNextLevel = this.calculateXPForLevel(stats.level + 1);
      
      this.checkLevelBadges(stats);
    }

    this.checkAchievements(stats, action);
    this.saveUserStats(stats);
    this.updateLeaderboard(userId, stats);

    personalizationEngine.trackBehavior({
      action: 'save',
      contentId: userId,
      contentType: 'educational',
      metadata: { points, action, category, newLevel: stats.level }
    });
  }

  private calculateXPForLevel(level: number): number {
    return Math.floor(100 * Math.pow(1.5, level - 1));
  }

  private checkLevelBadges(stats: UserStats): void {
    const levelBadges: Record<number, Badge> = {
      5: {
        id: 'level_5',
        name: 'Rising Star',
        description: 'Reached level 5',
        icon: '⭐',
        rarity: 'common',
        earnedAt: new Date().toISOString(),
        category: 'level'
      },
      10: {
        id: 'level_10',
        name: 'Dedicated User',
        description: 'Reached level 10',
        icon: '🌟',
        rarity: 'rare',
        earnedAt: new Date().toISOString(),
        category: 'level'
      },
      25: {
        id: 'level_25',
        name: 'Master Contributor',
        description: 'Reached level 25',
        icon: '💎',
        rarity: 'epic',
        earnedAt: new Date().toISOString(),
        category: 'level'
      },
      50: {
        id: 'level_50',
        name: 'Legend',
        description: 'Reached level 50',
        icon: '👑',
        rarity: 'legendary',
        earnedAt: new Date().toISOString(),
        category: 'level'
      }
    };

    if (levelBadges[stats.level] && !stats.badges.some(b => b.id === levelBadges[stats.level].id)) {
      stats.badges.push(levelBadges[stats.level]);
    }
  }

  private checkAchievements(stats: UserStats, action: string): void {
    stats.achievements.forEach(achievement => {
      if (!achievement.completed && achievement.progress >= achievement.target) {
        achievement.completed = true;
        stats.totalPoints += achievement.reward;
        
        const badge: Badge = {
          id: `achievement_${achievement.id}`,
          name: achievement.title,
          description: achievement.description,
          icon: '🏆',
          rarity: 'rare',
          earnedAt: new Date().toISOString(),
          category: achievement.category
        };
        stats.badges.push(badge);
      }
    });

    const contributionMilestones = [10, 50, 100, 500, 1000];
    for (const milestone of contributionMilestones) {
      if (stats.totalContributions === milestone) {
        const badge: Badge = {
          id: `contributions_${milestone}`,
          name: `${milestone} Contributions`,
          description: `Made ${milestone} contributions to the community`,
          icon: '📝',
          rarity: milestone >= 500 ? 'epic' : milestone >= 100 ? 'rare' : 'common',
          earnedAt: new Date().toISOString(),
          category: 'contribution'
        };
        if (!stats.badges.some(b => b.id === badge.id)) {
          stats.badges.push(badge);
        }
      }
    }
  }

  getUserStats(userId: string): UserStats {
    try {
      const stored = localStorage.getItem(`${this.USER_STATS_KEY}_${userId}`);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load user stats:', error);
    }

    return {
      userId,
      totalPoints: 0,
      level: 1,
      xpToNextLevel: 100,
      currentXP: 0,
      badges: [],
      achievements: this.getDefaultAchievements(),
      streakDays: 0,
      longestStreak: 0,
      totalContributions: 0,
      categories: {}
    };
  }

  private getDefaultAchievements(): Achievement[] {
    return [
      {
        id: 'first_post',
        title: 'First Steps',
        description: 'Make your first contribution',
        progress: 0,
        target: 1,
        reward: 50,
        completed: false,
        category: 'engagement'
      },
      {
        id: 'streak_7',
        title: 'Week Warrior',
        description: 'Maintain a 7-day streak',
        progress: 0,
        target: 7,
        reward: 100,
        completed: false,
        category: 'streak'
      },
      {
        id: 'social_butterfly',
        title: 'Social Butterfly',
        description: 'Interact with 10 different users',
        progress: 0,
        target: 10,
        reward: 150,
        completed: false,
        category: 'social'
      },
      {
        id: 'content_creator',
        title: 'Content Creator',
        description: 'Create 50 pieces of content',
        progress: 0,
        target: 50,
        reward: 250,
        completed: false,
        category: 'creation'
      }
    ];
  }

  private saveUserStats(stats: UserStats): void {
    localStorage.setItem(`${this.USER_STATS_KEY}_${stats.userId}`, JSON.stringify(stats));
  }

  private updateLeaderboard(userId: string, stats: UserStats): void {
    const leaderboard = this.getLeaderboard();
    const index = leaderboard.findIndex(entry => entry.userId === userId);

    const entry: LeaderboardEntry = {
      userId: stats.userId,
      username: `User ${userId.substring(0, 8)}`,
      avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${userId}`,
      points: stats.totalPoints,
      rank: 0,
      badges: stats.badges,
      level: stats.level,
      streak: stats.streakDays,
      contributions: stats.totalContributions
    };

    if (index >= 0) {
      leaderboard[index] = entry;
    } else {
      leaderboard.push(entry);
    }

    leaderboard.sort((a, b) => b.points - a.points);
    
    leaderboard.forEach((entry, index) => {
      entry.rank = index + 1;
    });

    localStorage.setItem(this.LEADERBOARD_KEY, JSON.stringify(leaderboard.slice(0, 100)));
  }

  getLeaderboard(limit: number = 50): LeaderboardEntry[] {
    try {
      const stored = localStorage.getItem(this.LEADERBOARD_KEY);
      const leaderboard = stored ? JSON.parse(stored) : [];
      return leaderboard.slice(0, limit);
    } catch {
      return [];
    }
  }

  getUserRank(userId: string): number {
    const leaderboard = this.getLeaderboard(1000);
    const entry = leaderboard.find(e => e.userId === userId);
    return entry ? entry.rank : -1;
  }

  updateStreak(userId: string): void {
    const stats = this.getUserStats(userId);
    stats.streakDays++;
    
    if (stats.streakDays > stats.longestStreak) {
      stats.longestStreak = stats.streakDays;
    }

    if (stats.streakDays === 7 || stats.streakDays === 30 || stats.streakDays === 100) {
      const badge: Badge = {
        id: `streak_${stats.streakDays}`,
        name: `${stats.streakDays} Day Streak`,
        description: `Maintained activity for ${stats.streakDays} consecutive days`,
        icon: '🔥',
        rarity: stats.streakDays >= 100 ? 'legendary' : stats.streakDays >= 30 ? 'epic' : 'rare',
        earnedAt: new Date().toISOString(),
        category: 'streak'
      };
      if (!stats.badges.some(b => b.id === badge.id)) {
        stats.badges.push(badge);
      }
    }

    this.saveUserStats(stats);
    this.updateLeaderboard(userId, stats);
  }

  getTopBadges(userId: string, limit: number = 5): Badge[] {
    const stats = this.getUserStats(userId);
    const rarityOrder = { legendary: 4, epic: 3, rare: 2, common: 1 };
    
    return stats.badges
      .sort((a, b) => rarityOrder[b.rarity] - rarityOrder[a.rarity])
      .slice(0, limit);
  }
}

export const leaderboardService = new LeaderboardService();
