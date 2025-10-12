import { personalizationEngine } from '../personalizationEngine';
import pollenAIUnified from '../pollenAIUnified';

export interface WellnessChallenge {
  id: string;
  title: string;
  description: string;
  category: 'fitness' | 'nutrition' | 'mental' | 'sleep' | 'social';
  duration: number;
  participants: number;
  goal: string;
  rewards: string[];
  startDate: string;
  endDate: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  dailyTasks: DailyTask[];
}

export interface DailyTask {
  day: number;
  task: string;
  completed: boolean;
  points: number;
}

export interface WellnessForum {
  id: string;
  title: string;
  category: string;
  posts: ForumPost[];
  members: number;
  createdAt: string;
}

export interface ForumPost {
  id: string;
  authorId: string;
  authorName: string;
  content: string;
  likes: number;
  replies: Reply[];
  createdAt: string;
  tags: string[];
}

export interface Reply {
  id: string;
  authorId: string;
  authorName: string;
  content: string;
  likes: number;
  createdAt: string;
}

export interface UserProgress {
  userId: string;
  goals: WellnessGoal[];
  achievements: Achievement[];
  totalPoints: number;
  streakDays: number;
  level: number;
}

export interface WellnessGoal {
  id: string;
  title: string;
  category: string;
  target: number;
  current: number;
  unit: string;
  deadline: string;
  completed: boolean;
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  unlockedAt: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
}

class WellnessCommunityService {
  private readonly CHALLENGES_KEY = 'wellness_challenges';
  private readonly FORUMS_KEY = 'wellness_forums';
  private readonly PROGRESS_KEY = 'user_wellness_progress';

  async generateChallenges(count: number = 5): Promise<WellnessChallenge[]> {
    const categories: Array<'fitness' | 'nutrition' | 'mental' | 'sleep' | 'social'> = [
      'fitness', 'nutrition', 'mental', 'sleep', 'social'
    ];

    const promises = Array.from({ length: count }, (_, i) => {
      const category = categories[i % categories.length];
      const duration = [7, 14, 21, 30][Math.floor(Math.random() * 4)];

      return pollenAIUnified.generate({
        prompt: `Create a ${duration}-day ${category} wellness challenge with a clear goal and daily tasks. Make it achievable and motivating.`,
        mode: 'wellness',
        type: 'challenge'
      }).then(response => {
        const challenge: WellnessChallenge = {
          id: `challenge_${Date.now()}_${i}`,
          title: `${duration}-Day ${category.charAt(0).toUpperCase() + category.slice(1)} Challenge`,
          description: response.content,
          category,
          duration,
          participants: Math.floor(Math.random() * 1000) + 100,
          goal: this.extractGoal(response.content),
          rewards: ['Badge', `${duration * 10} Points`, 'Certificate'],
          startDate: new Date().toISOString(),
          endDate: new Date(Date.now() + duration * 24 * 60 * 60 * 1000).toISOString(),
          difficulty: ['beginner', 'intermediate', 'advanced'][Math.floor(Math.random() * 3)] as any,
          dailyTasks: this.generateDailyTasks(duration)
        };
        return challenge;
      }).catch(error => {
        console.error('Failed to generate challenge:', error);
        return null;
      });
    });

    const results = await Promise.all(promises);
    return results.filter((c): c is WellnessChallenge => c !== null);
  }

  private extractGoal(content: string): string {
    const lines = content.split('\n');
    for (const line of lines) {
      if (line.toLowerCase().includes('goal') || line.toLowerCase().includes('achieve')) {
        return line;
      }
    }
    return 'Complete all daily tasks';
  }

  private generateDailyTasks(duration: number): DailyTask[] {
    const tasks: DailyTask[] = [];
    const activities = [
      'Complete 30 minutes of exercise',
      'Drink 8 glasses of water',
      'Practice 10 minutes of meditation',
      'Get 7-8 hours of sleep',
      'Eat 5 servings of fruits and vegetables',
      'Take a 15-minute walk',
      'Practice gratitude journaling',
      'Stretch for 10 minutes',
      'Limit screen time before bed'
    ];

    for (let day = 1; day <= duration; day++) {
      tasks.push({
        day,
        task: activities[day % activities.length],
        completed: false,
        points: 10
      });
    }

    return tasks;
  }

  joinChallenge(challengeId: string, userId: string): void {
    personalizationEngine.trackBehavior({
      action: 'view',
      contentId: challengeId,
      contentType: 'educational',
      metadata: { type: 'challenge' }
    });
  }

  completeTask(challengeId: string, day: number): void {
    const progress = this.getUserProgress('current_user');
    progress.totalPoints += 10;
    this.saveProgress(progress);

    personalizationEngine.trackBehavior({
      action: 'save',
      contentId: challengeId,
      contentType: 'educational',
      metadata: { day, points: 10 }
    });
  }

  createGoal(
    userId: string,
    title: string,
    category: string,
    target: number,
    unit: string,
    deadline: string
  ): WellnessGoal {
    const goal: WellnessGoal = {
      id: `goal_${Date.now()}`,
      title,
      category,
      target,
      current: 0,
      unit,
      deadline,
      completed: false
    };

    const progress = this.getUserProgress(userId);
    progress.goals.push(goal);
    this.saveProgress(progress);

    return goal;
  }

  updateGoalProgress(userId: string, goalId: string, progress: number): void {
    const userProgress = this.getUserProgress(userId);
    const goal = userProgress.goals.find(g => g.id === goalId);

    if (!goal) return;

    goal.current = progress;
    if (goal.current >= goal.target) {
      goal.completed = true;
      this.unlockAchievement(userId, {
        id: `achievement_${Date.now()}`,
        title: `${goal.title} Completed!`,
        description: `You achieved your ${goal.title} goal`,
        icon: '🏆',
        unlockedAt: new Date().toISOString(),
        rarity: 'rare'
      });
    }

    this.saveProgress(userProgress);
  }

  unlockAchievement(userId: string, achievement: Achievement): void {
    const progress = this.getUserProgress(userId);
    progress.achievements.push(achievement);
    progress.totalPoints += 50;
    progress.level = Math.floor(progress.totalPoints / 500) + 1;
    this.saveProgress(progress);
  }

  getUserProgress(userId: string): UserProgress {
    if (typeof window === 'undefined' || !window.localStorage) {
      return {
        userId,
        goals: [],
        achievements: [],
        totalPoints: 0,
        streakDays: 0,
        level: 1
      };
    }
    try {
      const stored = localStorage.getItem(`${this.PROGRESS_KEY}_${userId}`);
      return stored ? JSON.parse(stored) : {
        userId,
        goals: [],
        achievements: [],
        totalPoints: 0,
        streakDays: 0,
        level: 1
      };
    } catch {
      return {
        userId,
        goals: [],
        achievements: [],
        totalPoints: 0,
        streakDays: 0,
        level: 1
      };
    }
  }

  private saveProgress(progress: UserProgress): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(`${this.PROGRESS_KEY}_${progress.userId}`, JSON.stringify(progress));
    }
  }

  createForum(title: string, category: string): WellnessForum {
    const forum: WellnessForum = {
      id: `forum_${Date.now()}`,
      title,
      category,
      posts: [],
      members: 1,
      createdAt: new Date().toISOString()
    };

    const forums = this.getForums();
    forums.push(forum);
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.FORUMS_KEY, JSON.stringify(forums));
    }

    return forum;
  }

  getForums(): WellnessForum[] {
    if (typeof window === 'undefined' || !window.localStorage) {
      return [];
    }
    try {
      const stored = localStorage.getItem(this.FORUMS_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }
}

export const wellnessCommunityService = new WellnessCommunityService();
