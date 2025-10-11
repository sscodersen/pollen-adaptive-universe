export class FeedbackAnalyzer {
  analyzeSentiment(text: string): 'positive' | 'neutral' | 'negative' {
    const lowerText = text.toLowerCase();
    
    const positiveWords = ['great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful', 'fantastic', 'awesome', 'good', 'helpful', 'useful', 'appreciate', 'thank'];
    const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'useless', 'broken', 'bug', 'issue', 'problem', 'frustrating', 'disappointing', 'poor'];
    
    let positiveCount = 0;
    let negativeCount = 0;
    
    positiveWords.forEach(word => {
      if (lowerText.includes(word)) positiveCount++;
    });
    
    negativeWords.forEach(word => {
      if (lowerText.includes(word)) negativeCount++;
    });
    
    if (positiveCount > negativeCount) return 'positive';
    if (negativeCount > positiveCount) return 'negative';
    return 'neutral';
  }

  extractTopics(text: string): string[] {
    const lowerText = text.toLowerCase();
    const topics: string[] = [];
    
    const topicKeywords: Record<string, string[]> = {
      'ui_ux': ['interface', 'design', 'layout', 'button', 'menu', 'navigation', 'visual', 'theme', 'color'],
      'performance': ['slow', 'fast', 'loading', 'lag', 'speed', 'performance', 'optimize'],
      'ai_features': ['ai', 'recommendation', 'personalization', 'content', 'generation', 'chatbot'],
      'community': ['community', 'forum', 'chat', 'discussion', 'social', 'share'],
      'gamification': ['points', 'badges', 'leaderboard', 'achievement', 'reward', 'level'],
      'events': ['event', 'webinar', 'workshop', 'meetup', 'conference'],
      'feedback_system': ['feedback', 'suggestion', 'report', 'request'],
      'mobile': ['mobile', 'responsive', 'phone', 'tablet', 'touch'],
      'accessibility': ['accessibility', 'a11y', 'screen reader', 'contrast', 'keyboard'],
      'security': ['security', 'privacy', 'data', 'encrypt', 'safe']
    };
    
    Object.entries(topicKeywords).forEach(([topic, keywords]) => {
      if (keywords.some(keyword => lowerText.includes(keyword))) {
        topics.push(topic);
      }
    });
    
    return topics.length > 0 ? topics : ['general'];
  }

  categorizeFeedback(text: string, type: string): string {
    const lowerText = text.toLowerCase();
    
    const categories: Record<string, string[]> = {
      'platform': ['platform', 'system', 'infrastructure', 'backend', 'database'],
      'ai_features': ['ai', 'machine learning', 'recommendation', 'personalization', 'nlp'],
      'community': ['community', 'forum', 'chat', 'social', 'discussion'],
      'ui_ux': ['design', 'interface', 'user experience', 'layout', 'navigation'],
      'performance': ['performance', 'speed', 'loading', 'optimization', 'lag']
    };
    
    for (const [category, keywords] of Object.entries(categories)) {
      if (keywords.some(keyword => lowerText.includes(keyword))) {
        return category;
      }
    }
    
    return type === 'bug' ? 'platform' : 'general';
  }

  determinePriority(text: string, type: string, sentiment: string): string {
    const lowerText = text.toLowerCase();
    
    const criticalWords = ['critical', 'urgent', 'emergency', 'broken', 'crash', 'data loss', 'security'];
    const highWords = ['important', 'major', 'significant', 'blocking'];
    
    if (type === 'bug' && criticalWords.some(word => lowerText.includes(word))) {
      return 'critical';
    }
    
    if (type === 'bug' && highWords.some(word => lowerText.includes(word))) {
      return 'high';
    }
    
    if (sentiment === 'negative' && type === 'complaint') {
      return 'high';
    }
    
    if (type === 'feature_request' && highWords.some(word => lowerText.includes(word))) {
      return 'medium';
    }
    
    return 'medium';
  }

  async analyzeFeedback(feedback: {
    subject: string;
    description: string;
    feedbackType: string;
  }) {
    const fullText = `${feedback.subject} ${feedback.description}`;
    
    const sentiment = this.analyzeSentiment(fullText);
    const topics = this.extractTopics(fullText);
    const category = this.categorizeFeedback(fullText, feedback.feedbackType);
    const priority = this.determinePriority(fullText, feedback.feedbackType, sentiment);
    
    return {
      sentiment,
      topics,
      category,
      priority
    };
  }

  generateAISummary(feedbackList: any[]): string {
    const totalCount = feedbackList.length;
    const sentimentCounts = feedbackList.reduce((acc, f) => {
      acc[f.sentiment || 'neutral'] = (acc[f.sentiment || 'neutral'] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const topicCounts = feedbackList.reduce((acc, f) => {
      (f.topics || []).forEach((topic: string) => {
        acc[topic] = (acc[topic] || 0) + 1;
      });
      return acc;
    }, {} as Record<string, number>);
    
    const topTopics = Object.entries(topicCounts)
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .slice(0, 3)
      .map(([topic]) => topic);
    
    return `Analyzed ${totalCount} feedback submissions. ` +
           `Sentiment: ${sentimentCounts.positive || 0} positive, ${sentimentCounts.neutral || 0} neutral, ${sentimentCounts.negative || 0} negative. ` +
           `Top topics: ${topTopics.join(', ')}.`;
  }
}

export const feedbackAnalyzer = new FeedbackAnalyzer();
