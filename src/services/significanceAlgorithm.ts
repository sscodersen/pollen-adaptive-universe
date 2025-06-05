
// Significance Algorithm - 7-Factor Content Scoring System
export interface SignificanceFactors {
  scope: number; // 0-10: Number of individuals affected
  intensity: number; // 0-10: Magnitude of impact
  originality: number; // 0-10: Unexpected/distinctive nature
  immediacy: number; // 0-10: Temporal proximity
  practicability: number; // 0-10: Actionable potential for readers
  positivity: number; // 0-10: Positive aspects evaluation
  credibility: number; // 0-10: Source reliability
}

export interface ScoredContent {
  id: string;
  content: any;
  significanceScore: number;
  factors: SignificanceFactors;
  category: 'news' | 'social' | 'shop' | 'entertainment' | 'automation';
  timestamp: number;
}

class SignificanceAlgorithm {
  private weights = {
    scope: 0.18,
    intensity: 0.16,
    originality: 0.14,
    immediacy: 0.12,
    practicability: 0.16,
    positivity: 0.12,
    credibility: 0.12
  };

  calculateSignificance(factors: SignificanceFactors): number {
    const weightedScore = 
      factors.scope * this.weights.scope +
      factors.intensity * this.weights.intensity +
      factors.originality * this.weights.originality +
      factors.immediacy * this.weights.immediacy +
      factors.practicability * this.weights.practicability +
      factors.positivity * this.weights.positivity +
      factors.credibility * this.weights.credibility;

    return Math.round(weightedScore * 100) / 100;
  }

  analyzeContent(content: string, category: string, source?: string): SignificanceFactors {
    // AI-powered analysis simulation
    const keywords = content.toLowerCase();
    
    // Scope analysis
    const scopeKeywords = ['global', 'worldwide', 'millions', 'billions', 'universal', 'international'];
    const scope = Math.min(10, scopeKeywords.filter(k => keywords.includes(k)).length * 2 + Math.random() * 4 + 3);

    // Intensity analysis
    const intensityKeywords = ['breakthrough', 'revolutionary', 'crisis', 'major', 'significant', 'critical'];
    const intensity = Math.min(10, intensityKeywords.filter(k => keywords.includes(k)).length * 1.5 + Math.random() * 3 + 4);

    // Originality analysis
    const originalityKeywords = ['first', 'unprecedented', 'novel', 'innovative', 'unique', 'never'];
    const originality = Math.min(10, originalityKeywords.filter(k => keywords.includes(k)).length * 2 + Math.random() * 4 + 2);

    // Immediacy (simulated based on recency)
    const immediacy = Math.random() * 4 + 6; // Most content is recent

    // Practicability analysis
    const practicalKeywords = ['how to', 'guide', 'tips', 'actionable', 'implement', 'apply'];
    const practicability = Math.min(10, practicalKeywords.filter(k => keywords.includes(k)).length * 2 + Math.random() * 3 + 3);

    // Positivity analysis
    const positiveKeywords = ['solution', 'breakthrough', 'success', 'improvement', 'innovation', 'progress'];
    const negativeKeywords = ['crisis', 'failure', 'problem', 'decline', 'threat', 'risk'];
    const positiveCount = positiveKeywords.filter(k => keywords.includes(k)).length;
    const negativeCount = negativeKeywords.filter(k => keywords.includes(k)).length;
    const positivity = Math.max(0, Math.min(10, 5 + (positiveCount - negativeCount) * 2 + Math.random() * 2));

    // Credibility (simulated based on source and category)
    let credibility = 7; // Base credibility
    if (source === 'Pollen Analysis') credibility = 8.5;
    if (category === 'news') credibility += 0.5;
    credibility = Math.min(10, credibility + Math.random() * 1);

    return {
      scope: Math.round(scope * 10) / 10,
      intensity: Math.round(intensity * 10) / 10,
      originality: Math.round(originality * 10) / 10,
      immediacy: Math.round(immediacy * 10) / 10,
      practicability: Math.round(practicability * 10) / 10,
      positivity: Math.round(positivity * 10) / 10,
      credibility: Math.round(credibility * 10) / 10
    };
  }

  scoreContent(content: any, category: 'news' | 'social' | 'shop' | 'entertainment' | 'automation', source?: string): ScoredContent {
    const contentText = typeof content === 'string' ? content : content.description || content.summary || content.title || '';
    const factors = this.analyzeContent(contentText, category, source);
    const significanceScore = this.calculateSignificance(factors);

    return {
      id: Date.now().toString() + Math.random(),
      content,
      significanceScore,
      factors,
      category,
      timestamp: Date.now()
    };
  }

  filterHighSignificance(scoredContent: ScoredContent[]): ScoredContent[] {
    return scoredContent.filter(item => item.significanceScore > 7.0);
  }

  getTrendingTopics(): string[] {
    return [
      'AI breakthrough developments',
      'sustainable technology innovations',
      'global economic shifts',
      'climate change solutions',
      'space exploration advances',
      'biotechnology discoveries',
      'renewable energy progress',
      'quantum computing milestones',
      'digital privacy developments',
      'automation workplace impact'
    ];
  }
}

export const significanceAlgorithm = new SignificanceAlgorithm();
