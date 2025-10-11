import { analyticsEngine } from './analyticsEngine';

interface Experiment {
  id: string;
  name: string;
  description: string;
  variants: Variant[];
  startDate: Date;
  endDate?: Date;
  isActive: boolean;
  targetMetric: string;
}

interface Variant {
  id: string;
  name: string;
  description: string;
  weight: number; // 0-100, determines probability
  config: Record<string, any>;
}

interface ExperimentResult {
  experimentId: string;
  variantId: string;
  userId: string;
  metrics: Record<string, number>;
  timestamp: Date;
}

class ABTestingFramework {
  private experiments: Map<string, Experiment> = new Map();
  private userAssignments: Map<string, Map<string, string>> = new Map();
  private results: ExperimentResult[] = [];

  // Create a new experiment
  createExperiment(experiment: Omit<Experiment, 'isActive'>): Experiment {
    const newExperiment: Experiment = {
      ...experiment,
      isActive: true
    };

    // Validate variant weights sum to 100
    const totalWeight = experiment.variants.reduce((sum, v) => sum + v.weight, 0);
    if (Math.abs(totalWeight - 100) > 0.01) {
      throw new Error('Variant weights must sum to 100');
    }

    this.experiments.set(experiment.id, newExperiment);
    console.log(`📊 A/B Test Created: ${experiment.name}`);
    
    return newExperiment;
  }

  // Assign a user to a variant
  getVariant(experimentId: string, userId: string): Variant | null {
    const experiment = this.experiments.get(experimentId);
    if (!experiment || !experiment.isActive) {
      return null;
    }

    // Check if user already assigned
    if (!this.userAssignments.has(userId)) {
      this.userAssignments.set(userId, new Map());
    }

    const userExperiments = this.userAssignments.get(userId)!;
    const existingAssignment = userExperiments.get(experimentId);

    if (existingAssignment) {
      return experiment.variants.find(v => v.id === existingAssignment) || null;
    }

    // Assign user to variant based on weights
    const variant = this.assignVariant(experiment.variants);
    userExperiments.set(experimentId, variant.id);

    // Track assignment
    analyticsEngine.trackEvent(userId, 'ab_test_assigned', {
      experimentId,
      experimentName: experiment.name,
      variantId: variant.id,
      variantName: variant.name
    });

    console.log(`👤 User ${userId} assigned to variant ${variant.name} in ${experiment.name}`);
    
    return variant;
  }

  private assignVariant(variants: Variant[]): Variant {
    const random = Math.random() * 100;
    let cumulative = 0;

    for (const variant of variants) {
      cumulative += variant.weight;
      if (random <= cumulative) {
        return variant;
      }
    }

    // Fallback to first variant
    return variants[0];
  }

  // Track experiment results
  trackResult(
    experimentId: string,
    userId: string,
    metrics: Record<string, number>
  ) {
    const userVariant = this.userAssignments.get(userId)?.get(experimentId);
    if (!userVariant) {
      console.warn(`No variant assignment found for user ${userId} in experiment ${experimentId}`);
      return;
    }

    const result: ExperimentResult = {
      experimentId,
      variantId: userVariant,
      userId,
      metrics,
      timestamp: new Date()
    };

    this.results.push(result);

    // Track in analytics
    analyticsEngine.trackEvent(userId, 'ab_test_result', {
      experimentId,
      variantId: userVariant,
      ...metrics
    });
  }

  // Get experiment results
  getExperimentResults(experimentId: string) {
    const experiment = this.experiments.get(experimentId);
    if (!experiment) {
      return null;
    }

    const experimentResults = this.results.filter(r => r.experimentId === experimentId);
    
    // Calculate statistics for each variant
    const variantStats = experiment.variants.map(variant => {
      const variantResults = experimentResults.filter(r => r.variantId === variant.id);
      
      const stats: Record<string, any> = {
        variantId: variant.id,
        variantName: variant.name,
        sampleSize: variantResults.length,
        metrics: {}
      };

      // Calculate average for each metric
      if (variantResults.length > 0) {
        const metricKeys = Object.keys(variantResults[0].metrics);
        
        metricKeys.forEach(key => {
          const values = variantResults.map(r => r.metrics[key]);
          stats.metrics[key] = {
            mean: values.reduce((sum, v) => sum + v, 0) / values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            stdDev: this.calculateStdDev(values)
          };
        });
      }

      return stats;
    });

    // Determine winner based on target metric
    const winner = this.determineWinner(variantStats, experiment.targetMetric);

    return {
      experiment,
      variantStats,
      winner,
      totalSampleSize: experimentResults.length,
      startDate: experiment.startDate,
      endDate: experiment.endDate
    };
  }

  private calculateStdDev(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((sum, v) => sum + v, 0) / values.length;
    
    return Math.sqrt(variance);
  }

  private determineWinner(variantStats: any[], targetMetric: string) {
    const validStats = variantStats.filter(
      s => s.sampleSize > 0 && s.metrics[targetMetric]
    );

    if (validStats.length === 0) {
      return null;
    }

    // Find variant with highest mean for target metric
    const winner = validStats.reduce((best, current) => {
      const currentMean = current.metrics[targetMetric]?.mean || 0;
      const bestMean = best.metrics[targetMetric]?.mean || 0;
      return currentMean > bestMean ? current : best;
    });

    // Calculate confidence (simplified)
    const winnerMean = winner.metrics[targetMetric].mean;
    const runners = validStats.filter(s => s.variantId !== winner.variantId);
    
    if (runners.length === 0) {
      return { ...winner, confidence: 1.0 };
    }

    const maxRunnerMean = Math.max(...runners.map(r => r.metrics[targetMetric]?.mean || 0));
    const improvement = winnerMean > 0 ? ((winnerMean - maxRunnerMean) / maxRunnerMean) * 100 : 0;
    const confidence = Math.min(winner.sampleSize / 100, 1); // Simplified confidence

    return {
      ...winner,
      confidence,
      improvement: Math.max(0, improvement)
    };
  }

  // Stop an experiment
  stopExperiment(experimentId: string) {
    const experiment = this.experiments.get(experimentId);
    if (experiment) {
      experiment.isActive = false;
      experiment.endDate = new Date();
      console.log(`🛑 A/B Test Stopped: ${experiment.name}`);
    }
  }

  // Get all active experiments
  getActiveExperiments(): Experiment[] {
    return Array.from(this.experiments.values()).filter(e => e.isActive);
  }

  // Export results
  exportResults(experimentId: string): string {
    const results = this.getExperimentResults(experimentId);
    return JSON.stringify(results, null, 2);
  }

  // Pre-defined experiments
  initializeDefaultExperiments() {
    // Feed Algorithm Experiment
    this.createExperiment({
      id: 'feed_algorithm_v1',
      name: 'Feed Algorithm Comparison',
      description: 'Compare personalized vs chronological feed',
      variants: [
        {
          id: 'control',
          name: 'Chronological Feed',
          description: 'Traditional time-based feed',
          weight: 50,
          config: { algorithm: 'chronological' }
        },
        {
          id: 'treatment',
          name: 'AI Personalized Feed',
          description: 'ML-powered personalization',
          weight: 50,
          config: { algorithm: 'personalized' }
        }
      ],
      startDate: new Date(),
      targetMetric: 'engagement_time'
    });

    // Content Recommendation Experiment
    this.createExperiment({
      id: 'content_rec_v1',
      name: 'Content Recommendation Style',
      description: 'Test different recommendation approaches',
      variants: [
        {
          id: 'similarity',
          name: 'Similar Content',
          description: 'Recommend similar to viewed content',
          weight: 33.33,
          config: { strategy: 'content_based' }
        },
        {
          id: 'collaborative',
          name: 'Collaborative Filtering',
          description: 'Recommend based on similar users',
          weight: 33.33,
          config: { strategy: 'collaborative' }
        },
        {
          id: 'hybrid',
          name: 'Hybrid Approach',
          description: 'Combine multiple strategies',
          weight: 33.34,
          config: { strategy: 'hybrid' }
        }
      ],
      startDate: new Date(),
      targetMetric: 'click_through_rate'
    });

    console.log('✅ Default A/B tests initialized');
  }
}

export const abTestingFramework = new ABTestingFramework();
export type { Experiment, Variant, ExperimentResult };
