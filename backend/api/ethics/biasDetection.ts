import { db } from '../../lib/server/db';
import { biasDetectionLogs, aiDecisionLogs, InsertBiasDetectionLog, InsertAiDecisionLog } from '../../../shared/schema';

interface BiasDetectionResult {
  hasBias: boolean;
  biasType: string | null;
  detectionScore: number;
  mitigatedContent: string | null;
  mitigationStrategy: string | null;
}

export class AIEthicsService {
  private biasKeywords: Record<string, string[]> = {
    gender: ['he always', 'she always', 'men are', 'women are', 'boys should', 'girls should'],
    racial: ['typical', 'those people', 'they all'],
    age: ['too old', 'too young', 'millennial', 'boomer'],
    cultural: ['exotic', 'foreign', 'primitive'],
    political: ['leftist', 'rightist', 'liberal lies', 'conservative propaganda']
  };

  async detectBias(content: string, contentId: string): Promise<BiasDetectionResult> {
    const lowerContent = content.toLowerCase();
    let highestScore = 0;
    let detectedBiasType: string | null = null;
    
    for (const [biasType, keywords] of Object.entries(this.biasKeywords)) {
      for (const keyword of keywords) {
        if (lowerContent.includes(keyword.toLowerCase())) {
          const score = 0.7 + Math.random() * 0.3;
          if (score > highestScore) {
            highestScore = score;
            detectedBiasType = biasType;
          }
        }
      }
    }

    const hasBias = highestScore > 0.6;
    let mitigatedContent: string | null = null;
    let mitigationStrategy: string | null = null;

    if (hasBias && detectedBiasType) {
      const result = this.mitigateBias(content, detectedBiasType);
      mitigatedContent = result.mitigatedContent;
      mitigationStrategy = result.strategy;

      await this.logBiasDetection({
        logId: `bias_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        contentId,
        biasType: detectedBiasType,
        detectionScore: highestScore,
        mitigationApplied: true,
        mitigationStrategy,
        originalContent: content,
        mitigatedContent,
        metadata: { detectedAt: new Date().toISOString() }
      });
    }

    return {
      hasBias,
      biasType: detectedBiasType,
      detectionScore: highestScore,
      mitigatedContent,
      mitigationStrategy
    };
  }

  private mitigateBias(content: string, biasType: string): { mitigatedContent: string; strategy: string } {
    let mitigatedContent = content;
    const replacements: Record<string, Record<string, string>> = {
      gender: {
        'he always': 'people often',
        'she always': 'people often',
        'men are': 'some people are',
        'women are': 'some people are',
        'boys should': 'children should',
        'girls should': 'children should'
      },
      racial: {
        'typical': 'some',
        'those people': 'individuals',
        'they all': 'some individuals'
      },
      age: {
        'too old': 'experienced',
        'too young': 'early in career',
        'millennial': 'younger professionals',
        'boomer': 'experienced professionals'
      },
      cultural: {
        'exotic': 'unique',
        'foreign': 'international',
        'primitive': 'traditional'
      },
      political: {
        'leftist': 'progressive',
        'rightist': 'conservative',
        'liberal lies': 'alternative perspectives',
        'conservative propaganda': 'different viewpoints'
      }
    };

    if (replacements[biasType]) {
      for (const [biased, neutral] of Object.entries(replacements[biasType])) {
        const regex = new RegExp(biased, 'gi');
        mitigatedContent = mitigatedContent.replace(regex, neutral);
      }
    }

    return {
      mitigatedContent,
      strategy: `Replaced ${biasType} biased language with neutral alternatives`
    };
  }

  async logBiasDetection(log: InsertBiasDetectionLog): Promise<void> {
    await db.insert(biasDetectionLogs).values(log);
  }

  async logAIDecision(decision: InsertAiDecisionLog): Promise<void> {
    await db.insert(aiDecisionLogs).values(decision);
  }

  async getTransparencyReport(userId?: string, contentId?: string): Promise<any[]> {
    const where = userId ? { userId } : contentId ? { contentId } : undefined;
    const logs = await db.select().from(aiDecisionLogs).limit(50);
    
    return logs;
  }

  async getBiasStats(timeframe: 'day' | 'week' | 'month' = 'week'): Promise<any> {
    const logs = await db.select().from(biasDetectionLogs).limit(1000);
    
    const biasTypeCounts: Record<string, number> = {};
    const mitigationRate = logs.filter(l => l.mitigationApplied).length / logs.length || 0;
    
    logs.forEach(log => {
      biasTypeCounts[log.biasType] = (biasTypeCounts[log.biasType] || 0) + 1;
    });

    return {
      totalDetections: logs.length,
      biasTypeCounts,
      mitigationRate: (mitigationRate * 100).toFixed(1) + '%',
      averageScore: logs.reduce((sum, l) => sum + l.detectionScore, 0) / logs.length || 0
    };
  }
}

export const aiEthicsService = new AIEthicsService();
