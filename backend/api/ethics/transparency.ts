import { Request, Response } from 'express';
import { db } from '../../lib/server/db';
import { aiDecisionLogs, biasDetectionLogs } from '../../../shared/schema';
import { eq } from 'drizzle-orm';

export async function getTransparencyLogs(req: Request, res: Response) {
  try {
    const { userId, contentId } = req.query;

    let query = db.select().from(aiDecisionLogs);

    if (userId) {
      query = query.where(eq(aiDecisionLogs.userId, userId as string)) as any;
    } else if (contentId) {
      query = query.where(eq(aiDecisionLogs.contentId, contentId as string)) as any;
    }

    const logs = await query.limit(50);

    res.status(200).json({ success: true, logs });
  } catch (error) {
    console.error('Error fetching transparency logs:', error);
    res.status(500).json({ error: 'Failed to fetch transparency logs' });
  }
}

export async function getBiasStats(req: Request, res: Response) {
  try {
    const { timeframe = 'week' } = req.query;

    const logs = await db.select().from(biasDetectionLogs).limit(1000);
    
    const biasTypeCounts: Record<string, number> = {};
    const mitigationRate = logs.length > 0
      ? logs.filter(l => l.mitigationApplied).length / logs.length
      : 0;
    
    logs.forEach(log => {
      biasTypeCounts[log.biasType] = (biasTypeCounts[log.biasType] || 0) + 1;
    });

    const avgScore = logs.length > 0
      ? logs.reduce((sum, l) => sum + l.detectionScore, 0) / logs.length
      : 0;

    res.status(200).json({
      success: true,
      stats: {
        totalDetections: logs.length,
        biasTypeCounts,
        mitigationRate: (mitigationRate * 100).toFixed(1) + '%',
        averageScore: avgScore.toFixed(2)
      }
    });
  } catch (error) {
    console.error('Error fetching bias stats:', error);
    res.status(500).json({ error: 'Failed to fetch bias stats' });
  }
}
