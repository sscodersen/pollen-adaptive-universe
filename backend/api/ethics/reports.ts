import { Request, Response } from 'express';
import { db } from '../../lib/server/db';
import { ethicalReports, InsertEthicalReport } from '../../../shared/schema';
import { eq } from 'drizzle-orm';

export async function createEthicalReport(req: Request, res: Response) {
  try {
    const { userId, contentId, concernType, description, severity } = req.body;

    if (!userId || !concernType || !description || !severity) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const reportId = `report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const newReport: InsertEthicalReport = {
      reportId,
      userId,
      contentId: contentId || null,
      concernType,
      description,
      severity,
      status: 'pending',
      metadata: { reportedAt: new Date().toISOString() }
    };

    const [report] = await db.insert(ethicalReports).values(newReport).returning();

    res.status(201).json({
      success: true,
      report,
      message: 'Ethical concern reported successfully. Our team will review it shortly.'
    });
  } catch (error) {
    console.error('Error creating ethical report:', error);
    res.status(500).json({ error: 'Failed to create ethical report' });
  }
}

export async function getEthicalReports(req: Request, res: Response) {
  try {
    const { userId, status } = req.query;

    let query = db.select().from(ethicalReports);

    if (userId) {
      query = query.where(eq(ethicalReports.userId, userId as string)) as any;
    }

    if (status) {
      query = query.where(eq(ethicalReports.status, status as string)) as any;
    }

    const reports = await query.limit(100);

    res.status(200).json({ success: true, reports });
  } catch (error) {
    console.error('Error fetching ethical reports:', error);
    res.status(500).json({ error: 'Failed to fetch ethical reports' });
  }
}

export async function updateReportStatus(req: Request, res: Response) {
  try {
    const { reportId } = req.params;
    const { status, resolution } = req.body;

    if (!status) {
      return res.status(400).json({ error: 'Status is required' });
    }

    const [updatedReport] = await db
      .update(ethicalReports)
      .set({ status, resolution, updatedAt: new Date() })
      .where(eq(ethicalReports.reportId, reportId))
      .returning();

    if (!updatedReport) {
      return res.status(404).json({ error: 'Report not found' });
    }

    res.status(200).json({ success: true, report: updatedReport });
  } catch (error) {
    console.error('Error updating report status:', error);
    res.status(500).json({ error: 'Failed to update report status' });
  }
}
