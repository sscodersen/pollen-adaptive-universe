import axios from 'axios';

const API_BASE = '/api/health-research';

export interface HealthDataSubmission {
  userId: string;
  dataType: 'fitness' | 'nutrition' | 'mental_health' | 'sleep' | 'medical';
  category: string;
  metrics: Record<string, any>;
  demographics?: Record<string, any>;
  tags?: string[];
  isPublic?: boolean;
}

export interface WellnessJourneySubmission {
  userId: string;
  journeyType: 'weight_loss' | 'fitness' | 'mental_wellness' | 'recovery';
  startDate: string;
  endDate?: string;
  milestones?: any[];
  outcomes?: Record<string, any>;
  challenges?: string[];
  insights?: string;
  isActive?: boolean;
  isPublic?: boolean;
}

export interface HealthInsightCreation {
  insightType: 'trend' | 'correlation' | 'recommendation' | 'breakthrough';
  category: string;
  title: string;
  description: string;
  dataPoints?: number;
  confidence?: number;
  significance?: number;
  visualizationData?: any;
  metadata?: Record<string, any>;
}

export interface ResearchFindingCreation {
  title: string;
  summary: string;
  fullReport?: string;
  findingType: 'correlation' | 'breakthrough' | 'pattern' | 'anomaly';
  impactScore?: number;
  datasetSize?: number;
  categories?: string[];
  keyMetrics?: Record<string, any>;
  visualizations?: any[];
  citations?: any[];
  status?: 'draft' | 'published' | 'peer_review';
}

class HealthResearchService {
  async submitHealthData(data: HealthDataSubmission) {
    const response = await axios.post(`${API_BASE}/data`, data);
    return response.data;
  }

  async getHealthData(filters?: {
    dataType?: string;
    category?: string;
    isPublic?: boolean;
  }) {
    const response = await axios.get(`${API_BASE}/data`, { params: filters });
    return response.data;
  }

  async submitWellnessJourney(journey: WellnessJourneySubmission) {
    const response = await axios.post(`${API_BASE}/journeys`, journey);
    return response.data;
  }

  async getWellnessJourneys(filters?: {
    journeyType?: string;
    isActive?: boolean;
    isPublic?: boolean;
  }) {
    const response = await axios.get(`${API_BASE}/journeys`, { params: filters });
    return response.data;
  }

  async createHealthInsight(insight: HealthInsightCreation) {
    const response = await axios.post(`${API_BASE}/insights`, insight);
    return response.data;
  }

  async getHealthInsights(filters?: {
    insightType?: string;
    category?: string;
  }) {
    const response = await axios.get(`${API_BASE}/insights`, { params: filters });
    return response.data;
  }

  async createResearchFinding(finding: ResearchFindingCreation) {
    const response = await axios.post(`${API_BASE}/findings`, finding);
    return response.data;
  }

  async getResearchFindings(filters?: {
    findingType?: string;
    status?: string;
  }) {
    const response = await axios.get(`${API_BASE}/findings`, { params: filters });
    return response.data;
  }
}

export const healthResearchService = new HealthResearchService();
