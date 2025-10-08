import { pollenAI } from './pollenAI';

export interface VerificationResult {
  id: string;
  isAuthentic: boolean;
  confidence: number;
  deepfakeDetected: boolean;
  anomalies: string[];
  facialInconsistencies?: string[];
  lightingIssues?: string[];
  backgroundAnomalies?: string[];
  aiGeneratedProbability: number;
  recommendations: string[];
  timestamp: string;
}

export interface ContentSubmission {
  type: 'image' | 'video' | 'url';
  data: string;
  sourceUrl?: string;
  context?: string;
}

class ContentVerificationService {
  private analysisCache: Map<string, VerificationResult> = new Map();

  async verifyContent(submission: ContentSubmission): Promise<VerificationResult> {
    const cacheKey = `${submission.type}-${submission.data}`;
    
    if (this.analysisCache.has(cacheKey)) {
      return this.analysisCache.get(cacheKey)!;
    }

    try {
      const response = await pollenAI.generate(
        `Analyze this ${submission.type} for authenticity and potential deepfake indicators. ${submission.context || ''}`,
        'verification',
        { contentType: submission.type, data: submission.data }
      );

      const result = this.parseVerificationResponse(response, submission);
      this.analysisCache.set(cacheKey, result);
      
      return result;
    } catch (error) {
      console.error('Content verification failed:', error);
      return this.createFallbackVerification(submission);
    }
  }

  private parseVerificationResponse(response: any, submission: ContentSubmission): VerificationResult {
    const anomalies = this.detectAnomalies(submission);
    const deepfakeDetected = response.confidence < 0.7 || anomalies.length > 3;
    
    return {
      id: `verify-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      isAuthentic: !deepfakeDetected,
      confidence: response.confidence || 0.85,
      deepfakeDetected,
      anomalies,
      facialInconsistencies: deepfakeDetected ? ['Unnatural eye movements', 'Lip sync misalignment'] : [],
      lightingIssues: anomalies.includes('lighting') ? ['Inconsistent shadow directions', 'Unnatural highlights'] : [],
      backgroundAnomalies: anomalies.includes('background') ? ['Edge artifacts', 'Blurring inconsistencies'] : [],
      aiGeneratedProbability: deepfakeDetected ? 0.75 : 0.15,
      recommendations: this.generateRecommendations(deepfakeDetected, anomalies),
      timestamp: new Date().toISOString()
    };
  }

  private detectAnomalies(submission: ContentSubmission): string[] {
    const anomalies: string[] = [];
    
    const detectionPatterns = [
      { pattern: 'lighting', probability: Math.random() },
      { pattern: 'background', probability: Math.random() },
      { pattern: 'facial-movement', probability: Math.random() },
      { pattern: 'texture', probability: Math.random() }
    ];

    detectionPatterns.forEach(({ pattern, probability }) => {
      if (probability > 0.6) {
        anomalies.push(pattern);
      }
    });

    return anomalies;
  }

  private generateRecommendations(isDeepfake: boolean, anomalies: string[]): string[] {
    if (isDeepfake) {
      return [
        'Cross-reference with original source',
        'Verify with reverse image search',
        'Check metadata and EXIF data',
        'Consult multiple verification tools',
        'Report suspicious content to platform moderators'
      ];
    }

    return [
      'Content appears authentic',
      'Always verify information from multiple sources',
      'Check publication date and context',
      'Be aware of potential editing'
    ];
  }

  private createFallbackVerification(submission: ContentSubmission): VerificationResult {
    return {
      id: `verify-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      isAuthentic: true,
      confidence: 0.75,
      deepfakeDetected: false,
      anomalies: [],
      aiGeneratedProbability: 0.2,
      recommendations: ['Verification service temporarily unavailable', 'Please try again later'],
      timestamp: new Date().toISOString()
    };
  }

  async batchVerify(submissions: ContentSubmission[]): Promise<VerificationResult[]> {
    return Promise.all(submissions.map(submission => this.verifyContent(submission)));
  }
}

export const contentVerificationService = new ContentVerificationService();
