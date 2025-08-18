// Healthcare AI Service
import { pollenAdaptiveService } from './pollenAdaptiveService';

export interface PatientReport {
  id: string;
  patientId: string;
  summary: string;
  keyFindings: string[];
  recommendations: string[];
  riskFactors: string[];
  followUpRequired: boolean;
  urgencyLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface TreatmentPlan {
  id: string;
  condition: string;
  medications: Array<{name: string; dosage: string; frequency: string}>;
  lifestyle: string[];
  appointments: Array<{type: string; frequency: string}>;
  goals: string[];
  duration: string;
}

export interface MedicalResearch {
  title: string;
  summary: string;
  keyFindings: string[];
  relevance: string;
  source: string;
  date: string;
  impact: 'high' | 'medium' | 'low';
}

export class HealthcareService {
  async generatePatientReport(symptoms: string[], medicalHistory: string): Promise<PatientReport> {
    const analysis = await pollenAdaptiveService.analyzeTrends(`medical symptoms: ${symptoms.join(', ')}`);
    
    return {
      id: `report_${Date.now()}`,
      patientId: `patient_${Math.random().toString(36).substr(2, 9)}`,
      summary: this.generateSummary(symptoms, medicalHistory),
      keyFindings: this.analyzeSymptoms(symptoms),
      recommendations: this.generateRecommendations(symptoms),
      riskFactors: this.identifyRiskFactors(symptoms, medicalHistory),
      followUpRequired: this.requiresFollowUp(symptoms),
      urgencyLevel: this.assessUrgency(symptoms)
    };
  }

  async createTreatmentPlan(condition: string, patientProfile: {
    age: number;
    gender: string;
    allergies: string[];
    conditions: string[];
  }): Promise<TreatmentPlan> {
    const solution = await pollenAdaptiveService.solveTask(`Create treatment plan for ${condition}`);
    
    return {
      id: `treatment_${Date.now()}`,
      condition,
      medications: this.recommendMedications(condition, patientProfile),
      lifestyle: this.generateLifestyleRecommendations(condition),
      appointments: this.scheduleAppointments(condition),
      goals: this.setTreatmentGoals(condition),
      duration: this.estimateTreatmentDuration(condition)
    };
  }

  async curateMedicalResearch(topic: string): Promise<MedicalResearch[]> {
    const research: MedicalResearch[] = [];
    
    // Generate mock research papers
    const topics = [
      `Recent advances in ${topic} treatment`,
      `Clinical trials for ${topic}`,
      `${topic}: A systematic review`,
      `Novel therapeutic approaches to ${topic}`
    ];

    for (const title of topics) {
      research.push({
        title,
        summary: `This study examines the latest developments in ${topic} research and clinical applications.`,
        keyFindings: this.generateResearchFindings(topic),
        relevance: `Highly relevant for ${topic} treatment protocols`,
        source: this.generateSource(),
        date: this.generateRecentDate(),
        impact: this.assessResearchImpact()
      });
    }

    return research;
  }

  async generateHealthInsights(healthData: {
    vitals: {heartRate: number; bloodPressure: string; temperature: number};
    symptoms: string[];
    lifestyle: {exercise: string; diet: string; sleep: string};
  }): Promise<{
    insights: string[];
    recommendations: string[];
    alerts: string[];
    score: number;
  }> {
    return {
      insights: this.analyzeHealthData(healthData),
      recommendations: this.generateHealthRecommendations(healthData),
      alerts: this.generateHealthAlerts(healthData),
      score: this.calculateHealthScore(healthData)
    };
  }

  private generateSummary(symptoms: string[], medicalHistory: string): string {
    const primarySymptom = symptoms[0] || 'general discomfort';
    return `Patient presents with ${primarySymptom} and additional symptoms. Medical history includes ${medicalHistory}. Comprehensive evaluation recommended.`;
  }

  private analyzeSymptoms(symptoms: string[]): string[] {
    return symptoms.map(symptom => `Analysis of ${symptom}: Contributing factor identified`);
  }

  private generateRecommendations(symptoms: string[]): string[] {
    const baseRecommendations = [
      'Maintain adequate hydration',
      'Monitor symptoms closely',
      'Follow up if symptoms persist or worsen'
    ];

    if (symptoms.some(s => s.toLowerCase().includes('pain'))) {
      baseRecommendations.push('Consider pain management strategies');
    }

    if (symptoms.some(s => s.toLowerCase().includes('fever'))) {
      baseRecommendations.push('Monitor temperature regularly');
    }

    return baseRecommendations;
  }

  private identifyRiskFactors(symptoms: string[], medicalHistory: string): string[] {
    const riskFactors = [];
    
    if (medicalHistory.toLowerCase().includes('diabetes')) {
      riskFactors.push('Pre-existing diabetes condition');
    }
    
    if (medicalHistory.toLowerCase().includes('hypertension')) {
      riskFactors.push('History of high blood pressure');
    }

    if (symptoms.length > 3) {
      riskFactors.push('Multiple concurrent symptoms');
    }

    return riskFactors;
  }

  private requiresFollowUp(symptoms: string[]): boolean {
    const seriousSymptoms = ['chest pain', 'difficulty breathing', 'severe headache', 'high fever'];
    return symptoms.some(symptom => 
      seriousSymptoms.some(serious => symptom.toLowerCase().includes(serious))
    );
  }

  private assessUrgency(symptoms: string[]): 'low' | 'medium' | 'high' | 'critical' {
    const criticalSymptoms = ['chest pain', 'difficulty breathing', 'loss of consciousness'];
    const highSymptoms = ['severe pain', 'high fever', 'bleeding'];
    const mediumSymptoms = ['persistent pain', 'fever', 'nausea'];

    if (symptoms.some(s => criticalSymptoms.some(c => s.toLowerCase().includes(c)))) {
      return 'critical';
    } else if (symptoms.some(s => highSymptoms.some(h => s.toLowerCase().includes(h)))) {
      return 'high';
    } else if (symptoms.some(s => mediumSymptoms.some(m => s.toLowerCase().includes(m)))) {
      return 'medium';
    }
    return 'low';
  }

  private recommendMedications(condition: string, profile: any): Array<{name: string; dosage: string; frequency: string}> {
    const medications = [];
    
    if (condition.toLowerCase().includes('pain')) {
      medications.push({
        name: 'Ibuprofen',
        dosage: '400mg',
        frequency: 'Every 6-8 hours as needed'
      });
    }

    if (condition.toLowerCase().includes('infection')) {
      medications.push({
        name: 'Antibiotic (to be prescribed)',
        dosage: 'As prescribed',
        frequency: 'As directed by physician'
      });
    }

    return medications;
  }

  private generateLifestyleRecommendations(condition: string): string[] {
    return [
      'Maintain a balanced diet rich in fruits and vegetables',
      'Exercise regularly as tolerated',
      'Get adequate sleep (7-9 hours per night)',
      'Manage stress through relaxation techniques',
      'Avoid smoking and limit alcohol consumption'
    ];
  }

  private scheduleAppointments(condition: string): Array<{type: string; frequency: string}> {
    return [
      {type: 'Primary Care Follow-up', frequency: 'In 2 weeks'},
      {type: 'Specialist Consultation (if needed)', frequency: 'As referred'},
      {type: 'Lab Work Review', frequency: 'In 1 month'}
    ];
  }

  private setTreatmentGoals(condition: string): string[] {
    return [
      'Symptom resolution or significant improvement',
      'Return to normal daily activities',
      'Prevention of complications',
      'Patient education and understanding'
    ];
  }

  private estimateTreatmentDuration(condition: string): string {
    const acuteConditions = ['infection', 'cold', 'flu'];
    const chronicConditions = ['diabetes', 'hypertension', 'arthritis'];
    
    if (acuteConditions.some(acute => condition.toLowerCase().includes(acute))) {
      return '1-2 weeks';
    } else if (chronicConditions.some(chronic => condition.toLowerCase().includes(chronic))) {
      return 'Ongoing management';
    }
    return '2-4 weeks';
  }

  private generateResearchFindings(topic: string): string[] {
    return [
      `New treatment approaches show 85% efficacy in ${topic} management`,
      `Clinical trials demonstrate significant improvement in patient outcomes`,
      `Innovative diagnostic methods improve early detection rates`,
      `Combination therapy shows promise for treatment-resistant cases`
    ];
  }

  private generateSource(): string {
    const sources = [
      'New England Journal of Medicine',
      'The Lancet',
      'JAMA - Journal of the American Medical Association',
      'Nature Medicine',
      'BMJ - British Medical Journal'
    ];
    return sources[Math.floor(Math.random() * sources.length)];
  }

  private generateRecentDate(): string {
    const date = new Date();
    date.setDate(date.getDate() - Math.floor(Math.random() * 90)); // Within last 90 days
    return date.toISOString().split('T')[0];
  }

  private assessResearchImpact(): 'high' | 'medium' | 'low' {
    const impacts = ['high', 'medium', 'low'];
    return impacts[Math.floor(Math.random() * impacts.length)] as any;
  }

  private analyzeHealthData(healthData: any): string[] {
    const insights = [];
    
    if (healthData.vitals.heartRate > 100) {
      insights.push('Elevated heart rate detected - may indicate stress or physical activity');
    }

    if (healthData.vitals.temperature > 99.5) {
      insights.push('Slightly elevated temperature - monitor for fever development');
    }

    if (healthData.lifestyle.exercise === 'sedentary') {
      insights.push('Low physical activity levels - increased exercise recommended');
    }

    return insights;
  }

  private generateHealthRecommendations(healthData: any): string[] {
    return [
      'Maintain regular exercise routine',
      'Follow a heart-healthy diet',
      'Monitor vital signs regularly',
      'Stay hydrated throughout the day',
      'Get adequate sleep for recovery'
    ];
  }

  private generateHealthAlerts(healthData: any): string[] {
    const alerts = [];
    
    if (healthData.vitals.heartRate > 120) {
      alerts.push('High heart rate - consult healthcare provider');
    }

    if (healthData.vitals.temperature > 101) {
      alerts.push('Fever detected - seek medical attention');
    }

    return alerts;
  }

  private calculateHealthScore(healthData: any): number {
    let score = 100;
    
    if (healthData.vitals.heartRate > 100) score -= 10;
    if (healthData.vitals.temperature > 99.5) score -= 15;
    if (healthData.lifestyle.exercise === 'sedentary') score -= 20;
    if (healthData.symptoms.length > 0) score -= 10 * healthData.symptoms.length;

    return Math.max(0, Math.min(100, score));
  }
}

export const healthcareService = new HealthcareService();