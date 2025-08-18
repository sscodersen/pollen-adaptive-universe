// Finance AI Service
import { pollenAdaptiveService } from './pollenAdaptiveService';

export interface MarketAnalysis {
  symbol: string;
  currentPrice: number;
  analysis: string;
  prediction: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  timeframe: string;
  keyFactors: string[];
  recommendation: 'buy' | 'hold' | 'sell';
}

export interface InvestmentStrategy {
  id: string;
  name: string;
  riskLevel: 'conservative' | 'moderate' | 'aggressive';
  expectedReturn: number;
  timeHorizon: string;
  allocation: Array<{asset: string; percentage: number}>;
  description: string;
  suitableFor: string[];
}

export interface BudgetPlan {
  id: string;
  monthlyIncome: number;
  expenses: Array<{category: string; amount: number; percentage: number}>;
  savings: {amount: number; percentage: number};
  recommendations: string[];
  goals: Array<{name: string; target: number; timeline: string}>;
}

export interface RetirementPlan {
  currentAge: number;
  retirementAge: number;
  currentSavings: number;
  monthlyContribution: number;
  projectedValue: number;
  shortfall: number;
  recommendations: string[];
  milestones: Array<{age: number; target: number}>;
}

export class FinanceService {
  async generateMarketAnalysis(symbol: string): Promise<MarketAnalysis> {
    const analysis = await pollenAdaptiveService.analyzeTrends(`stock market ${symbol}`);
    
    return {
      symbol: symbol.toUpperCase(),
      currentPrice: this.generateStockPrice(),
      analysis: this.generateAnalysisText(symbol),
      prediction: this.generatePrediction(),
      confidence: analysis.trendScore,
      timeframe: '30 days',
      keyFactors: this.generateKeyFactors(symbol),
      recommendation: this.generateRecommendation()
    };
  }

  async createInvestmentStrategy(profile: {
    age: number;
    riskTolerance: string;
    investmentGoals: string[];
    timeHorizon: string;
    investmentAmount: number;
  }): Promise<InvestmentStrategy> {
    const strategy = await pollenAdaptiveService.proposeTask(
      `Create investment strategy for ${profile.age} year old with ${profile.riskTolerance} risk tolerance`
    );

    const riskLevel = this.mapRiskTolerance(profile.riskTolerance);
    
    return {
      id: `strategy_${Date.now()}`,
      name: `${profile.riskTolerance.charAt(0).toUpperCase() + profile.riskTolerance.slice(1)} Growth Strategy`,
      riskLevel,
      expectedReturn: this.calculateExpectedReturn(riskLevel),
      timeHorizon: profile.timeHorizon,
      allocation: this.generateAllocation(riskLevel),
      description: strategy.description,
      suitableFor: this.generateSuitabilityProfile(profile)
    };
  }

  async generateBudgetPlan(income: number, expenses: Array<{category: string; amount: number}>): Promise<BudgetPlan> {
    const totalExpenses = expenses.reduce((sum, exp) => sum + exp.amount, 0);
    const savingsAmount = income - totalExpenses;
    
    const expensesWithPercentage = expenses.map(exp => ({
      ...exp,
      percentage: Math.round((exp.amount / income) * 100)
    }));

    return {
      id: `budget_${Date.now()}`,
      monthlyIncome: income,
      expenses: expensesWithPercentage,
      savings: {
        amount: Math.max(0, savingsAmount),
        percentage: Math.round((savingsAmount / income) * 100)
      },
      recommendations: this.generateBudgetRecommendations(income, totalExpenses),
      goals: this.generateFinancialGoals(income)
    };
  }

  async createRetirementPlan(profile: {
    currentAge: number;
    retirementAge: number;
    currentSavings: number;
    monthlyIncome: number;
    expectedExpenses: number;
  }): Promise<RetirementPlan> {
    const yearsToRetirement = profile.retirementAge - profile.currentAge;
    const monthlyContribution = Math.max(0, profile.monthlyIncome * 0.1); // 10% recommended
    const projectedValue = this.calculateRetirementValue(
      profile.currentSavings,
      monthlyContribution,
      yearsToRetirement
    );
    
    const retirementGoal = profile.expectedExpenses * 25; // 25x rule
    const shortfall = Math.max(0, retirementGoal - projectedValue);

    return {
      currentAge: profile.currentAge,
      retirementAge: profile.retirementAge,
      currentSavings: profile.currentSavings,
      monthlyContribution,
      projectedValue,
      shortfall,
      recommendations: this.generateRetirementRecommendations(shortfall, yearsToRetirement),
      milestones: this.generateRetirementMilestones(profile.currentAge, profile.retirementAge, retirementGoal)
    };
  }

  async analyzeMarketTrends(sector: string): Promise<{
    sector: string;
    trends: string[];
    opportunities: string[];
    risks: string[];
    outlook: string;
  }> {
    const trendAnalysis = await pollenAdaptiveService.analyzeTrends(`${sector} market trends`);
    
    return {
      sector,
      trends: this.generateSectorTrends(sector),
      opportunities: this.generateOpportunities(sector),
      risks: this.generateRisks(sector),
      outlook: trendAnalysis.insights[0] || `${sector} sector showing steady growth potential`
    };
  }

  private generateStockPrice(): number {
    return Math.round((50 + Math.random() * 450) * 100) / 100;
  }

  private generateAnalysisText(symbol: string): string {
    return `${symbol} shows strong fundamentals with positive technical indicators. Recent market movements suggest continued momentum in the near term.`;
  }

  private generatePrediction(): 'bullish' | 'bearish' | 'neutral' {
    const predictions = ['bullish', 'bearish', 'neutral'];
    return predictions[Math.floor(Math.random() * predictions.length)] as any;
  }

  private generateKeyFactors(symbol: string): string[] {
    return [
      'Strong quarterly earnings growth',
      'Positive industry outlook',
      'Favorable market conditions',
      'Technical breakout pattern',
      'Analyst upgrades'
    ];
  }

  private generateRecommendation(): 'buy' | 'hold' | 'sell' {
    const recommendations = ['buy', 'hold', 'sell'];
    return recommendations[Math.floor(Math.random() * recommendations.length)] as any;
  }

  private mapRiskTolerance(tolerance: string): 'conservative' | 'moderate' | 'aggressive' {
    if (tolerance.toLowerCase().includes('conservative') || tolerance.toLowerCase().includes('low')) {
      return 'conservative';
    } else if (tolerance.toLowerCase().includes('aggressive') || tolerance.toLowerCase().includes('high')) {
      return 'aggressive';
    }
    return 'moderate';
  }

  private calculateExpectedReturn(riskLevel: 'conservative' | 'moderate' | 'aggressive'): number {
    const returns = {
      conservative: 5.5,
      moderate: 8.0,
      aggressive: 11.5
    };
    return returns[riskLevel];
  }

  private generateAllocation(riskLevel: 'conservative' | 'moderate' | 'aggressive'): Array<{asset: string; percentage: number}> {
    const allocations = {
      conservative: [
        {asset: 'Bonds', percentage: 60},
        {asset: 'Large Cap Stocks', percentage: 30},
        {asset: 'Cash/Money Market', percentage: 10}
      ],
      moderate: [
        {asset: 'Large Cap Stocks', percentage: 40},
        {asset: 'Small Cap Stocks', percentage: 20},
        {asset: 'International Stocks', percentage: 20},
        {asset: 'Bonds', percentage: 20}
      ],
      aggressive: [
        {asset: 'Large Cap Stocks', percentage: 50},
        {asset: 'Small Cap Stocks', percentage: 25},
        {asset: 'International Stocks', percentage: 15},
        {asset: 'Growth Stocks', percentage: 10}
      ]
    };
    return allocations[riskLevel];
  }

  private generateSuitabilityProfile(profile: any): string[] {
    const suitable = ['Long-term investors'];
    
    if (profile.age < 35) {
      suitable.push('Young professionals', 'Growth-oriented investors');
    } else if (profile.age > 55) {
      suitable.push('Pre-retirees', 'Income-focused investors');
    }

    if (profile.riskTolerance === 'aggressive') {
      suitable.push('Risk-tolerant investors');
    }

    return suitable;
  }

  private generateBudgetRecommendations(income: number, expenses: number): string[] {
    const savingsRate = (income - expenses) / income;
    const recommendations = [];

    if (savingsRate < 0.1) {
      recommendations.push('Increase savings rate to at least 10% of income');
    }

    if (savingsRate < 0) {
      recommendations.push('Review and reduce expenses to avoid debt');
    }

    recommendations.push(
      'Build emergency fund covering 3-6 months of expenses',
      'Consider automating savings transfers',
      'Review and optimize recurring subscriptions'
    );

    return recommendations;
  }

  private generateFinancialGoals(income: number): Array<{name: string; target: number; timeline: string}> {
    return [
      {name: 'Emergency Fund', target: income * 6, timeline: '12 months'},
      {name: 'Down Payment for House', target: 50000, timeline: '5 years'},
      {name: 'Vacation Fund', target: 5000, timeline: '18 months'},
      {name: 'Retirement Savings', target: income * 10, timeline: '30 years'}
    ];
  }

  private calculateRetirementValue(currentSavings: number, monthlyContribution: number, years: number): number {
    const annualReturn = 0.07; // 7% average return
    const monthlyReturn = annualReturn / 12;
    const months = years * 12;

    // Future value of current savings
    const futureValueCurrent = currentSavings * Math.pow(1 + annualReturn, years);
    
    // Future value of monthly contributions (annuity)
    const futureValueContributions = monthlyContribution * 
      (Math.pow(1 + monthlyReturn, months) - 1) / monthlyReturn;

    return Math.round(futureValueCurrent + futureValueContributions);
  }

  private generateRetirementRecommendations(shortfall: number, yearsToRetirement: number): string[] {
    const recommendations = [];

    if (shortfall > 0) {
      const additionalMonthly = Math.round(shortfall / (yearsToRetirement * 12));
      recommendations.push(`Consider increasing monthly contributions by $${additionalMonthly}`);
    }

    recommendations.push(
      'Maximize employer 401(k) matching',
      'Consider increasing contributions annually',
      'Review investment allocation periodically',
      'Take advantage of catch-up contributions after age 50'
    );

    return recommendations;
  }

  private generateRetirementMilestones(currentAge: number, retirementAge: number, goal: number): Array<{age: number; target: number}> {
    const milestones = [];
    const years = retirementAge - currentAge;
    const increment = Math.floor(years / 4);

    for (let i = 1; i <= 4; i++) {
      milestones.push({
        age: currentAge + (increment * i),
        target: Math.round((goal / 4) * i)
      });
    }

    return milestones;
  }

  private generateSectorTrends(sector: string): string[] {
    return [
      `${sector} sector experiencing digital transformation`,
      'Increased consumer demand driving growth',
      'Innovation and technology adoption accelerating',
      'Regulatory changes creating new opportunities'
    ];
  }

  private generateOpportunities(sector: string): string[] {
    return [
      'Market expansion in emerging economies',
      'Technology-driven efficiency improvements',
      'New product development opportunities',
      'Strategic partnerships and mergers'
    ];
  }

  private generateRisks(sector: string): string[] {
    return [
      'Economic downturn impact on consumer spending',
      'Regulatory compliance costs',
      'Competitive pressure from new entrants',
      'Supply chain disruption risks'
    ];
  }
}

export const financeService = new FinanceService();