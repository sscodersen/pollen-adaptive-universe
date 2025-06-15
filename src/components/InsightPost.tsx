
import React from 'react';
import { Badge } from './ui/badge';
import { 
  Lightbulb, 
  Zap, 
  TrendingUp, 
  BrainCircuit, 
  AlertTriangle,
} from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { PLATFORM_CONFIG } from '../lib/platformConfig';

interface IntelligenceInsight {
    id: string;
    type: 'breakthrough' | 'optimization' | 'prediction' | 'correlation' | 'anomaly';
    title: string;
    description: string;
    significance: number;
    domains: string[];
    actionable: boolean;
    timestamp: string;
    confidence: number;
}

interface InsightPostProps {
  insight: IntelligenceInsight;
}

const domainColorClasses: { [key: string]: { bg: string, text: string, border: string } } = {
    purple: { bg: 'bg-purple-500/10', text: 'text-purple-300', border: 'border-purple-500/20' },
    cyan: { bg: 'bg-cyan-500/10', text: 'text-cyan-300', border: 'border-cyan-500/20' },
    green: { bg: 'bg-green-500/10', text: 'text-green-300', border: 'border-green-500/20' },
    orange: { bg: 'bg-orange-500/10', text: 'text-orange-300', border: 'border-orange-500/20' },
    violet: { bg: 'bg-violet-500/10', text: 'text-violet-300', border: 'border-violet-500/20' },
    pink: { bg: 'bg-pink-500/10', text: 'text-pink-300', border: 'border-pink-500/20' },
    system: { bg: 'bg-gray-500/10', text: 'text-gray-300', border: 'border-gray-500/20' },
    prediction: { bg: 'bg-indigo-500/10', text: 'text-indigo-300', border: 'border-indigo-500/20' },
    analytics: { bg: 'bg-blue-500/10', text: 'text-blue-300', border: 'border-blue-500/20' },
    default: { bg: 'bg-slate-700/50', text: 'text-slate-300', border: 'border-slate-600' }
};

const InsightPost: React.FC<InsightPostProps> = ({ insight }) => {
  const getInsightIcon = () => {
    const icons = {
      breakthrough: <Lightbulb className="w-5 h-5 text-yellow-400" />,
      optimization: <Zap className="w-5 h-5 text-cyan-400" />,
      prediction: <TrendingUp className="w-5 h-5 text-green-400" />,
      correlation: <BrainCircuit className="w-5 h-5 text-purple-400" />,
      anomaly: <AlertTriangle className="w-5 h-5 text-red-400" />,
    };
    return icons[insight.type] || <Lightbulb className="w-5 h-5 text-gray-400" />;
  };

  const getSignificanceColor = () => {
    if (insight.significance > 9.5) return 'border-red-500/50';
    if (insight.significance > 9.0) return 'border-orange-500/50';
    if (insight.significance > 8.0) return 'border-yellow-500/50';
    return 'border-slate-700/50';
  };

  return (
    <Card className={`bg-slate-800/40 border ${getSignificanceColor()} rounded-xl hover:bg-slate-800/70 transition-colors duration-300 animate-fade-in`}>
      <CardContent className="p-4">
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0 mt-1">{getInsightIcon()}</div>
          <div className="flex-1">
            <div className="flex justify-between items-start">
              <h3 className="text-md font-semibold text-white mb-1">{insight.title}</h3>
              <Badge variant="outline" className="text-xs capitalize border-slate-600 text-slate-300">
                {insight.type}
              </Badge>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed mb-3">{insight.description}</p>
            <div className="flex flex-wrap gap-2 mb-3">
              {insight.domains.map(domain => {
                const domainConfig = PLATFORM_CONFIG.domains[domain as keyof typeof PLATFORM_CONFIG.domains];
                const colorKey = domainConfig ? domainConfig.color : domain;
                const colorClasses = domainColorClasses[colorKey as keyof typeof domainColorClasses] || domainColorClasses.default;
                
                return (
                  <Badge key={domain} variant="secondary" className={`${colorClasses.bg} ${colorClasses.text} ${colorClasses.border}`}>
                    {domainConfig?.name || domain}
                  </Badge>
                );
              })}
            </div>
            <div className="flex items-center justify-between text-xs text-slate-500">
              <div className="flex items-center space-x-3">
                <span className="text-green-400 font-medium">
                  Significance: {insight.significance.toFixed(1)}
                </span>
                <span className="text-cyan-400">
                  Confidence: {insight.confidence.toFixed(1)}%
                </span>
              </div>
              <span className="capitalize">{insight.timestamp}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default InsightPost;
