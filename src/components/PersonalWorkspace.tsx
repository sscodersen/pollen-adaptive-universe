
import React, { useState, useEffect } from 'react';
import { Brain, FileText, Lightbulb, Target, TrendingUp, Zap } from 'lucide-react';
import { ActivityFeed } from './ActivityFeed';
import { pollenAI } from '../services/pollenAI';

interface PersonalWorkspaceProps {
  activities: any[];
  isGenerating: boolean;
}

export const PersonalWorkspace = ({ activities, isGenerating }: PersonalWorkspaceProps) => {
  const [insights, setInsights] = useState([]);
  const [goals, setGoals] = useState([]);
  const [productivity, setProductivity] = useState(0);

  useEffect(() => {
    generatePersonalInsights();
  }, []);

  const generatePersonalInsights = async () => {
    try {
      const insightResponse = await pollenAI.generate(
        "Generate personal productivity insights and recommendations",
        "personal"
      );
      
      setInsights([
        {
          id: 1,
          title: "Focus Time Optimization",
          description: "Your most productive hours are 9-11 AM. Consider scheduling deep work during this time.",
          impact: "high",
          icon: Brain
        },
        {
          id: 2,
          title: "Learning Path Suggestion",
          description: "Based on your interests, consider exploring Advanced React Patterns next.",
          impact: "medium",
          icon: Lightbulb
        },
        {
          id: 3,
          title: "Goal Progress",
          description: "You're 75% towards your monthly learning target. Great progress!",
          impact: "high",
          icon: Target
        }
      ]);

      setGoals([
        { id: 1, title: "Complete React Advanced Course", progress: 65, deadline: "This Month" },
        { id: 2, title: "Build Portfolio Project", progress: 40, deadline: "Next Month" },
        { id: 3, title: "Learn TypeScript Patterns", progress: 20, deadline: "2 Months" }
      ]);

      setProductivity(78);
    } catch (error) {
      console.error('Failed to generate insights:', error);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Personal Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Productivity Score */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-white">Productivity Score</h3>
            <TrendingUp className="w-5 h-5 text-green-400" />
          </div>
          <div className="space-y-3">
            <div className="text-3xl font-bold text-green-400">{productivity}%</div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-green-400 to-blue-400 h-2 rounded-full transition-all duration-500"
                style={{ width: `${productivity}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-400">+12% from last week</p>
          </div>
        </div>

        {/* AI Insights */}
        <div className="lg:col-span-2 bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <Zap className="w-5 h-5 text-yellow-400" />
            <h3 className="font-semibold text-white">AI Insights</h3>
          </div>
          <div className="space-y-3">
            {insights.map((insight) => (
              <div key={insight.id} className="flex items-start space-x-3 p-3 bg-gray-700/50 rounded-lg">
                <insight.icon className={`w-5 h-5 mt-0.5 ${
                  insight.impact === 'high' ? 'text-red-400' : 'text-yellow-400'
                }`} />
                <div>
                  <h4 className="font-medium text-white text-sm">{insight.title}</h4>
                  <p className="text-gray-300 text-sm">{insight.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Goals Progress */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="font-semibold text-white mb-4">Current Goals</h3>
        <div className="space-y-4">
          {goals.map((goal) => (
            <div key={goal.id} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-white font-medium">{goal.title}</span>
                <span className="text-sm text-gray-400">{goal.deadline}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${goal.progress}%` }}
                ></div>
              </div>
              <div className="text-sm text-gray-300">{goal.progress}% complete</div>
            </div>
          ))}
        </div>
      </div>

      {/* Activity Feed */}
      <div className="bg-gray-800 rounded-xl border border-gray-700">
        <div className="p-4 border-b border-gray-700">
          <h3 className="font-semibold text-white">Personal Activity</h3>
        </div>
        <ActivityFeed 
          activities={activities.filter(a => !a.teamActivity)} 
          isGenerating={isGenerating} 
        />
      </div>
    </div>
  );
};
