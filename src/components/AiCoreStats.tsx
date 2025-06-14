
import React, { useState, useEffect } from 'react';
import { BrainCircuit, TrendingUp, CheckCircle, Percent, BarChart } from 'lucide-react';
import { pollenAI, type CoreStats } from '../services/pollenAI';
import { LoadingSpinner } from './optimized/LoadingSpinner';

const StatCard = ({ icon: Icon, title, value, unit, colorClass }) => (
  <div className="bg-gray-900/50 rounded-lg p-4 flex items-center space-x-4 border border-gray-800/50">
    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${colorClass}`}>
      <Icon className="w-5 h-5 text-white" />
    </div>
    <div>
      <p className="text-sm text-gray-400">{title}</p>
      <p className="text-xl font-bold text-white">
        {value}
        <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>
      </p>
    </div>
  </div>
);

export const AiCoreStats = () => {
  const [stats, setStats] = useState<CoreStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      setLoading(true);
      const coreStats = await pollenAI.getCoreStats();
      setStats(coreStats);
      setLoading(false);
    };

    fetchStats();
    const interval = setInterval(fetchStats, 15000); // Refresh every 15 seconds

    return () => clearInterval(interval);
  }, []);

  if (loading && !stats) {
    return (
      <div className="bg-gray-950/50 backdrop-blur-xl rounded-xl border border-gray-800/60 p-6 flex items-center justify-center h-96">
        <LoadingSpinner />
        <span className="ml-3 text-gray-400">Syncing with AI Core...</span>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-gray-950/50 backdrop-blur-xl rounded-xl border border-gray-800/60 p-6 text-center h-96 flex flex-col items-center justify-center">
        <BrainCircuit className="w-12 h-12 text-red-500 mb-4" />
        <h3 className="text-xl font-bold text-white">Connection Error</h3>
        <p className="text-gray-400">Could not retrieve stats from the AI Core.</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-950/50 backdrop-blur-xl rounded-xl border border-gray-800/60 p-6">
      <div className="flex items-center space-x-4 mb-6">
        <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg">
          <BrainCircuit className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-white">Absolute Zero Reasoner</h3>
          <p className="text-gray-400">Real-time self-improvement statistics.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <StatCard icon={BarChart} title="Total Reasoning Tasks" value={stats.total_tasks} unit="tasks" colorClass="bg-cyan-500/80" />
        <StatCard icon={CheckCircle} title="Success Rate" value={(stats.success_rate * 100).toFixed(1)} unit="%" colorClass="bg-green-500/80" />
        <StatCard icon={TrendingUp} title="Recent Performance" value={(stats.recent_performance * 100).toFixed(1)} unit="%" colorClass="bg-purple-500/80" />
        <StatCard icon={Percent} title="Average Reward" value={stats.average_reward.toFixed(3)} unit="reward" colorClass="bg-yellow-500/80" />
      </div>

      <div>
        <h4 className="text-lg font-semibold text-white mb-3">Task Type Distribution</h4>
        <div className="space-y-3">
          {Object.entries(stats.task_types_distribution).map(([type, count]) => (
            <div key={type}>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-gray-300 capitalize">{type}</span>
                <span className="text-sm text-gray-400">{count} tasks</span>
              </div>
              <div className="w-full bg-gray-800/50 rounded-full h-2.5">
                <div className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2.5 rounded-full" style={{ width: `${(count / Math.max(stats.total_tasks, 1)) * 100}%` }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
