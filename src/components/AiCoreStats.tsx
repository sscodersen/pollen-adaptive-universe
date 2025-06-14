
import React, { useState, useEffect } from 'react';
import { BrainCircuit, TrendingUp, CheckCircle, Percent, Zap } from 'lucide-react';
import { pollenAI, type CoreStats } from '../services/pollenAI';
import { LoadingSpinner } from './optimized/LoadingSpinner';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';


const StatCard = ({ icon: Icon, title, value, unit, colorClass, description }) => (
  <div className={`bg-gray-900/50 rounded-xl p-5 flex flex-col justify-between border-l-4 ${colorClass}`}>
    <div>
      <div className="flex items-center space-x-3 mb-2">
        <Icon className="w-5 h-5 text-gray-300" />
        <p className="text-sm text-gray-400">{title}</p>
      </div>
      <p className="text-3xl font-bold text-white">
        {value}
        <span className="text-lg font-normal text-gray-400 ml-1">{unit}</span>
      </p>
    </div>
    <p className="text-xs text-gray-500 mt-3">{description}</p>
  </div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-950/80 backdrop-blur-sm p-2 border border-gray-700 rounded-md shadow-lg">
        <p className="label text-white font-bold">{`${label.charAt(0).toUpperCase() + label.slice(1)}`}</p>
        <p className="intro text-cyan-400">{`Tasks: ${payload[0].value}`}</p>
      </div>
    );
  }
  return null;
};

export const AiCoreStats = () => {
  const [stats, setStats] = useState<CoreStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      // Don't set loading to true on interval fetches to avoid flicker
      const coreStats = await pollenAI.getCoreStats();
      if (coreStats) {
        setStats(coreStats);
      }
      setLoading(false);
    };

    fetchStats();
    const interval = setInterval(fetchStats, 5000); // Refresh every 5 seconds for more "real-time" feel

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
  
  const chartData = stats ? Object.entries(stats.task_types_distribution).map(([name, value]) => ({
    name,
    tasks: value,
  })) : [];

  return (
    <div className="bg-gray-950/50 backdrop-blur-xl rounded-xl border border-gray-800/60 p-6 h-full flex flex-col">
      <div className="flex items-center space-x-4 mb-6">
        <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg animate-pulse">
          <BrainCircuit className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-white">Absolute Zero Reasoner</h3>
          <p className="text-gray-400">Real-time self-improvement statistics.</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard 
          icon={Zap} 
          title="Total Tasks" 
          value={stats.total_tasks.toLocaleString()} 
          unit="" 
          colorClass="border-cyan-500"
          description="Total reasoning cycles executed."
        />
        <StatCard 
          icon={CheckCircle} 
          title="Success Rate" 
          value={(stats.success_rate * 100).toFixed(1)} 
          unit="%" 
          colorClass="border-green-500"
          description="Percentage of tasks solved correctly."
        />
        <StatCard 
          icon={TrendingUp} 
          title="Performance" 
          value={(stats.recent_performance * 100).toFixed(1)} 
          unit="%" 
          colorClass="border-purple-500"
          description="Success rate over last 100 tasks."
        />
        <StatCard 
          icon={Percent} 
          title="Average Reward" 
          value={stats.average_reward.toFixed(3)} 
          unit="" 
          colorClass="border-yellow-500"
          description="Average learning gain per task."
        />
      </div>

      <div className="flex-grow">
        <h4 className="text-lg font-semibold text-white mb-3">Task Type Distribution</h4>
        <div className="w-full h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <defs>
                <linearGradient id="colorUv" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0.5}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
              <XAxis dataKey="name" tick={{ fill: '#9ca3af' }} tickLine={{ stroke: '#4b5563' }} />
              <YAxis tick={{ fill: '#9ca3af' }} tickLine={{ stroke: '#4b5563' }} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(107, 114, 128, 0.1)' }}/>
              <Bar dataKey="tasks" fill="url(#colorUv)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
