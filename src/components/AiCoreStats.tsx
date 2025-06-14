
import React, { useState, useEffect } from 'react';
import { BrainCircuit, TrendingUp, CheckCircle, Percent, Zap, Activity } from 'lucide-react';
import { pollenAI, type CoreStats } from '../services/pollenAI';
import { LoadingSpinner } from './optimized/LoadingSpinner';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  TooltipProps
} from 'recharts';


const StatCard = ({ icon: Icon, title, value, unit, colorClass, description }) => (
    <div className={`relative bg-black/20 rounded-xl p-5 flex flex-col justify-between border border-white/10`}>
    <div>
      <div className="flex items-center space-x-3 mb-2">
        <Icon className={`w-5 h-5 ${colorClass}`} />
        <p className="text-sm text-gray-400">{title}</p>
      </div>
      <p className="text-3xl font-bold text-white">
        {value}
        <span className="text-lg font-normal text-gray-400 ml-1.5">{unit}</span>
      </p>
    </div>
    <p className="text-xs text-gray-500 mt-3 h-8">{description}</p>
  </div>
);

const CustomTooltip = ({ active, payload, label }: TooltipProps<number, string>) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-950/80 backdrop-blur-sm p-2 border border-gray-700 rounded-md shadow-lg">
        <p className="label text-white font-bold">{`${String(label).charAt(0).toUpperCase() + String(label).slice(1)}`}</p>
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
      <div className="bg-black/20 backdrop-blur-xl rounded-xl border border-white/10 p-6 flex items-center justify-center h-96">
        <LoadingSpinner />
        <span className="ml-3 text-gray-400">Syncing with AI Core...</span>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-black/20 backdrop-blur-xl rounded-xl border border-white/10 p-6 text-center h-96 flex flex-col items-center justify-center">
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
    <div className="bg-black/20 backdrop-blur-xl rounded-2xl border border-white/10 p-6 h-full flex flex-col">
      <div className="flex items-center space-x-4 mb-6">
        <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg">
          <BrainCircuit className="w-7 h-7 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-white">Core Statistics</h3>
          <p className="text-gray-400">Real-time self-improvement metrics.</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard 
          icon={Zap} 
          title="Total Tasks" 
          value={stats.total_tasks.toLocaleString()} 
          unit="" 
          colorClass="text-cyan-400"
          description="Total reasoning cycles executed."
        />
        <StatCard 
          icon={CheckCircle} 
          title="Success Rate" 
          value={(stats.success_rate * 100).toFixed(1)} 
          unit="%" 
          colorClass="text-green-400"
          description="Percentage of tasks solved correctly."
        />
        <StatCard 
          icon={TrendingUp} 
          title="Performance" 
          value={(stats.recent_performance * 100).toFixed(1)} 
          unit="%" 
          colorClass="text-purple-400"
          description="Success rate over last 100 tasks."
        />
        <StatCard 
          icon={Percent} 
          title="Average Reward" 
          value={stats.average_reward.toFixed(3)} 
          unit="" 
          colorClass="text-yellow-400"
          description="Average learning gain per task."
        />
      </div>

      <div className="flex-grow">
        <div className="flex items-center gap-3 mb-3">
          <Activity className="w-5 h-5 text-gray-300"/>
          <h4 className="text-lg font-semibold text-white">Task Type Distribution</h4>
        </div>
        <div className="w-full h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <defs>
                <linearGradient id="colorUv" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#c084fc" stopOpacity={0.3}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 12 }} tickLine={{ stroke: '#4b5563' }} axisLine={{ stroke: '#4b5563' }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} tickLine={{ stroke: '#4b5563' }} axisLine={{ stroke: '#4b5563' }} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(192, 132, 252, 0.1)' }}/>
              <Bar dataKey="tasks" fill="url(#colorUv)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
