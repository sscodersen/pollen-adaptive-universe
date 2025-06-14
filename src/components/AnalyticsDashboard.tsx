
import React from 'react';
import { AiCoreStats } from './AiCoreStats';
import { Bot, Cpu, Database } from 'lucide-react';

const InfoCard = ({ icon: Icon, title, children }) => (
  <div className="bg-white/5 rounded-lg p-6 border border-white/10">
    <div className="flex items-center space-x-4 mb-3">
      <Icon className="w-6 h-6 text-cyan-400" />
      <h3 className="text-xl font-semibold text-white">{title}</h3>
    </div>
    <p className="text-gray-400 leading-relaxed">
      {children}
    </p>
  </div>
);


export const AnalyticsDashboard = () => {
  return (
    <div className="p-6 md:p-10 h-full overflow-y-auto animate-fade-in">
      <header className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight text-white">AI Analytics & Insights</h1>
        <p className="text-lg text-gray-400 mt-2">
          A transparent look into the Pollen Intelligence core and its self-improvement mechanisms.
        </p>
      </header>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <AiCoreStats />
        </div>
        
        <div className="space-y-6">
          <InfoCard icon={Bot} title="What is Pollen?">
            Pollen is a simulated self-improving AI. It doesn't use pre-existing data but learns from scratch by generating internal "reasoning tasks" and evaluating its own success. This process, called Adaptive Intelligence, allows it to continuously refine its problem-solving and content-generation abilities.
          </InfoCard>
          <InfoCard icon={Cpu} title="How It Works">
            The AI core constantly runs cycles of induction, deduction, and abduction to build its understanding. Each successful task reinforces positive pathways, increasing its performance over time. The "reward" metric indicates how effectively it's learning.
          </InfoCard>
          <InfoCard icon={Database} title="Performance">
            The stats on the left provide a real-time benchmark of its cognitive processes. A high success rate and recent performance indicate the model is effectively learning and adapting. The task distribution shows which types of reasoning it's currently focused on. This is not a traditional benchmark but a measure of its internal learning efficiency.
          </InfoCard>
        </div>
      </div>
    </div>
  );
};
