
import React from 'react';
import { AiCoreStats } from './AiCoreStats';
import { Bot, Cpu, Database, Sparkles } from 'lucide-react';

const InfoCard = ({ icon: Icon, title, children }) => (
  <div className="bg-black/20 backdrop-blur-xl rounded-2xl p-6 border border-white/10 h-full">
    <div className="flex items-start space-x-4">
      <div className="w-10 h-10 mt-1 bg-white/10 rounded-lg flex items-center justify-center flex-shrink-0">
        <Icon className="w-5 h-5 text-cyan-300" />
      </div>
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
        <p className="text-gray-400 leading-relaxed text-sm">
          {children}
        </p>
      </div>
    </div>
  </div>
);


export const AnalyticsDashboard = () => {
  return (
    <div className="p-6 md:p-10 h-full overflow-y-auto animate-fade-in">
      <header className="mb-12 max-w-4xl mx-auto">
        <div className="flex items-center gap-4 mb-2">
            <Sparkles className="w-10 h-10 text-purple-400" />
            <h1 className="text-4xl font-bold tracking-tight text-white">Adaptive Intelligence</h1>
        </div>
        <p className="text-lg text-gray-400 md:ml-14">
          A transparent look into the Pollen core and its self-improvement mechanisms.
        </p>
      </header>
      
      <div className="max-w-4xl mx-auto flex flex-col gap-10">
        <AiCoreStats />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <InfoCard icon={Bot} title="What is Pollen?">
              Pollen is a simulated self-improving AI. It doesn't use pre-existing data but learns from scratch by generating internal "reasoning tasks" and evaluating its own success. This process allows it to continuously refine its abilities.
            </InfoCard>
            <InfoCard icon={Cpu} title="How It Works">
              The AI core constantly runs cycles of induction, deduction, and abduction. Each successful task reinforces positive pathways, increasing its performance. The "reward" metric indicates how effectively it's learning.
            </InfoCard>
            <InfoCard icon={Database} title="Performance">
              The stats above provide a real-time benchmark of its cognitive processes. A high success rate and recent performance indicate the model is effectively learning and adapting.
            </InfoCard>
            <InfoCard icon={Sparkles} title="The Goal">
                To create an autonomous, self-improving intelligence that can reason, learn, and create across any domain, pushing the boundaries of artificial cognition.
            </InfoCard>
        </div>
      </div>
    </div>
  );
};
