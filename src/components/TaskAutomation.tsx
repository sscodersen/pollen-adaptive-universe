
import React, { useState, useEffect } from 'react';
import { Bot, Play, Pause, Settings, Zap, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface Workflow {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'paused' | 'completed' | 'error';
  triggers: string[];
  actions: string[];
  lastRun: string;
  success_rate: number;
}

interface TaskAutomationProps {
  isGenerating?: boolean;
}

export const TaskAutomation = ({ isGenerating = true }: TaskAutomationProps) => {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [activeWorkflows, setActiveWorkflows] = useState(0);
  const [totalExecutions, setTotalExecutions] = useState(0);

  useEffect(() => {
    generateWorkflows();
  }, []);

  const generateWorkflows = async () => {
    const sampleWorkflows: Workflow[] = [
      {
        id: '1',
        name: 'Social Media Automation',
        description: 'Automatically post content across platforms and engage with followers',
        status: 'running',
        triggers: ['Schedule', 'New Content'],
        actions: ['Post to Twitter', 'Post to LinkedIn', 'Analyze Engagement'],
        lastRun: '5 minutes ago',
        success_rate: 94
      },
      {
        id: '2',
        name: 'Data Processing Pipeline',
        description: 'Extract, transform, and load data from multiple sources',
        status: 'completed',
        triggers: ['File Upload', 'API Webhook'],
        actions: ['Clean Data', 'Transform Format', 'Update Database'],
        lastRun: '2 hours ago',
        success_rate: 98
      },
      {
        id: '3',
        name: 'Email Campaign Manager',
        description: 'Automated email sequences based on user behavior',
        status: 'paused',
        triggers: ['User Signup', 'Purchase'],
        actions: ['Send Welcome Email', 'Track Opens', 'Follow Up'],
        lastRun: '1 day ago',
        success_rate: 89
      }
    ];

    setWorkflows(sampleWorkflows);
    setActiveWorkflows(sampleWorkflows.filter(w => w.status === 'running').length);
    setTotalExecutions(1247);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-green-400';
      case 'paused': return 'text-yellow-400';
      case 'completed': return 'text-blue-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Play className="w-4 h-4" />;
      case 'paused': return <Pause className="w-4 h-4" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'error': return <AlertCircle className="w-4 h-4" />;
      default: return <Settings className="w-4 h-4" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-3 mb-4">
            <Bot className="w-8 h-8 text-cyan-400" />
            <div>
              <h3 className="font-semibold text-white">Active Workflows</h3>
              <p className="text-2xl font-bold text-cyan-400">{activeWorkflows}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-3 mb-4">
            <Zap className="w-8 h-8 text-yellow-400" />
            <div>
              <h3 className="font-semibold text-white">Total Executions</h3>
              <p className="text-2xl font-bold text-yellow-400">{totalExecutions.toLocaleString()}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-3 mb-4">
            <Clock className="w-8 h-8 text-green-400" />
            <div>
              <h3 className="font-semibold text-white">Time Saved</h3>
              <p className="text-2xl font-bold text-green-400">240hrs</p>
            </div>
          </div>
        </div>
      </div>

      {/* Workflow Builder */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Quick Workflow Builder</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <h4 className="font-medium text-white mb-2">Trigger</h4>
            <select className="w-full bg-gray-600 border border-gray-500 rounded px-3 py-2 text-white">
              <option>Schedule</option>
              <option>Webhook</option>
              <option>File Upload</option>
              <option>Email Received</option>
            </select>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <h4 className="font-medium text-white mb-2">Action</h4>
            <select className="w-full bg-gray-600 border border-gray-500 rounded px-3 py-2 text-white">
              <option>Send Email</option>
              <option>Process Data</option>
              <option>Update Database</option>
              <option>Create Report</option>
            </select>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <button className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 px-4 py-2 rounded-lg font-medium text-white transition-all">
              Create Workflow
            </button>
          </div>
        </div>
      </div>

      {/* Active Workflows */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Your Workflows</h3>
        <div className="space-y-4">
          {workflows.map((workflow) => (
            <div key={workflow.id} className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <div className={`${getStatusColor(workflow.status)}`}>
                    {getStatusIcon(workflow.status)}
                  </div>
                  <div>
                    <h4 className="font-medium text-white">{workflow.name}</h4>
                    <p className="text-sm text-gray-400">{workflow.description}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-400">Success Rate</p>
                  <p className="font-bold text-green-400">{workflow.success_rate}%</p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                <div>
                  <p className="text-xs text-gray-400 mb-1">Triggers</p>
                  <div className="flex space-x-2">
                    {workflow.triggers.map((trigger, index) => (
                      <span key={index} className="px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded">
                        {trigger}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-400 mb-1">Actions</p>
                  <div className="flex space-x-2">
                    {workflow.actions.map((action, index) => (
                      <span key={index} className="px-2 py-1 bg-purple-500/20 text-purple-300 text-xs rounded">
                        {action}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <p className="text-xs text-gray-400">Last run: {workflow.lastRun}</p>
                <div className="flex space-x-2">
                  <button className="px-3 py-1 bg-gray-600 hover:bg-gray-500 text-white text-xs rounded transition-colors">
                    Edit
                  </button>
                  <button className={`px-3 py-1 text-xs rounded transition-colors ${
                    workflow.status === 'running' 
                      ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' 
                      : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                  }`}>
                    {workflow.status === 'running' ? 'Pause' : 'Start'}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
