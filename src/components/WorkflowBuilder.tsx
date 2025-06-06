
import React, { useState } from 'react';
import { Plus, Play, Pause, Settings, Trash2, ArrowRight, Clock, Zap, CheckCircle } from 'lucide-react';

interface WorkflowNode {
  id: string;
  type: 'trigger' | 'action' | 'condition';
  name: string;
  config: Record<string, any>;
  position: { x: number; y: number };
}

interface WorkflowConnection {
  from: string;
  to: string;
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  status: 'active' | 'inactive' | 'error';
  lastRun?: string;
  runs: number;
}

export const WorkflowBuilder = () => {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const [isBuilding, setIsBuilding] = useState(false);

  const availableNodes = {
    triggers: [
      { type: 'schedule', name: 'Schedule', icon: Clock, description: 'Run on a schedule' },
      { type: 'webhook', name: 'Webhook', icon: Zap, description: 'Trigger via HTTP request' },
      { type: 'email', name: 'Email Received', icon: '📧', description: 'When email is received' },
      { type: 'file', name: 'File Upload', icon: '📄', description: 'When file is uploaded' }
    ],
    actions: [
      { type: 'send_email', name: 'Send Email', icon: '📨', description: 'Send an email' },
      { type: 'api_call', name: 'API Call', icon: '🔗', description: 'Make HTTP request' },
      { type: 'ai_generate', name: 'AI Generate', icon: '🤖', description: 'Generate content with AI' },
      { type: 'data_transform', name: 'Transform Data', icon: '🔄', description: 'Process and transform data' }
    ],
    conditions: [
      { type: 'if_then', name: 'If/Then', icon: '🔀', description: 'Conditional branching' },
      { type: 'filter', name: 'Filter', icon: '🔍', description: 'Filter data' },
      { type: 'delay', name: 'Delay', icon: '⏰', description: 'Wait for specified time' }
    ]
  };

  const createNewWorkflow = () => {
    const newWorkflow: Workflow = {
      id: `workflow-${Date.now()}`,
      name: 'New Workflow',
      description: 'Describe what this workflow does',
      nodes: [],
      connections: [],
      status: 'inactive',
      runs: 0
    };

    setWorkflows([...workflows, newWorkflow]);
    setSelectedWorkflow(newWorkflow);
    setIsBuilding(true);
  };

  const addNode = (nodeType: string, category: 'triggers' | 'actions' | 'conditions') => {
    if (!selectedWorkflow) return;

    const nodeTemplate = availableNodes[category].find(n => n.type === nodeType);
    if (!nodeTemplate) return;

    const newNode: WorkflowNode = {
      id: `node-${Date.now()}`,
      type: category.slice(0, -1) as 'trigger' | 'action' | 'condition',
      name: nodeTemplate.name,
      config: {},
      position: { x: Math.random() * 400, y: Math.random() * 300 }
    };

    const updatedWorkflow = {
      ...selectedWorkflow,
      nodes: [...selectedWorkflow.nodes, newNode]
    };

    setSelectedWorkflow(updatedWorkflow);
    updateWorkflow(updatedWorkflow);
  };

  const updateWorkflow = (workflow: Workflow) => {
    setWorkflows(workflows.map(w => w.id === workflow.id ? workflow : w));
  };

  const toggleWorkflow = (workflowId: string) => {
    const workflow = workflows.find(w => w.id === workflowId);
    if (!workflow) return;

    const updatedWorkflow = {
      ...workflow,
      status: workflow.status === 'active' ? 'inactive' : 'active' as 'active' | 'inactive'
    };

    updateWorkflow(updatedWorkflow);
  };

  const deleteWorkflow = (workflowId: string) => {
    setWorkflows(workflows.filter(w => w.id !== workflowId));
    if (selectedWorkflow?.id === workflowId) {
      setSelectedWorkflow(null);
      setIsBuilding(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Workflow Automation</h1>
          <p className="text-gray-400">Build and manage automated workflows</p>
        </div>
        <button
          onClick={createNewWorkflow}
          className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 px-4 py-2 rounded-lg font-medium text-white transition-all flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>New Workflow</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Workflow List */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4">Your Workflows</h3>
            <div className="space-y-3">
              {workflows.map((workflow) => (
                <div
                  key={workflow.id}
                  className={`p-4 rounded-lg border cursor-pointer transition-all ${
                    selectedWorkflow?.id === workflow.id
                      ? 'border-cyan-500 bg-cyan-500/10'
                      : 'border-gray-600 bg-gray-700/50 hover:bg-gray-700'
                  }`}
                  onClick={() => setSelectedWorkflow(workflow)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-white">{workflow.name}</h4>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleWorkflow(workflow.id);
                        }}
                        className={`p-1 rounded transition-colors ${
                          workflow.status === 'active' 
                            ? 'text-green-400 hover:text-green-300' 
                            : 'text-gray-400 hover:text-gray-300'
                        }`}
                      >
                        {workflow.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteWorkflow(workflow.id);
                        }}
                        className="text-red-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400 mb-2">{workflow.description}</p>
                  <div className="flex items-center justify-between text-xs">
                    <span className={`px-2 py-1 rounded-full ${
                      workflow.status === 'active' 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-gray-500/20 text-gray-400'
                    }`}>
                      {workflow.status}
                    </span>
                    <span className="text-gray-500">{workflow.runs} runs</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Workflow Builder */}
        <div className="lg:col-span-2">
          {selectedWorkflow ? (
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <input
                    type="text"
                    value={selectedWorkflow.name}
                    onChange={(e) => {
                      const updated = { ...selectedWorkflow, name: e.target.value };
                      setSelectedWorkflow(updated);
                      updateWorkflow(updated);
                    }}
                    className="text-xl font-bold text-white bg-transparent border-none outline-none"
                  />
                  <input
                    type="text"
                    value={selectedWorkflow.description}
                    onChange={(e) => {
                      const updated = { ...selectedWorkflow, description: e.target.value };
                      setSelectedWorkflow(updated);
                      updateWorkflow(updated);
                    }}
                    className="text-gray-400 bg-transparent border-none outline-none w-full"
                    placeholder="Describe what this workflow does"
                  />
                </div>
              </div>

              {/* Node Palette */}
              <div className="mb-6">
                <h4 className="text-sm font-medium text-gray-300 mb-3">Add Components</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(availableNodes).map(([category, nodes]) => (
                    <div key={category} className="space-y-2">
                      <h5 className="text-xs text-gray-400 uppercase tracking-wider">{category}</h5>
                      {nodes.map((node) => (
                        <button
                          key={node.type}
                          onClick={() => addNode(node.type, category as 'triggers' | 'actions' | 'conditions')}
                          className="w-full p-3 bg-gray-700/50 hover:bg-gray-700 border border-gray-600 rounded-lg text-left transition-colors"
                        >
                          <div className="flex items-center space-x-3">
                            <span className="text-lg">{typeof node.icon === 'string' ? node.icon : '⚡'}</span>
                            <div>
                              <p className="text-sm font-medium text-white">{node.name}</p>
                              <p className="text-xs text-gray-400">{node.description}</p>
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  ))}
                </div>
              </div>

              {/* Canvas */}
              <div className="h-96 bg-gray-900/50 rounded-lg border border-gray-600 relative overflow-hidden">
                {selectedWorkflow.nodes.length === 0 ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <Zap className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                      <p className="text-gray-400">Add components to start building your workflow</p>
                    </div>
                  </div>
                ) : (
                  <div className="p-4">
                    {selectedWorkflow.nodes.map((node) => (
                      <div
                        key={node.id}
                        className="absolute bg-gray-700 border border-gray-600 rounded-lg p-3 min-w-32"
                        style={{
                          left: node.position.x,
                          top: node.position.y
                        }}
                      >
                        <div className="flex items-center space-x-2">
                          <div className={`w-3 h-3 rounded-full ${
                            node.type === 'trigger' ? 'bg-green-400' :
                            node.type === 'action' ? 'bg-blue-400' : 'bg-yellow-400'
                          }`}></div>
                          <span className="text-sm font-medium text-white">{node.name}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 flex items-center justify-center h-96">
              <div className="text-center">
                <Settings className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400">Select a workflow to edit or create a new one</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
