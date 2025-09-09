import React, { useState, useEffect, useCallback } from 'react';
import { 
  Zap, 
  Settings, 
  Play, 
  Pause, 
  Square, 
  Plus, 
  ExternalLink, 
  Code, 
  Globe,
  Upload,
  Download,
  Share2,
  Timer,
  CheckCircle,
  AlertCircle,
  Activity
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface AutomationTask {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'paused' | 'stopped' | 'completed' | 'error';
  type: 'workflow' | 'api' | 'scraping' | 'data_processing' | 'notification';
  embedUrl?: string;
  lastRun: string;
  nextRun?: string;
  success: number;
  errors: number;
  config: any;
}

const taskTypes = [
  {
    id: 'workflow',
    name: 'Workflow Automation',
    icon: Zap,
    description: 'Create multi-step automated workflows',
    color: 'from-blue-500 to-cyan-500'
  },
  {
    id: 'api',
    name: 'API Integration',
    icon: Code,
    description: 'Connect and automate API calls',
    color: 'from-green-500 to-emerald-500'
  },
  {
    id: 'scraping',
    name: 'Web Scraping',
    icon: Globe,
    description: 'Extract data from websites automatically',
    color: 'from-purple-500 to-pink-500'
  },
  {
    id: 'data_processing',
    name: 'Data Processing',
    icon: Activity,
    description: 'Process and transform data pipelines',
    color: 'from-orange-500 to-red-500'
  },
  {
    id: 'notification',
    name: 'Smart Notifications',
    icon: AlertCircle,
    description: 'Automated alerts and notifications',
    color: 'from-indigo-500 to-purple-500'
  }
];

export function TaskAutomationPage() {
  const [tasks, setTasks] = useState<AutomationTask[]>([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [isCreating, setIsCreating] = useState(false);
  const [newTask, setNewTask] = useState({
    name: '',
    description: '',
    type: 'workflow' as const,
    config: {}
  });

  // Mock data for demonstration
  useEffect(() => {
    const mockTasks: AutomationTask[] = [
      {
        id: '1',
        name: 'Daily Analytics Report',
        description: 'Generate and email daily analytics reports',
        status: 'running',
        type: 'workflow',
        embedUrl: `${window.location.origin}/automation/embed/daily-analytics`,
        lastRun: '2 hours ago',
        nextRun: 'Tomorrow 9:00 AM',
        success: 47,
        errors: 2,
        config: { schedule: 'daily', recipients: ['team@company.com'] }
      },
      {
        id: '2',
        name: 'Social Media Monitor',
        description: 'Monitor brand mentions across social platforms',
        status: 'running',
        type: 'scraping',
        embedUrl: `${window.location.origin}/automation/embed/social-monitor`,
        lastRun: '15 minutes ago',
        nextRun: 'Every 30 minutes',
        success: 234,
        errors: 12,
        config: { keywords: ['brand', 'product'], platforms: ['twitter', 'reddit'] }
      },
      {
        id: '3',
        name: 'Inventory Sync',
        description: 'Sync inventory data between systems',
        status: 'paused',
        type: 'api',
        embedUrl: `${window.location.origin}/automation/embed/inventory-sync`,
        lastRun: '1 day ago',
        success: 89,
        errors: 5,
        config: { source: 'ERP System', target: 'E-commerce Platform' }
      }
    ];
    setTasks(mockTasks);
  }, []);

  const getStatusColor = (status: AutomationTask['status']) => {
    switch (status) {
      case 'running': return 'text-green-400 bg-green-500/20';
      case 'paused': return 'text-yellow-400 bg-yellow-500/20';
      case 'stopped': return 'text-gray-400 bg-gray-500/20';
      case 'completed': return 'text-blue-400 bg-blue-500/20';
      case 'error': return 'text-red-400 bg-red-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getStatusIcon = (status: AutomationTask['status']) => {
    switch (status) {
      case 'running': return <Play className="w-4 h-4" />;
      case 'paused': return <Pause className="w-4 h-4" />;
      case 'stopped': return <Square className="w-4 h-4" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'error': return <AlertCircle className="w-4 h-4" />;
      default: return <Timer className="w-4 h-4" />;
    }
  };

  const handleCreateTask = () => {
    if (!newTask.name.trim()) return;

    const task: AutomationTask = {
      id: Date.now().toString(),
      name: newTask.name,
      description: newTask.description,
      status: 'stopped',
      type: newTask.type,
      embedUrl: `${window.location.origin}/automation/embed/${newTask.name.toLowerCase().replace(/\s+/g, '-')}`,
      lastRun: 'Never',
      success: 0,
      errors: 0,
      config: newTask.config
    };

    setTasks(prev => [task, ...prev]);
    setNewTask({ name: '', description: '', type: 'workflow', config: {} });
    setIsCreating(false);
  };

  const toggleTaskStatus = (taskId: string) => {
    setTasks(prev => prev.map(task => {
      if (task.id === taskId) {
        const newStatus = task.status === 'running' ? 'paused' : 'running';
        return { ...task, status: newStatus };
      }
      return task;
    }));
  };

  return (
    <div className="flex-1 bg-gray-950 min-h-0 flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border-b border-gray-800/50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-2xl border border-indigo-500/30">
                <Zap className="w-8 h-8 text-indigo-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Task Automation</h1>
                <p className="text-gray-400">Create, manage, and embed automated workflows</p>
              </div>
            </div>
            <Button
              onClick={() => setIsCreating(true)}
              className="bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600"
            >
              <Plus className="w-4 h-4 mr-2" />
              New Automation
            </Button>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-4 gap-6 mt-6">
            <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
              <div className="flex items-center space-x-2 text-green-400 mb-2">
                <Play className="w-5 h-5" />
                <span className="font-semibold">Active</span>
              </div>
              <p className="text-2xl font-bold text-white">{tasks.filter(t => t.status === 'running').length}</p>
            </div>
            <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
              <div className="flex items-center space-x-2 text-blue-400 mb-2">
                <CheckCircle className="w-5 h-5" />
                <span className="font-semibold">Success Rate</span>
              </div>
              <p className="text-2xl font-bold text-white">94.2%</p>
            </div>
            <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
              <div className="flex items-center space-x-2 text-purple-400 mb-2">
                <Activity className="w-5 h-5" />
                <span className="font-semibold">Total Runs</span>
              </div>
              <p className="text-2xl font-bold text-white">{tasks.reduce((sum, task) => sum + task.success + task.errors, 0)}</p>
            </div>
            <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
              <div className="flex items-center space-x-2 text-cyan-400 mb-2">
                <Globe className="w-5 h-5" />
                <span className="font-semibold">Embeddable</span>
              </div>
              <p className="text-2xl font-bold text-white">{tasks.filter(t => t.embedUrl).length}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 max-w-7xl mx-auto p-6 w-full">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3 mb-8 bg-gray-900/50 border border-gray-800/50">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="create">Create New</TabsTrigger>
            <TabsTrigger value="embed">Embed & Share</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-white">Your Automations</h2>
              
              {tasks.map((task) => (
                <div key={task.id} className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-4">
                      <div className={`flex items-center space-x-2 px-3 py-1 rounded-lg ${getStatusColor(task.status)}`}>
                        {getStatusIcon(task.status)}
                        <span className="capitalize font-medium">{task.status}</span>
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">{task.name}</h3>
                        <p className="text-gray-400">{task.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => toggleTaskStatus(task.id)}
                        className="border-gray-700"
                      >
                        {task.status === 'running' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </Button>
                      <Button size="sm" variant="outline" className="border-gray-700">
                        <Settings className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Last Run:</span>
                      <p className="text-white">{task.lastRun}</p>
                    </div>
                    <div>
                      <span className="text-gray-400">Next Run:</span>
                      <p className="text-white">{task.nextRun || 'Not scheduled'}</p>
                    </div>
                    <div>
                      <span className="text-gray-400">Success:</span>
                      <p className="text-green-400">{task.success}</p>
                    </div>
                    <div>
                      <span className="text-gray-400">Errors:</span>
                      <p className="text-red-400">{task.errors}</p>
                    </div>
                  </div>

                  {task.embedUrl && (
                    <div className="mt-4 pt-4 border-t border-gray-800/50">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-400">Embeddable URL:</span>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => navigator.clipboard.writeText(task.embedUrl || '')}
                          className="border-gray-700"
                        >
                          <Share2 className="w-4 h-4 mr-2" />
                          Copy Link
                        </Button>
                      </div>
                      <div className="mt-2 p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
                        <code className="text-cyan-400 text-sm break-all">{task.embedUrl}</code>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </TabsContent>

          {/* Create Tab */}
          <TabsContent value="create" className="space-y-6">
            <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 space-y-6">
              <h2 className="text-2xl font-bold text-white">Create New Automation</h2>
              
              {/* Task Types */}
              <div>
                <label className="text-lg font-semibold text-white mb-4 block">Choose Automation Type</label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {taskTypes.map((type) => (
                    <button
                      key={type.id}
                      onClick={() => setNewTask(prev => ({ ...prev, type: type.id as any }))}
                      className={`p-4 rounded-xl border transition-all ${
                        newTask.type === type.id
                          ? 'border-purple-500/50 bg-purple-500/10'
                          : 'border-gray-700/50 bg-gray-800/30 hover:bg-gray-800/50'
                      }`}
                    >
                      <div className={`p-2 bg-gradient-to-br ${type.color} bg-opacity-20 rounded-lg mb-3 w-fit`}>
                        <type.icon className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="font-semibold text-white mb-2">{type.name}</h3>
                      <p className="text-sm text-gray-400">{type.description}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Task Details */}
              <div className="space-y-4">
                <div>
                  <label className="text-white font-medium mb-2 block">Automation Name</label>
                  <Input
                    value={newTask.name}
                    onChange={(e) => setNewTask(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="Enter automation name..."
                    className="bg-gray-800/50 border-gray-700 text-white"
                  />
                </div>
                <div>
                  <label className="text-white font-medium mb-2 block">Description</label>
                  <Textarea
                    value={newTask.description}
                    onChange={(e) => setNewTask(prev => ({ ...prev, description: e.target.value }))}
                    placeholder="Describe what this automation does..."
                    className="bg-gray-800/50 border-gray-700 text-white"
                    rows={3}
                  />
                </div>
              </div>

              <Button
                onClick={handleCreateTask}
                className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
                disabled={!newTask.name.trim()}
              >
                <Plus className="w-4 h-4 mr-2" />
                Create Automation
              </Button>
            </div>
          </TabsContent>

          {/* Embed Tab */}
          <TabsContent value="embed" className="space-y-6">
            <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 space-y-6">
              <h2 className="text-2xl font-bold text-white">Embed & Share Automations</h2>
              <p className="text-gray-400">
                Make your automations accessible via direct URLs that can be embedded in other applications or shared with teams.
              </p>

              <div className="space-y-4">
                {tasks.filter(t => t.embedUrl).map((task) => (
                  <div key={task.id} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-semibold text-white">{task.name}</h3>
                      <div className="flex space-x-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => window.open(task.embedUrl, '_blank')}
                          className="border-gray-700"
                        >
                          <ExternalLink className="w-4 h-4 mr-2" />
                          Open
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => navigator.clipboard.writeText(task.embedUrl || '')}
                          className="border-gray-700"
                        >
                          <Share2 className="w-4 h-4 mr-2" />
                          Copy
                        </Button>
                      </div>
                    </div>
                    <div className="bg-gray-900/50 rounded p-3 border border-gray-700/30">
                      <code className="text-cyan-400 text-sm break-all">{task.embedUrl}</code>
                    </div>
                  </div>
                ))}
              </div>

              {tasks.filter(t => t.embedUrl).length === 0 && (
                <div className="text-center py-8">
                  <Globe className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">No embeddable automations yet. Create some automations to get started!</p>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}