
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { Calendar, Mail, FileText, CheckSquare, Clock, Plus, Bot } from 'lucide-react';
import { useApp } from '../contexts/AppContext';

const Tasks = () => {
  const { state, dispatch } = useApp();
  const [activeTab, setActiveTab] = useState('calendar');
  const [newTask, setNewTask] = useState('');

  const tasks = [
    { id: '1', title: 'Quarterly Review Meeting', type: 'meeting', time: '2:00 PM', status: 'pending' },
    { id: '2', title: 'Email Newsletter Draft', type: 'content', time: '4:30 PM', status: 'in-progress' },
    { id: '3', title: 'Client Proposal Review', type: 'document', time: '9:00 AM', status: 'completed' }
  ];

  const addTask = () => {
    if (!newTask.trim()) return;
    // Simulate adding task
    setNewTask('');
  };

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Task Executor</h1>
          <p className="text-slate-400">AI-powered task automation and scheduling</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="space-y-2">
              {[
                { id: 'calendar', name: 'Calendar', icon: Calendar },
                { id: 'emails', name: 'Email Assistant', icon: Mail },
                { id: 'documents', name: 'Document Automation', icon: FileText },
                { id: 'tasks', name: 'Task Management', icon: CheckSquare }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-all ${
                    activeTab === tab.id
                      ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300'
                      : 'bg-slate-800/50 border border-slate-700/50 text-slate-300 hover:bg-slate-700/50'
                  }`}
                >
                  <tab.icon className="w-5 h-5" />
                  <span className="text-sm font-medium">{tab.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Content */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
              {activeTab === 'calendar' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Smart Calendar</h2>
                  <div className="space-y-4">
                    {tasks.map((task) => (
                      <div key={task.id} className="flex items-center justify-between bg-slate-700/30 rounded-lg p-4">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            task.status === 'completed' ? 'bg-green-400' :
                            task.status === 'in-progress' ? 'bg-yellow-400' : 'bg-slate-400'
                          }`}></div>
                          <div>
                            <h3 className="font-medium">{task.title}</h3>
                            <p className="text-sm text-slate-400">{task.time}</p>
                          </div>
                        </div>
                        <Bot className="w-5 h-5 text-cyan-400" />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === 'emails' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Email Assistant</h2>
                  <div className="space-y-4">
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <h3 className="font-medium mb-2">Draft Email Templates</h3>
                      <p className="text-sm text-slate-400">AI-generated templates based on your communication style</p>
                    </div>
                    <div className="bg-slate-700/30 rounded-lg p-4">
                      <h3 className="font-medium mb-2">Smart Responses</h3>
                      <p className="text-sm text-slate-400">Suggested replies for incoming emails</p>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'tasks' && (
                <div>
                  <h2 className="text-xl font-semibold mb-4">Task Management</h2>
                  <div className="flex space-x-2 mb-4">
                    <input
                      type="text"
                      value={newTask}
                      onChange={(e) => setNewTask(e.target.value)}
                      placeholder="Add a new task..."
                      className="flex-1 bg-slate-700/50 border border-slate-600/50 rounded-lg px-4 py-2 text-white placeholder-slate-400"
                    />
                    <button
                      onClick={addTask}
                      className="bg-cyan-500 hover:bg-cyan-600 px-4 py-2 rounded-lg transition-colors"
                    >
                      <Plus className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Tasks;
