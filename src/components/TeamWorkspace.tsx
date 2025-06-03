
import React, { useState, useEffect } from 'react';
import { Users, MessageSquare, GitBranch, Calendar, CheckCircle, AlertCircle } from 'lucide-react';
import { ActivityFeed } from './ActivityFeed';
import { pollenAI } from '../services/pollenAI';

interface TeamWorkspaceProps {
  activities: any[];
  isGenerating: boolean;
}

export const TeamWorkspace = ({ activities, isGenerating }: TeamWorkspaceProps) => {
  const [teamMembers, setTeamMembers] = useState([]);
  const [projects, setProjects] = useState([]);
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    generateTeamData();
  }, []);

  const generateTeamData = async () => {
    try {
      await pollenAI.generate("Generate team collaboration insights", "team");
      
      setTeamMembers([
        { name: "Sarah Chen", role: "Designer", status: "online", avatar: "bg-blue-500", initial: "S" },
        { name: "Alex Kumar", role: "Developer", status: "away", avatar: "bg-green-500", initial: "A" },
        { name: "Maria Santos", role: "PM", status: "online", avatar: "bg-purple-500", initial: "M" },
        { name: "David Kim", role: "Engineer", status: "offline", avatar: "bg-orange-500", initial: "D" }
      ]);

      setProjects([
        { name: "Mobile App Redesign", progress: 75, status: "active", dueDate: "2 days" },
        { name: "API Integration", progress: 40, status: "in-review", dueDate: "1 week" },
        { name: "User Research", progress: 90, status: "completed", dueDate: "Completed" }
      ]);

      setNotifications([
        { id: 1, type: "review", message: "Sarah requested review on design system", time: "5m ago", urgent: true },
        { id: 2, type: "meeting", message: "Daily standup in 30 minutes", time: "25m", urgent: false },
        { id: 3, type: "completion", message: "Alex completed the authentication flow", time: "1h ago", urgent: false }
      ]);
    } catch (error) {
      console.error('Failed to generate team data:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-400';
      case 'away': return 'bg-yellow-400';
      case 'offline': return 'bg-gray-400';
      default: return 'bg-gray-400';
    }
  };

  const getProjectStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-blue-400';
      case 'in-review': return 'text-yellow-400';
      case 'completed': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Team Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Team Members */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <Users className="w-5 h-5 text-blue-400" />
            <h3 className="font-semibold text-white">Team Members</h3>
          </div>
          <div className="space-y-3">
            {teamMembers.map((member, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-10 h-10 ${member.avatar} rounded-full flex items-center justify-center text-white font-medium relative`}>
                    {member.initial}
                    <div className={`absolute -bottom-1 -right-1 w-3 h-3 ${getStatusColor(member.status)} rounded-full border-2 border-gray-800`}></div>
                  </div>
                  <div>
                    <div className="font-medium text-white">{member.name}</div>
                    <div className="text-sm text-gray-400">{member.role}</div>
                  </div>
                </div>
                <div className="text-xs text-gray-500 capitalize">{member.status}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Notifications */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <MessageSquare className="w-5 h-5 text-green-400" />
              <h3 className="font-semibold text-white">Notifications</h3>
            </div>
            <span className="bg-red-500 text-white text-xs px-2 py-1 rounded-full">3</span>
          </div>
          <div className="space-y-3">
            {notifications.map((notification) => (
              <div key={notification.id} className={`p-3 rounded-lg border-l-4 ${
                notification.urgent 
                  ? 'bg-red-500/10 border-red-500' 
                  : 'bg-gray-700/50 border-gray-600'
              }`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="text-white text-sm">{notification.message}</p>
                    <p className="text-gray-400 text-xs mt-1">{notification.time}</p>
                  </div>
                  {notification.urgent && (
                    <AlertCircle className="w-4 h-4 text-red-400 ml-2 flex-shrink-0" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Project Progress */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center space-x-2 mb-4">
          <GitBranch className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">Active Projects</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {projects.map((project, index) => (
            <div key={index} className="p-4 bg-gray-700/50 rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium text-white text-sm">{project.name}</h4>
                <span className={`text-xs px-2 py-1 rounded-full bg-gray-600 ${getProjectStatusColor(project.status)}`}>
                  {project.status}
                </span>
              </div>
              <div className="space-y-2">
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${project.progress}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-300">{project.progress}%</span>
                  <span className="text-gray-400">Due: {project.dueDate}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Team Activity Feed */}
      <div className="bg-gray-800 rounded-xl border border-gray-700">
        <div className="p-4 border-b border-gray-700">
          <h3 className="font-semibold text-white">Team Activity</h3>
        </div>
        <ActivityFeed 
          activities={activities.filter(a => a.teamActivity)} 
          isGenerating={isGenerating} 
        />
      </div>
    </div>
  );
};
