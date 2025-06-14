
import React from 'react';
import { Sparkles } from 'lucide-react';
import { usePlayground } from '../contexts/PlaygroundContext';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from './ui/sidebar';

export const AppSidebar = () => {
  const { tabs, activeTab, setActiveTab } = usePlayground();

  return (
    <Sidebar>
      <SidebarHeader className="p-4 border-b border-slate-700/50">
        <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div className="group-data-[collapsed=icon]/sidebar-wrapper:hidden">
              <h1 className="text-xl font-bold">Pollen</h1>
              <p className="text-sm text-slate-400">LLMX Platform</p>
            </div>
          </div>
      </SidebarHeader>
      <SidebarContent className="p-2">
        <SidebarMenu>
          {tabs.map((item) => (
            <SidebarMenuItem key={item.id}>
              <SidebarMenuButton
                onClick={() => setActiveTab(item.id)}
                isActive={activeTab === item.id}
                tooltip={{children: item.name, side: 'right'}}
              >
                <item.icon className="w-5 h-5" />
                <span className="group-data-[collapsed=icon]/sidebar-wrapper:hidden">{item.name}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarContent>
    </Sidebar>
  );
};
