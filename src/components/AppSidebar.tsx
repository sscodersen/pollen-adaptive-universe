
import React from "react";
import {
  Home,
  Sparkles,
  Compass,
  Film,
  Gamepad2,
  Music,
  Settings,
  Zap,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

const items = [
  { title: "Feed", url: "/", icon: Home },
  { title: "Generate", url: "#", icon: Sparkles },
  { title: "Explore", url: "#", icon: Compass },
  { title: "Pollen AI", url: "/pollen", icon: Zap },
  { title: "Entertainment", url: "#", icon: Film },
  { title: "Games", url: "#", icon: Gamepad2 },
  { title: "Music", url: "#", icon: Music },
  { title: "Settings", url: "#", icon: Settings },
];

export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Pollen AI</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <a href={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
