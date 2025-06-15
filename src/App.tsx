
import React from "react";
import { SidebarProvider, SidebarTrigger, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { TopNav } from "@/components/TopNav";
import { SocialFeed } from "@/components/SocialFeed";

function App() {
  return (
    <SidebarProvider>
      <div className="min-h-screen w-full flex font-sans bg-slate-950">
        <AppSidebar />
        <SidebarInset>
          <TopNav />
          {/* Main platform content */}
          <div className="flex-1 h-full min-h-screen flex flex-col bg-slate-950">
            <SocialFeed />
          </div>
          <SidebarTrigger className="fixed top-4 left-4 z-30 md:hidden" />
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
}

export default App;
