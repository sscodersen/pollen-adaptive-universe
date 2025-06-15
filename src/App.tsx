
import React from "react";
import { TopNav } from "@/components/TopNav";
import { MainTabs } from "@/components/MainTabs";

function App() {
  return (
    <div className="min-h-screen w-full flex flex-col font-sans bg-slate-950">
      <TopNav />
      <MainTabs />
    </div>
  );
}

export default App;
