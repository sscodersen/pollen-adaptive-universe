
import React, { useState } from "react";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import { SocialFeed } from "@/components/SocialFeed";
import { ExplorePage } from "@/components/ExplorePage";
import { EntertainmentPage } from "@/components/EntertainmentPage";
import { GamesPage } from "@/components/GamesPage";
import { MusicPage } from "@/components/MusicPage";
import { AdCreationPage } from "@/components/AdCreationPage";
import { Compass, Film, Gamepad2, Music, Home, TrendingUp, Award, Globe, Megaphone } from "lucide-react";

// Main navigation
const navTabs = [
  { id: "feed", name: "Feed", icon: Home },
  { id: "explore", name: "Explore", icon: Compass },
  { id: "entertainment", name: "Entertainment", icon: Film },
  { id: "games", name: "Games", icon: Gamepad2 },
  { id: "music", name: "Music", icon: Music },
  { id: "ads", name: "Create Ads", icon: Megaphone },
];

// Feed category sub-tabs
const feedCategories = [
  { id: "all", name: "All Posts", icon: Globe },
  { id: "trending", name: "Trending", icon: TrendingUp },
  { id: "high-impact", name: "High Impact", icon: Award },
];

export function MainTabs() {
  const [currentTab, setCurrentTab] = useState("feed");
  const [feedCategory, setFeedCategory] = useState("all");

  return (
    <Tabs value={currentTab} onValueChange={setCurrentTab} className="w-full flex flex-col">
      <div className="sticky top-0 z-20 w-full bg-slate-950 border-b border-slate-800/60">
        <TabsList className="w-full justify-start px-4 py-2 gap-2 bg-slate-950">
          {navTabs.map((tab) => (
            <TabsTrigger key={tab.id} value={tab.id} className="flex items-center gap-2 rounded-lg px-4 py-2 text-base">
              <tab.icon className="w-5 h-5" />
              {tab.name}
            </TabsTrigger>
          ))}
        </TabsList>
        {currentTab === "feed" && (
          <div className="flex px-4 py-3 gap-2 border-b border-slate-800/50 bg-slate-950">
            {feedCategories.map((fc) => (
              <button
                key={fc.id}
                onClick={() => setFeedCategory(fc.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  feedCategory === fc.id
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                    : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 border border-slate-700/30"
                }`}
              >
                <fc.icon className="w-4 h-4" />
                <span className="text-sm font-medium">{fc.name}</span>
              </button>
            ))}
          </div>
        )}
      </div>
      
      {/* TAB CONTENT AREAS */}
      <TabsContent value="feed" className="flex-1">
        <SocialFeed filter={feedCategory} />
      </TabsContent>
      <TabsContent value="explore" className="flex-1">
        <ExplorePage />
      </TabsContent>
      <TabsContent value="entertainment" className="flex-1">
        <EntertainmentPage />
      </TabsContent>
      <TabsContent value="games" className="flex-1">
        <GamesPage />
      </TabsContent>
      <TabsContent value="music" className="flex-1">
        <MusicPage />
      </TabsContent>
      <TabsContent value="ads" className="flex-1">
        <AdCreationPage />
      </TabsContent>
    </Tabs>
  );
}
