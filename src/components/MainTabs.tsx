
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
import { SmartShopPage } from "@/components/SmartShopPage";
import { AppStorePage } from "@/components/AppStorePage";
import { AIPlayground } from "@/components/AIPlayground";
import { LearningCenter } from "@/components/LearningCenter";
import { Compass, Film, Gamepad2, Music, Home, TrendingUp, Award, Globe, Megaphone, ShoppingBag, Smartphone, Brain, GraduationCap, Zap, BarChart3 } from "lucide-react";
import { PollenDashboard } from "@/components/PollenDashboard";
import TrendDashboard from "@/components/TrendDashboard";

// Main navigation
const navTabs = [
  { id: "feed", name: "Feed", icon: Home },
  { id: "explore", name: "Explore", icon: Compass },
  { id: "shop", name: "Smart Shop", icon: ShoppingBag },
  { id: "appstore", name: "App Store", icon: Smartphone },
  { id: "entertainment", name: "Entertainment", icon: Film },
  { id: "games", name: "Games", icon: Gamepad2 },
  { id: "music", name: "Music", icon: Music },
  { id: "ai-playground", name: "AI Playground", icon: Brain },
  { id: "pollen", name: "Pollen AI", icon: Zap },
  { id: "trends", name: "Trend Intelligence", icon: BarChart3 },
  { id: "learning", name: "Learning Center", icon: GraduationCap },
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
      <div className="sticky top-0 z-20 w-full bg-surface-primary border-b border-border">
        <TabsList className="w-full justify-start px-4 py-2 gap-2 bg-surface-primary">
          {navTabs.map((tab) => (
            <TabsTrigger key={tab.id} value={tab.id} className="flex items-center gap-2 rounded-lg px-4 py-2 text-base">
              <tab.icon className="w-5 h-5" />
              {tab.name}
            </TabsTrigger>
          ))}
        </TabsList>
        {currentTab === "feed" && (
          <div className="flex px-4 py-3 gap-2 border-b border-border/50 bg-surface-primary">
            {feedCategories.map((fc) => (
              <button
                key={fc.id}
                onClick={() => setFeedCategory(fc.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  feedCategory === fc.id
                    ? "bg-primary/20 text-primary border border-primary/30"
                    : "bg-surface-secondary text-muted-foreground hover:bg-surface-tertiary border border-border"
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
      <TabsContent value="shop" className="flex-1">
        <SmartShopPage />
      </TabsContent>
      <TabsContent value="appstore" className="flex-1">
        <AppStorePage />
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
      <TabsContent value="ai-playground" className="flex-1">
        <AIPlayground />
      </TabsContent>
      <TabsContent value="pollen" className="flex-1">
        <PollenDashboard />
      </TabsContent>
      <TabsContent value="trends" className="flex-1">
        <TrendDashboard />
      </TabsContent>
      <TabsContent value="learning" className="flex-1">
        <LearningCenter />
      </TabsContent>
      <TabsContent value="ads" className="flex-1">
        <AdCreationPage />
      </TabsContent>
    </Tabs>
  );
}
