
import React, { useState } from "react";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import { SocialFeed } from "@/components/SocialFeed";
import { ExplorePage } from "@/components/ExplorePage";
import { SmartShopPage } from "@/components/SmartShopPage";
import { AppStorePage } from "@/components/AppStorePage";
import { UnifiedAIPlayground } from "@/components/UnifiedAIPlayground";
import { GamingPage } from "@/components/GamingPage";
import { Compass, Home, TrendingUp, Award, Globe, ShoppingBag, Smartphone, Brain, Gamepad2 } from "lucide-react";

// Primary navigation (top)
const primaryTabs = [
  { id: "feed", name: "Feed", icon: Home },
  { id: "explore", name: "Explore", icon: Compass },
  { id: "shop", name: "Smart Shop", icon: ShoppingBag },
];

// Secondary navigation (stacked below)
const secondaryTabs = [
  { id: "ai-playground", name: "AI Playground", icon: Brain },
  { id: "appstore", name: "App Store", icon: Smartphone },
  { id: "gaming", name: "My Games", icon: Gamepad2 },
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
    <div className="flex flex-col h-screen">
      <Tabs value={currentTab} onValueChange={setCurrentTab} className="flex flex-col h-full">
        <div className="sticky top-0 z-20 w-full bg-surface-primary border-b border-border shrink-0">
          {/* Primary Tabs */}
          <TabsList className="w-full justify-start px-4 py-2 gap-2 bg-surface-primary h-auto">
            {primaryTabs.map((tab) => (
              <TabsTrigger key={tab.id} value={tab.id} className="flex items-center gap-2 rounded-lg px-4 py-2 text-base whitespace-nowrap">
                <tab.icon className="w-5 h-5" />
                {tab.name}
              </TabsTrigger>
            ))}
          </TabsList>
          
          {/* Secondary Tabs */}
          <div className="px-4 py-2 border-t border-border/30">
            <TabsList className="w-full justify-start gap-2 bg-surface-secondary/50 h-auto">
              {secondaryTabs.map((tab) => (
                <TabsTrigger key={tab.id} value={tab.id} className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm whitespace-nowrap">
                  <tab.icon className="w-4 h-4" />
                  {tab.name}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>
          {currentTab === "feed" && (
            <div className="flex px-4 py-3 gap-2 border-b border-border/50 bg-surface-primary">
              {feedCategories.map((fc) => (
                <button
                  key={fc.id}
                  onClick={() => setFeedCategory(fc.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
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
        
        <div className="flex-1 min-h-0">{/* Content container with proper flex */}
      
          {/* TAB CONTENT AREAS */}
          <TabsContent value="feed" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            <SocialFeed filter={feedCategory} />
          </TabsContent>
          <TabsContent value="explore" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            <ExplorePage />
          </TabsContent>
          <TabsContent value="shop" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            <SmartShopPage />
          </TabsContent>
          <TabsContent value="appstore" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            <AppStorePage />
          </TabsContent>
          <TabsContent value="ai-playground" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            <UnifiedAIPlayground />
          </TabsContent>
          <TabsContent value="gaming" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            <GamingPage />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
