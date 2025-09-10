
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
import { SmartShopPage } from "@/components/SmartShopPage";
import { AppStorePage } from "@/components/AppStorePage";
import { UnifiedAIPlayground } from "@/components/UnifiedAIPlayground";
import { TaskAutomationPage } from "@/components/TaskAutomationPage";
import { LearningCenter } from "@/components/LearningCenter";
import { AdSpace } from "@/components/AdSpace";
import { Compass, Film, Home, TrendingUp, Award, Globe, ShoppingBag, Smartphone, Brain, GraduationCap, Zap } from "lucide-react";

// Primary navigation (top)
const primaryTabs = [
  { id: "feed", name: "Feed", icon: Home },
  { id: "explore", name: "Explore", icon: Compass },
  { id: "shop", name: "Smart Shop", icon: ShoppingBag },
  { id: "entertainment", name: "Entertainment", icon: Film },
];

// Secondary navigation (stacked below)
const secondaryTabs = [
  { id: "ai-playground", name: "AI Playground", icon: Brain },
  { id: "appstore", name: "App Store", icon: Smartphone },
  { id: "learning", name: "Learning Center", icon: GraduationCap },
  { id: "task-automation", name: "Task Automation", icon: Zap },
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
    <div className="flex flex-col h-screen liquid-gradient-animated">
      {/* Top banner ad space */}
      <AdSpace size="banner" position="top" category="tech" className="mx-4 mt-4" />
      
      <Tabs value={currentTab} onValueChange={setCurrentTab} className="flex flex-col h-full">
        <div className="sticky top-0 z-20 w-full glass-nav shrink-0">
          {/* Primary Tabs */}
          <TabsList className="w-full justify-start px-4 py-3 gap-3 glass h-auto border-none">
            {primaryTabs.map((tab) => (
              <TabsTrigger 
                key={tab.id} 
                value={tab.id} 
                className="
                  flex items-center gap-2 rounded-xl px-4 py-2.5 text-base whitespace-nowrap 
                  glass-button text-white border-white/10 
                  data-[state=active]:liquid-gradient-accent data-[state=active]:text-white
                  hover:scale-105 transition-all duration-300
                "
                data-testid={`tab-${tab.id}`}
              >
                <tab.icon className="w-5 h-5" />
                {tab.name}
              </TabsTrigger>
            ))}
          </TabsList>
          
          {/* Secondary Tabs */}
          <div className="px-4 py-2 border-t border-white/10">
            <TabsList className="w-full justify-start gap-2 glass h-auto border-none">
              {secondaryTabs.map((tab) => (
                <TabsTrigger 
                  key={tab.id} 
                  value={tab.id} 
                  className="
                    flex items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap 
                    glass-button text-white/80 border-white/10
                    data-[state=active]:liquid-gradient-secondary data-[state=active]:text-white
                    hover:scale-105 transition-all duration-300
                  "
                  data-testid={`tab-${tab.id}`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.name}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>
          
          {/* Feed categories with liquid glass design */}
          {currentTab === "feed" && (
            <div className="flex px-4 py-3 gap-2 border-t border-white/10 glass">
              {feedCategories.map((fc) => (
                <button
                  key={fc.id}
                  onClick={() => setFeedCategory(fc.id)}
                  className={`
                    flex items-center gap-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap
                    ${feedCategory === fc.id
                      ? "liquid-gradient-warm text-white"
                      : "glass-button text-white/70 hover:text-white"
                    }
                  `}
                  data-testid={`feed-category-${fc.id}`}
                >
                  <fc.icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{fc.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        
        <div className="flex-1 min-h-0 relative">
          {/* Sidebar ad space */}
          <AdSpace size="sidebar" position="right" category="business" className="absolute top-4 right-4 z-10 hidden xl:block" />
          
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
          
          <TabsContent value="task-automation" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            {/* Blank page for user's future embedding */}
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center glass-card p-8 max-w-md">
                <Zap className="w-16 h-16 text-white/60 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Task Automation</h3>
                <p className="text-white/60 mb-6">This space is reserved for your custom automation tools.</p>
                <div className="glass-button px-6 py-3 rounded-lg text-white cursor-default">
                  Ready for Integration
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="entertainment" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            <EntertainmentPage />
          </TabsContent>
          
          <TabsContent value="learning" className="flex-1 h-full data-[state=active]:flex data-[state=active]:flex-col">
            <LearningCenter />
          </TabsContent>
        </div>
        
        {/* Bottom banner ad space */}
        <AdSpace size="premium" position="bottom" category="ai" className="mx-4 mb-4" />
      </Tabs>
    </div>
  );
}
