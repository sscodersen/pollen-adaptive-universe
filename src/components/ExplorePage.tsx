
import React, { useState } from 'react';
import { Search, TrendingUp, Compass, Filter, Globe, Zap, Users, Calendar } from 'lucide-react';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

const trendingTopics = [
  { name: 'Climate Technology', posts: 12450, growth: '+23%' },
  { name: 'AI Consciousness', posts: 8920, growth: '+45%' },
  { name: 'Space Colonization', posts: 6780, growth: '+18%' },
  { name: 'Digital Currency', posts: 15600, growth: '+12%' },
  { name: 'Quantum Computing', posts: 4320, growth: '+67%' },
  { name: 'Gene Therapy', posts: 3890, growth: '+34%' },
  { name: 'Virtual Reality', posts: 9850, growth: '+29%' },
  { name: 'Sustainable Energy', posts: 11200, growth: '+15%' }
];

const newsResults = [
  {
    title: 'Breakthrough in Fusion Energy Achieved',
    source: 'Science Daily',
    time: '2h ago',
    category: 'Science',
    snippet: 'Scientists at MIT achieve sustained fusion reaction with net energy gain for the first time in history...'
  },
  {
    title: 'New AI Model Shows Human-Level Reasoning',
    source: 'Tech Review',
    time: '4h ago',
    category: 'Technology',
    snippet: 'Latest language model demonstrates unprecedented reasoning capabilities in complex problem solving...'
  },
  {
    title: 'Global Climate Summit Reaches Historic Agreement',
    source: 'Reuters',
    time: '6h ago',
    category: 'Environment',
    snippet: '195 nations commit to aggressive carbon reduction targets with binding enforcement mechanisms...'
  },
  {
    title: 'Breakthrough Drug Reverses Alzheimer\'s Symptoms',
    source: 'Medical Journal',
    time: '8h ago',
    category: 'Health',
    snippet: 'Phase III trials show 89% improvement in cognitive function for early-stage patients...'
  }
];

const discoveryCategories = [
  { name: 'Science & Research', icon: Zap, count: 2340, color: 'bg-blue-500/20 text-blue-300' },
  { name: 'Technology', icon: Globe, count: 4567, color: 'bg-cyan-500/20 text-cyan-300' },
  { name: 'Society & Culture', icon: Users, count: 3421, color: 'bg-purple-500/20 text-purple-300' },
  { name: 'Business & Economy', icon: TrendingUp, count: 2890, color: 'bg-green-500/20 text-green-300' },
  { name: 'Environment', icon: Globe, count: 1987, color: 'bg-emerald-500/20 text-emerald-300' },
  { name: 'Health & Medicine', icon: Zap, count: 2156, color: 'bg-red-500/20 text-red-300' }
];

export function ExplorePage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');

  return (
    <div className="flex-1 bg-gray-950 min-h-screen">
      {/* Header */}
      <div className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                <Compass className="w-8 h-8 text-cyan-400" />
                Explore
              </h1>
              <p className="text-gray-400">Discover trending topics, breaking news, and global conversations</p>
            </div>
          </div>
          
          {/* Search and Filters */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <Input
                type="text"
                placeholder="Search topics, news, people, trends..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-12 h-12 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 focus:border-cyan-500 text-lg"
              />
            </div>
            <div className="flex gap-2">
              {['all', 'trending', 'news', 'people'].map((filter) => (
                <Button
                  key={filter}
                  variant={activeFilter === filter ? "default" : "outline"}
                  onClick={() => setActiveFilter(filter)}
                  className={`capitalize ${
                    activeFilter === filter 
                      ? 'bg-cyan-500 hover:bg-cyan-600' 
                      : 'border-gray-700 text-gray-300 hover:bg-gray-800'
                  }`}
                >
                  {filter}
                </Button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-8">
          {/* Breaking News */}
          <div>
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
              <Calendar className="w-6 h-6 text-red-400" />
              Breaking News
            </h2>
            <div className="space-y-4">
              {newsResults.map((news, index) => (
                <div key={index} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      news.category === 'Science' ? 'bg-blue-500/20 text-blue-300' :
                      news.category === 'Technology' ? 'bg-cyan-500/20 text-cyan-300' :
                      news.category === 'Environment' ? 'bg-green-500/20 text-green-300' :
                      'bg-purple-500/20 text-purple-300'
                    }`}>
                      {news.category}
                    </span>
                    <span className="text-gray-400 text-sm">{news.time}</span>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">{news.title}</h3>
                  <p className="text-gray-300 mb-3">{news.snippet}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">{news.source}</span>
                    <Button size="sm" variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                      Read More
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Discovery Categories */}
          <div>
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
              <Filter className="w-6 h-6 text-cyan-400" />
              Explore Categories
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {discoveryCategories.map((category, index) => (
                <div key={index} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-colors cursor-pointer">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-3 rounded-lg ${category.color}`}>
                        <category.icon className="w-6 h-6" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">{category.name}</h3>
                        <p className="text-sm text-gray-400">{category.count.toLocaleString()} posts</p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-8">
          {/* Trending Topics */}
          <div>
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-orange-400" />
              Trending Now
            </h2>
            <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4">
              <div className="space-y-4">
                {trendingTopics.map((topic, index) => (
                  <div key={index} className="flex items-center justify-between hover:bg-gray-800/50 rounded-lg p-3 transition-colors cursor-pointer">
                    <div>
                      <h4 className="font-medium text-white">{topic.name}</h4>
                      <p className="text-sm text-gray-400">{topic.posts.toLocaleString()} posts</p>
                    </div>
                    <span className="text-green-400 text-sm font-medium">{topic.growth}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          <div>
            <h2 className="text-xl font-bold text-white mb-4">Platform Stats</h2>
            <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Active Users</span>
                <span className="text-white font-semibold">2.4M</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Posts Today</span>
                <span className="text-white font-semibold">156K</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Trending Topics</span>
                <span className="text-white font-semibold">847</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Global Reach</span>
                <span className="text-white font-semibold">195 Countries</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
