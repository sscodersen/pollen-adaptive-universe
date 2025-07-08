import React from 'react';
import { RefreshCw, Smartphone } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface AppStoreHeaderProps {
  loading: boolean;
  onRefresh: () => void;
}

export const AppStoreHeader: React.FC<AppStoreHeaderProps> = ({ loading, onRefresh }) => {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl border border-blue-500/30">
            <Smartphone className="w-8 h-8 text-blue-400" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">App Store</h1>
            <p className="text-gray-400">Discover and download amazing applications</p>
          </div>
        </div>
        <Button
          onClick={onRefresh}
          disabled={loading}
          className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh Apps
        </Button>
      </div>
    </div>
  );
};