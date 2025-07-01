
import React, { useState, useEffect } from 'react';
import { Activity, CheckCircle, XCircle, Clock } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface PythonScriptStatusProps {
  scriptType: 'music' | 'games' | 'entertainment' | 'shop' | 'ads';
}

export const PythonScriptStatus: React.FC<PythonScriptStatusProps> = ({ scriptType }) => {
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');

  useEffect(() => {
    // Check if Python script is running
    const checkConnection = async () => {
      setStatus('connecting');
      try {
        const response = await fetch(`http://localhost:8000/health/${scriptType}`);
        setStatus(response.ok ? 'connected' : 'disconnected');
      } catch {
        setStatus('disconnected');
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [scriptType]);

  const getStatusConfig = () => {
    switch (status) {
      case 'connected':
        return {
          icon: CheckCircle,
          color: 'text-green-400',
          bg: 'bg-green-500/20',
          text: 'Python Script Connected'
        };
      case 'connecting':
        return {
          icon: Clock,
          color: 'text-yellow-400',
          bg: 'bg-yellow-500/20',
          text: 'Connecting...'
        };
      default:
        return {
          icon: XCircle,
          color: 'text-red-400',
          bg: 'bg-red-500/20',
          text: 'Python Script Offline'
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <Badge className={`${config.bg} ${config.color} border-gray-600`}>
      <Icon className="w-3 h-3 mr-1" />
      {config.text}
    </Badge>
  );
};
