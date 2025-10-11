import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Lock, AlertCircle } from 'lucide-react';

interface AdminLoginProps {
  onLogin: (apiKey: string) => void;
}

export function AdminLogin({ onLogin }: AdminLoginProps) {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (!apiKey.trim()) {
      setError('Please enter an API key');
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('/api/admin/metrics', {
        headers: {
          'X-API-Key': apiKey
        }
      });

      if (response.ok) {
        sessionStorage.setItem('admin_api_key', apiKey);
        onLogin(apiKey);
      } else {
        setError('Invalid API key. Access denied.');
        setLoading(false);
      }
    } catch (err) {
      setError('Failed to verify API key. Please try again.');
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
            <Lock className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          </div>
          <CardTitle className="text-2xl">Admin Access Required</CardTitle>
          <CardDescription>
            Enter your admin API key to access the dashboard
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleLogin} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="apiKey">Admin API Key</Label>
              <Input
                id="apiKey"
                type="password"
                placeholder="Enter your API key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                disabled={loading}
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400" />
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Verifying...' : 'Access Dashboard'}
            </Button>

            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                <strong>Development Note:</strong> The default API key is set via the ADMIN_API_KEY environment variable. Contact your system administrator for access credentials.
              </p>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
