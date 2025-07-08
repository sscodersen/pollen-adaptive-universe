import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Label } from '@/components/ui/label';
import { 
  Settings, User, Shield, Palette, Volume2, Download, 
  Search, Filter, Trash2, RefreshCw, Save, Bell,
  Eye, Database, Zap
} from 'lucide-react';
import { storageService, UserPreferences } from '@/services/storageService';
import { useToast } from '@/hooks/use-toast';

export const SettingsPage: React.FC = () => {
  const [preferences, setPreferences] = useState<UserPreferences | null>(null);
  const [loading, setSaving] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = () => {
    const prefs = storageService.getUserPreferences();
    setPreferences(prefs);
  };

  const savePreferences = async () => {
    if (!preferences) return;
    
    setSaving(true);
    try {
      storageService.setUserPreferences(preferences);
      storageService.trackEvent('settings_saved', { preferences });
      toast({
        title: 'Settings Saved',
        description: 'Your preferences have been updated successfully.',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to save settings. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = () => {
    const defaultPrefs: UserPreferences = {
      theme: 'dark',
      notifications: true,
      autoplay: false,
      contentFilter: 'all',
      language: 'en',
      privacy: {
        analytics: true,
        recommendations: true,
        dataCollection: false
      }
    };
    setPreferences(defaultPrefs);
    toast({
      title: 'Settings Reset',
      description: 'All settings have been reset to defaults.',
    });
  };

  const clearAllData = async () => {
    try {
      await storageService.clearAllData();
      toast({
        title: 'Data Cleared',
        description: 'All local data has been removed.',
        variant: 'destructive',
      });
      window.location.reload();
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to clear data. Please try again.',
        variant: 'destructive',
      });
    }
  };

  if (!preferences) {
    return (
      <div className="p-6 space-y-6 animate-fade-in">
        <Card className="bg-card border-border">
          <CardContent className="p-6">
            <div className="animate-pulse space-y-4">
              <div className="h-4 bg-muted rounded w-1/4"></div>
              <div className="h-8 bg-muted rounded w-1/2"></div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Settings</h1>
          <p className="text-muted-foreground">Customize your platform experience</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={resetSettings}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <Button onClick={savePreferences} disabled={loading}>
            <Save className="w-4 h-4 mr-2" />
            {loading ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="general" className="space-y-4">
        <TabsList className="bg-muted">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="appearance">Appearance</TabsTrigger>
          <TabsTrigger value="privacy">Privacy</TabsTrigger>
          <TabsTrigger value="content">Content</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-4">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="flex items-center text-foreground">
                <Settings className="w-5 h-5 mr-2 text-primary" />
                General Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-sm font-medium text-foreground">Notifications</Label>
                  <p className="text-sm text-muted-foreground">Receive updates and alerts</p>
                </div>
                <Switch
                  checked={preferences.notifications}
                  onCheckedChange={(checked) =>
                    setPreferences({ ...preferences, notifications: checked })
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-sm font-medium text-foreground">Auto-play Content</Label>
                  <p className="text-sm text-muted-foreground">Automatically play media content</p>
                </div>
                <Switch
                  checked={preferences.autoplay}
                  onCheckedChange={(checked) =>
                    setPreferences({ ...preferences, autoplay: checked })
                  }
                />
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium text-foreground">Language</Label>
                <Select
                  value={preferences.language}
                  onValueChange={(value) =>
                    setPreferences({ ...preferences, language: value })
                  }
                >
                  <SelectTrigger className="bg-background border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="es">Español</SelectItem>
                    <SelectItem value="fr">Français</SelectItem>
                    <SelectItem value="de">Deutsch</SelectItem>
                    <SelectItem value="it">Italiano</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="appearance" className="space-y-4">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="flex items-center text-foreground">
                <Palette className="w-5 h-5 mr-2 text-primary" />
                Appearance Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label className="text-sm font-medium text-foreground">Theme</Label>
                <Select
                  value={preferences.theme}
                  onValueChange={(value: 'light' | 'dark' | 'auto') =>
                    setPreferences({ ...preferences, theme: value })
                  }
                >
                  <SelectTrigger className="bg-background border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">Light</SelectItem>
                    <SelectItem value="dark">Dark</SelectItem>
                    <SelectItem value="auto">Auto</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="p-4 bg-muted/50 rounded-lg border border-border">
                <p className="text-sm text-muted-foreground">
                  Theme changes will be applied in future versions of the platform.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="privacy" className="space-y-4">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="flex items-center text-foreground">
                <Shield className="w-5 h-5 mr-2 text-primary" />
                Privacy & Data
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-sm font-medium text-foreground">Analytics</Label>
                  <p className="text-sm text-muted-foreground">Help improve the platform with usage data</p>
                </div>
                <Switch
                  checked={preferences.privacy.analytics}
                  onCheckedChange={(checked) =>
                    setPreferences({
                      ...preferences,
                      privacy: { ...preferences.privacy, analytics: checked }
                    })
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-sm font-medium text-foreground">Personalized Recommendations</Label>
                  <p className="text-sm text-muted-foreground">Use your activity to personalize content</p>
                </div>
                <Switch
                  checked={preferences.privacy.recommendations}
                  onCheckedChange={(checked) =>
                    setPreferences({
                      ...preferences,
                      privacy: { ...preferences.privacy, recommendations: checked }
                    })
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-sm font-medium text-foreground">Data Collection</Label>
                  <p className="text-sm text-muted-foreground">Allow collection of additional usage data</p>
                </div>
                <Switch
                  checked={preferences.privacy.dataCollection}
                  onCheckedChange={(checked) =>
                    setPreferences({
                      ...preferences,
                      privacy: { ...preferences.privacy, dataCollection: checked }
                    })
                  }
                />
              </div>

              <div className="p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
                <div className="flex items-start space-x-2">
                  <Eye className="w-4 h-4 text-yellow-500 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-foreground">Anonymous Platform</p>
                    <p className="text-sm text-muted-foreground">
                      All data is stored locally on your device. No personal information is collected or transmitted.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="content" className="space-y-4">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="flex items-center text-foreground">
                <Filter className="w-5 h-5 mr-2 text-primary" />
                Content Preferences
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label className="text-sm font-medium text-foreground">Content Filter</Label>
                <Select
                  value={preferences.contentFilter}
                  onValueChange={(value: 'all' | 'curated' | 'strict') =>
                    setPreferences({ ...preferences, contentFilter: value })
                  }
                >
                  <SelectTrigger className="bg-background border-border">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Content</SelectItem>
                    <SelectItem value="curated">Curated Only</SelectItem>
                    <SelectItem value="strict">Strict Filtering</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-sm text-muted-foreground">
                  Control what type of content appears in your feeds
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-3 bg-primary/5 rounded-lg border border-primary/20">
                  <h4 className="font-medium text-foreground mb-1">All Content</h4>
                  <p className="text-sm text-muted-foreground">See everything available</p>
                </div>
                <div className="p-3 bg-secondary/5 rounded-lg border border-secondary/20">
                  <h4 className="font-medium text-foreground mb-1">Curated Only</h4>
                  <p className="text-sm text-muted-foreground">Hand-picked quality content</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="advanced" className="space-y-4">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="flex items-center text-foreground">
                <Database className="w-5 h-5 mr-2 text-primary" />
                Data Management
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-muted/50 rounded-lg border border-border">
                  <h4 className="font-medium text-foreground mb-2">Storage Usage</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Local data stored on your device
                  </p>
                  <Badge variant="secondary">~2.3 MB used</Badge>
                </div>
                
                <div className="p-4 bg-muted/50 rounded-lg border border-border">
                  <h4 className="font-medium text-foreground mb-2">Cache Status</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Temporary files for faster loading
                  </p>
                  <Badge variant="secondary">~890 KB cached</Badge>
                </div>
              </div>

              <div className="space-y-4">
                <div className="p-4 bg-destructive/10 rounded-lg border border-destructive/20">
                  <div className="flex items-start space-x-3">
                    <Trash2 className="w-5 h-5 text-destructive mt-0.5" />
                    <div className="flex-1">
                      <h4 className="font-medium text-foreground">Clear All Data</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Remove all locally stored data including preferences, cache, and analytics.
                        This action cannot be undone.
                      </p>
                      <Button 
                        variant="destructive" 
                        size="sm" 
                        onClick={clearAllData}
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        Clear All Data
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="flex items-center text-foreground">
                <Zap className="w-5 h-5 mr-2 text-primary" />
                Performance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="p-4 bg-muted/50 rounded-lg border border-border">
                <p className="text-sm text-muted-foreground">
                  Performance optimizations are automatically applied based on your device capabilities.
                  Virtual scrolling and caching strategies are enabled for optimal experience.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};