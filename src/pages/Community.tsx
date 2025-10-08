import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { CommunityCard } from '@/components/community/CommunityCard';
import { DiscussionBoard } from '@/components/community/DiscussionBoard';
import { EthicsReportForm } from '@/components/ethics/EthicsReportForm';
import { TransparencyDashboard } from '@/components/ethics/TransparencyDashboard';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Users, Shield, Eye, Plus } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

export default function Community() {
  const [communities, setCommunities] = useState([]);
  const [suggestedCommunities, setSuggestedCommunities] = useState([]);
  const [selectedCommunity, setSelectedCommunity] = useState<string | null>(null);
  const [joinedCommunities, setJoinedCommunities] = useState<Set<string>>(new Set());
  const { toast } = useToast();

  const mockUserId = 'user_demo_123';

  useEffect(() => {
    fetchCommunities();
    fetchSuggestions();
  }, []);

  const fetchCommunities = async () => {
    try {
      const response = await fetch('/api/community/communities');
      const data = await response.json();
      if (data.success) {
        setCommunities(data.communities);
      }
    } catch (error) {
      console.error('Error fetching communities:', error);
    }
  };

  const fetchSuggestions = async () => {
    try {
      const response = await fetch(`/api/community/users/${mockUserId}/suggestions`);
      const data = await response.json();
      if (data.success) {
        setSuggestedCommunities(data.suggestions);
      }
    } catch (error) {
      console.error('Error fetching suggestions:', error);
    }
  };

  const handleJoinCommunity = async (communityId: string) => {
    try {
      const response = await fetch(`/api/community/${communityId}/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: mockUserId })
      });

      const data = await response.json();
      if (data.success) {
        setJoinedCommunities(prev => new Set(prev).add(communityId));
        toast({ title: 'Joined community successfully!' });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to join community',
        variant: 'destructive'
      });
    }
  };

  return (
    <div className="container mx-auto py-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold">Community Hub</h1>
          <p className="text-muted-foreground mt-2">
            Connect, share, and grow with like-minded individuals
          </p>
        </div>
        <Button>
          <Plus className="w-4 h-4 mr-2" />
          Create Community
        </Button>
      </div>

      <Tabs defaultValue="discover" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="discover">
            <Users className="w-4 h-4 mr-2" />
            Discover
          </TabsTrigger>
          <TabsTrigger value="my-communities">
            My Communities
          </TabsTrigger>
          <TabsTrigger value="ethics">
            <Shield className="w-4 h-4 mr-2" />
            AI Ethics
          </TabsTrigger>
          <TabsTrigger value="transparency">
            <Eye className="w-4 h-4 mr-2" />
            Transparency
          </TabsTrigger>
        </TabsList>

        <TabsContent value="discover" className="space-y-6">
          {suggestedCommunities.length > 0 && (
            <div>
              <h2 className="text-2xl font-semibold mb-4">Suggested for You</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {suggestedCommunities.slice(0, 3).map((suggestion: any) => (
                  <CommunityCard
                    key={suggestion.community.communityId}
                    community={suggestion.community}
                    onJoin={handleJoinCommunity}
                    isJoined={joinedCommunities.has(suggestion.community.communityId)}
                  />
                ))}
              </div>
            </div>
          )}

          <div>
            <h2 className="text-2xl font-semibold mb-4">All Communities</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {communities.map((community: any) => (
                <CommunityCard
                  key={community.communityId}
                  community={community}
                  onJoin={handleJoinCommunity}
                  isJoined={joinedCommunities.has(community.communityId)}
                />
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="my-communities" className="space-y-6">
          {selectedCommunity ? (
            <div>
              <Button variant="outline" onClick={() => setSelectedCommunity(null)} className="mb-4">
                ← Back to Communities
              </Button>
              <DiscussionBoard communityId={selectedCommunity} userId={mockUserId} />
            </div>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Your Communities</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Array.from(joinedCommunities).map((communityId) => {
                    const community = communities.find((c: any) => c.communityId === communityId);
                    return community ? (
                      <div key={communityId} onClick={() => setSelectedCommunity(communityId)}>
                        <CommunityCard
                          community={community as any}
                          onJoin={() => {}}
                          isJoined={true}
                        />
                      </div>
                    ) : null;
                  })}
                </div>
                {joinedCommunities.size === 0 && (
                  <p className="text-muted-foreground text-center py-8">
                    You haven't joined any communities yet. Explore communities to get started!
                  </p>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="ethics" className="space-y-6">
          <EthicsReportForm userId={mockUserId} />
        </TabsContent>

        <TabsContent value="transparency" className="space-y-6">
          <TransparencyDashboard userId={mockUserId} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
