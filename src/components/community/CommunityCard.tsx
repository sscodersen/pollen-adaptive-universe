import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Users, Lock, Globe } from 'lucide-react';

interface Community {
  communityId: string;
  name: string;
  description: string;
  type: string;
  category: string;
  isPrivate: boolean;
  memberCount: number;
}

interface CommunityCardProps {
  community: Community;
  onJoin: (communityId: string) => void;
  isJoined?: boolean;
}

export function CommunityCard({ community, onJoin, isJoined }: CommunityCardProps) {
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="flex items-center gap-2">
              {community.name}
              {community.isPrivate ? (
                <Lock className="w-4 h-4 text-muted-foreground" />
              ) : (
                <Globe className="w-4 h-4 text-muted-foreground" />
              )}
            </CardTitle>
            <CardDescription className="mt-2">{community.description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex gap-2">
            <Badge variant="secondary">{community.type.replace('_', ' ')}</Badge>
            <Badge variant="outline">{community.category}</Badge>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Users className="w-4 h-4" />
              <span>{community.memberCount} members</span>
            </div>
            
            <Button
              onClick={() => onJoin(community.communityId)}
              disabled={isJoined}
              variant={isJoined ? "outline" : "default"}
            >
              {isJoined ? 'Joined' : 'Join Community'}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
