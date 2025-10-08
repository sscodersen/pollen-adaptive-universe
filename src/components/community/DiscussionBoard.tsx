import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { MessageSquare, Heart, Pin } from 'lucide-react';

interface Post {
  postId: string;
  userId: string;
  content: string;
  postType: string;
  likes: number;
  replies: number;
  isPinned: boolean;
  createdAt: string;
}

interface DiscussionBoardProps {
  communityId: string;
  userId: string;
}

export function DiscussionBoard({ communityId, userId }: DiscussionBoardProps) {
  const [posts, setPosts] = useState<Post[]>([]);
  const [newPost, setNewPost] = useState('');
  const [isPosting, setIsPosting] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    fetchPosts();
  }, [communityId]);

  const fetchPosts = async () => {
    try {
      const response = await fetch(`/api/community/${communityId}/posts`);
      const data = await response.json();
      if (data.success) {
        setPosts(data.posts);
      }
    } catch (error) {
      console.error('Error fetching posts:', error);
    }
  };

  const handleCreatePost = async () => {
    if (!newPost.trim()) return;

    setIsPosting(true);
    try {
      const response = await fetch(`/api/community/${communityId}/posts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId,
          content: newPost,
          postType: 'discussion'
        })
      });

      const data = await response.json();
      if (data.success) {
        toast({ title: 'Post created successfully' });
        setNewPost('');
        fetchPosts();
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to create post',
        variant: 'destructive'
      });
    } finally {
      setIsPosting(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Create a Post</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            value={newPost}
            onChange={(e) => setNewPost(e.target.value)}
            placeholder="Share your thoughts with the community..."
            rows={4}
          />
          <Button onClick={handleCreatePost} disabled={isPosting || !newPost.trim()}>
            {isPosting ? 'Posting...' : 'Post'}
          </Button>
        </CardContent>
      </Card>

      <div className="space-y-4">
        {posts.map((post) => (
          <Card key={post.postId}>
            <CardContent className="pt-6">
              <div className="space-y-4">
                {post.isPinned && (
                  <Badge className="flex items-center gap-1 w-fit">
                    <Pin className="w-3 h-3" />
                    Pinned
                  </Badge>
                )}
                <p className="text-sm">{post.content}</p>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <button className="flex items-center gap-1 hover:text-foreground">
                    <Heart className="w-4 h-4" />
                    {post.likes}
                  </button>
                  <button className="flex items-center gap-1 hover:text-foreground">
                    <MessageSquare className="w-4 h-4" />
                    {post.replies}
                  </button>
                  <span>{new Date(post.createdAt).toLocaleDateString()}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
