import { useState } from 'react';
import { HStack, Icon, Text, Button, useDisclosure, useToast } from '@chakra-ui/react';
import { Heart, MessageCircle, Share2, Bookmark } from 'lucide-react';
import SharePostDialog from './SharePostDialog';

const PostActions = ({ post, variant = 'default' }) => {
  const [liked, setLiked] = useState(false);
  const [bookmarked, setBookmarked] = useState(false);
  const [likeCount, setLikeCount] = useState(post?.likes || Math.floor(Math.random() * 500) + 50);
  const [commentCount] = useState(post?.comments || Math.floor(Math.random() * 100) + 10);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();

  const handleLike = () => {
    const newLiked = !liked;
    setLiked(newLiked);
    setLikeCount(prev => newLiked ? prev + 1 : prev - 1);

    const storedLikes = JSON.parse(localStorage.getItem('likedPosts') || '[]');
    const postId = post?.id || post?.content_id || post?.title;
    
    if (newLiked) {
      localStorage.setItem('likedPosts', JSON.stringify([...storedLikes, postId]));
    } else {
      localStorage.setItem('likedPosts', JSON.stringify(storedLikes.filter(id => id !== postId)));
    }
  };

  const handleBookmark = () => {
    const newBookmarked = !bookmarked;
    setBookmarked(newBookmarked);

    const storedBookmarks = JSON.parse(localStorage.getItem('bookmarkedPosts') || '[]');
    const postId = post?.id || post?.content_id || post?.title;
    
    if (newBookmarked) {
      const bookmarkData = {
        id: postId,
        post: post,
        savedAt: new Date().toISOString()
      };
      localStorage.setItem('bookmarkedPosts', JSON.stringify([...storedBookmarks, bookmarkData]));
      toast({
        title: 'Post saved',
        description: 'Added to your bookmarks',
        status: 'success',
        duration: 2000,
        position: 'bottom'
      });
    } else {
      localStorage.setItem('bookmarkedPosts', JSON.stringify(storedBookmarks.filter(b => b.id !== postId)));
      toast({
        title: 'Bookmark removed',
        status: 'info',
        duration: 2000,
        position: 'bottom'
      });
    }
  };

  const handleComment = () => {
    toast({
      title: 'Coming soon',
      description: 'AI-powered comments will be available soon',
      status: 'info',
      duration: 2000
    });
  };

  if (variant === 'compact') {
    return (
      <>
        <HStack spacing={4}>
          <Button
            variant="ghost"
            size="sm"
            leftIcon={<Icon as={Heart} boxSize={4} />}
            onClick={handleLike}
            color={liked ? 'red.400' : 'gray.400'}
            _hover={{ color: liked ? 'red.300' : 'gray.300' }}
          >
            <Text fontSize="xs">{likeCount}</Text>
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            leftIcon={<Icon as={MessageCircle} boxSize={4} />}
            onClick={handleComment}
            color="gray.400"
            _hover={{ color: 'gray.300' }}
          >
            <Text fontSize="xs">{commentCount}</Text>
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            leftIcon={<Icon as={Share2} boxSize={4} />}
            onClick={onOpen}
            color="gray.400"
            _hover={{ color: 'purple.400' }}
          >
            <Text fontSize="xs">Share</Text>
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            leftIcon={<Icon as={Bookmark} boxSize={4} />}
            onClick={handleBookmark}
            color={bookmarked ? 'purple.400' : 'gray.400'}
            _hover={{ color: bookmarked ? 'purple.300' : 'gray.300' }}
          />
        </HStack>
        <SharePostDialog isOpen={isOpen} onClose={onClose} post={post} />
      </>
    );
  }

  return (
    <>
      <HStack spacing={6} justify="space-between">
        <HStack spacing={6}>
          <HStack
            spacing={2}
            cursor="pointer"
            onClick={handleLike}
            _hover={{ transform: 'scale(1.05)' }}
            transition="all 0.2s"
          >
            <Icon
              as={Heart}
              boxSize={5}
              color={liked ? 'red.400' : 'gray.400'}
              fill={liked ? 'currentColor' : 'none'}
            />
            <Text fontSize="sm" color="gray.400" fontWeight="medium">
              {likeCount}
            </Text>
          </HStack>

          <HStack
            spacing={2}
            cursor="pointer"
            onClick={handleComment}
            _hover={{ transform: 'scale(1.05)' }}
            transition="all 0.2s"
          >
            <Icon as={MessageCircle} boxSize={5} color="gray.400" />
            <Text fontSize="sm" color="gray.400" fontWeight="medium">
              {commentCount}
            </Text>
          </HStack>

          <HStack
            spacing={2}
            cursor="pointer"
            onClick={onOpen}
            _hover={{ transform: 'scale(1.05)' }}
            transition="all 0.2s"
          >
            <Icon as={Share2} boxSize={5} color="gray.400" />
            <Text fontSize="sm" color="gray.400" fontWeight="medium">
              Share
            </Text>
          </HStack>
        </HStack>

        <Icon
          as={Bookmark}
          boxSize={5}
          color={bookmarked ? 'purple.400' : 'gray.400'}
          fill={bookmarked ? 'currentColor' : 'none'}
          cursor="pointer"
          onClick={handleBookmark}
          _hover={{ transform: 'scale(1.1)' }}
          transition="all 0.2s"
        />
      </HStack>
      <SharePostDialog isOpen={isOpen} onClose={onClose} post={post} />
    </>
  );
};

export default PostActions;
