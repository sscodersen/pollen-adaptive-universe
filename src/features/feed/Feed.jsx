import { useState, useEffect } from 'react';
import { Box, VStack, HStack, Text, Avatar, Icon, Image, Button, Input, InputGroup, InputRightElement, Skeleton } from '@chakra-ui/react';
import { Heart, MessageCircle, Share2, Bookmark, MoreHorizontal, ThumbsUp, Send, Smile } from 'lucide-react';

const Feed = () => {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newPost, setNewPost] = useState('');

  useEffect(() => {
    fetchPosts();
    const interval = setInterval(fetchPosts, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchPosts = async () => {
    try {
      const response = await fetch('/api/feed/posts');
      if (response.ok) {
        const data = await response.json();
        setPosts(data);
      } else {
        console.warn('Failed to fetch posts from server, using fallback data');
        setPosts(mockPosts);
      }
    } catch (error) {
      console.error('Error fetching posts:', error);
      setPosts(mockPosts);
    } finally {
      setLoading(false);
    }
  };

  const mockPosts = [
    {
      id: 1,
      user: {
        name: 'Brandon Morton',
        username: '@brandonm',
        avatar: null,
        verified: true
      },
      time: '12 Aug at 4:21 PM',
      content: 'Design Shot is an invitation to ponder on design as a living entity, capable of embodying and influencing the flow of thoughts and sensations in an ever-changing reality...',
      tags: ['#blender', '#render', '#design'],
      image: null,
      likes: 1600,
      comments: 2300,
      shares: 351,
      saved: true,
      type: 'post'
    },
    {
      id: 2,
      user: {
        name: 'Benjamin',
        username: '@benjamin',
        avatar: null,
        verified: false
      },
      time: '1h',
      content: 'What a good design! I like how you dealt with the spacing. Where can I get this file?',
      comments: [
        {
          user: '@Benjamin',
          content: 'here is the link supaui.com/download 👍',
          time: '45m'
        }
      ],
      type: 'comment_thread'
    },
    {
      id: 3,
      user: {
        name: 'Jacob',
        username: '@jacobdesign',
        avatar: null,
        verified: false
      },
      time: '8h',
      content: 'generated a new images on Midjourney',
      image: 'https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=600&h=400&fit=crop',
      likes: 856,
      comments: 124,
      type: 'image_post'
    }
  ];

  const stories = [
    { id: 1, name: 'Add story', type: 'add', avatar: null },
    { id: 2, name: 'Stephen', avatar: null, viewed: false },
    { id: 3, name: 'Edgar', avatar: null, viewed: false },
    { id: 4, name: 'Joyce', avatar: null, viewed: false },
    { id: 5, name: 'Minnie', avatar: null, viewed: false },
    { id: 6, name: 'Leon', avatar: null, viewed: false },
    { id: 7, name: 'Jordan', avatar: null, viewed: false },
  ];

  const handleLike = (postId) => {
    setPosts(posts.map(post => 
      post.id === postId 
        ? { ...post, likes: post.likes + (post.liked ? -1 : 1), liked: !post.liked }
        : post
    ));
  };

  const handleSave = (postId) => {
    setPosts(posts.map(post => 
      post.id === postId 
        ? { ...post, saved: !post.saved }
        : post
    ));
  };

  return (
    <VStack spacing={4} align="stretch" pb={4}>
      <Box
        p={4}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
      >
        <HStack spacing={3}>
          <Avatar
            size="sm"
            name="Jane"
            bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
          />
          <Input
            placeholder="What is happening?"
            value={newPost}
            onChange={(e) => setNewPost(e.target.value)}
            bg="transparent"
            border="none"
            color="white"
            _placeholder={{ color: 'gray.500' }}
            _focus={{ outline: 'none' }}
          />
        </HStack>
        <HStack justify="space-between" mt={3} pt={3} borderTop="1px solid" borderColor="whiteAlpha.200">
          <HStack spacing={3}>
            <Icon as={Smile} boxSize={5} color="gray.400" cursor="pointer" _hover={{ color: 'purple.400' }} />
            <Icon as={Image} boxSize={5} color="gray.400" cursor="pointer" _hover={{ color: 'purple.400' }} />
          </HStack>
          <Button
            size="sm"
            colorScheme="purple"
            isDisabled={!newPost.trim()}
            rightIcon={<Send size={14} />}
          >
            Post
          </Button>
        </HStack>
      </Box>

      <Box
        overflowX="auto"
        css={{
          '&::-webkit-scrollbar': {
            display: 'none',
          },
          scrollbarWidth: 'none',
        }}
      >
        <HStack spacing={3} pb={2}>
          {stories.map((story) => (
            <VStack
              key={story.id}
              spacing={1}
              cursor="pointer"
              minW="70px"
            >
              <Box
                position="relative"
                p={story.type === 'add' ? 0 : '2px'}
                borderRadius="full"
                bgGradient={story.type === 'add' ? 'none' : 'linear(to-br, purple.400, pink.400)'}
              >
                <Avatar
                  size="md"
                  name={story.name}
                  bg={story.type === 'add' ? 'whiteAlpha.200' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'}
                  border={story.type === 'add' ? '2px dashed' : 'none'}
                  borderColor="whiteAlpha.400"
                />
                {story.type === 'add' && (
                  <Box
                    position="absolute"
                    bottom="0"
                    right="0"
                    bg="purple.500"
                    borderRadius="full"
                    w="20px"
                    h="20px"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    border="2px solid black"
                  >
                    <Text fontSize="lg" color="white" lineHeight="1">+</Text>
                  </Box>
                )}
              </Box>
              <Text fontSize="xs" color="gray.400" textAlign="center" noOfLines={1}>
                {story.name}
              </Text>
            </VStack>
          ))}
        </HStack>
      </Box>

      <HStack spacing={2} p={2} bg="black" borderRadius="lg" border="1px solid" borderColor="whiteAlpha.200">
        <Button size="sm" variant="ghost" colorScheme="purple" isActive>
          Personal
        </Button>
        <Button size="sm" variant="ghost" colorScheme="gray" color="gray.400">
          All Workspace
        </Button>
        <Button size="sm" variant="ghost" colorScheme="gray" color="gray.400">
          Team
        </Button>
        <Button size="sm" variant="ghost" colorScheme="gray" color="gray.400">
          Community
        </Button>
      </HStack>

      {loading ? (
        Array(3).fill(0).map((_, idx) => (
          <Box
            key={idx}
            p={4}
            bg="black"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
          >
            <Skeleton height="200px" />
          </Box>
        ))
      ) : (
        posts.map((post) => (
          <Box
            key={post.id}
            p={4}
            bg="black"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
            _hover={{ borderColor: 'whiteAlpha.300' }}
            transition="all 0.2s"
          >
            <HStack justify="space-between" mb={3}>
              <HStack spacing={3}>
                <Avatar
                  size="sm"
                  name={post.user.name}
                  bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                />
                <VStack align="start" spacing={0}>
                  <HStack spacing={1}>
                    <Text fontSize="sm" fontWeight="bold" color="white">
                      {post.user.name}
                    </Text>
                    {post.user.verified && (
                      <Text fontSize="xs" color="blue.400">✓</Text>
                    )}
                  </HStack>
                  <Text fontSize="xs" color="gray.400">
                    {post.time}
                  </Text>
                </VStack>
              </HStack>
              <Icon as={MoreHorizontal} boxSize={5} color="gray.400" cursor="pointer" />
            </HStack>

            {post.content && (
              <Text fontSize="sm" color="white" mb={3}>
                {post.content}
              </Text>
            )}

            {post.tags && (
              <HStack spacing={2} mb={3} flexWrap="wrap">
                {post.tags.map((tag, idx) => (
                  <Text key={idx} fontSize="xs" color="blue.400" cursor="pointer" _hover={{ textDecoration: 'underline' }}>
                    {tag}
                  </Text>
                ))}
              </HStack>
            )}

            {post.image && (
              <Image
                src={post.image}
                alt="Post image"
                borderRadius="lg"
                mb={3}
                w="100%"
                h="300px"
                objectFit="cover"
              />
            )}

            {post.comments && post.type === 'comment_thread' && (
              <Box pl={4} borderLeft="2px solid" borderColor="whiteAlpha.200" ml={2} mb={3}>
                {post.comments.map((comment, idx) => (
                  <Box key={idx} mb={2}>
                    <HStack spacing={2} mb={1}>
                      <Text fontSize="xs" color="purple.400" fontWeight="bold">
                        {comment.user}
                      </Text>
                      <Text fontSize="xs" color="gray.500">
                        {comment.time}
                      </Text>
                    </HStack>
                    <Text fontSize="sm" color="white">
                      {comment.content}
                    </Text>
                  </Box>
                ))}
              </Box>
            )}

            <HStack justify="space-between" pt={3} borderTop="1px solid" borderColor="whiteAlpha.200">
              <HStack spacing={4}>
                <HStack
                  spacing={1}
                  cursor="pointer"
                  onClick={() => handleLike(post.id)}
                  _hover={{ color: 'red.400' }}
                >
                  <Icon
                    as={Heart}
                    boxSize={5}
                    color={post.liked ? 'red.400' : 'gray.400'}
                    fill={post.liked ? 'currentColor' : 'none'}
                  />
                  {post.likes && (
                    <Text fontSize="xs" color="gray.400">
                      {post.likes.toLocaleString()}
                    </Text>
                  )}
                </HStack>
                <HStack spacing={1} cursor="pointer" _hover={{ color: 'blue.400' }}>
                  <Icon as={MessageCircle} boxSize={5} color="gray.400" />
                  {post.comments && typeof post.comments === 'number' && (
                    <Text fontSize="xs" color="gray.400">
                      {post.comments.toLocaleString()}
                    </Text>
                  )}
                </HStack>
                <HStack spacing={1} cursor="pointer" _hover={{ color: 'green.400' }}>
                  <Icon as={Share2} boxSize={5} color="gray.400" />
                  {post.shares && (
                    <Text fontSize="xs" color="gray.400">
                      {post.shares}
                    </Text>
                  )}
                </HStack>
              </HStack>
              <Icon
                as={Bookmark}
                boxSize={5}
                color={post.saved ? 'purple.400' : 'gray.400'}
                fill={post.saved ? 'currentColor' : 'none'}
                cursor="pointer"
                onClick={() => handleSave(post.id)}
              />
            </HStack>
          </Box>
        ))
      )}
    </VStack>
  );
};

export default Feed;
