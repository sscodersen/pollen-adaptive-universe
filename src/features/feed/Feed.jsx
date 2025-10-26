import { useState, useEffect } from 'react';
import { Box, VStack, HStack, Text, Avatar, Icon, Image, Button, Badge, Skeleton } from '@chakra-ui/react';
import { Eye, TrendingUp, Zap, MoreHorizontal, Star } from 'lucide-react';

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
      views: 24567,
      engagement: 87,
      qualityScore: 92,
      trending: true,
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
      views: 8234,
      engagement: 65,
      qualityScore: 78,
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
      views: 15789,
      engagement: 73,
      qualityScore: 85,
      trending: true,
      type: 'image_post'
    }
  ];

  const getQualityColor = (score) => {
    if (score >= 90) return 'green';
    if (score >= 70) return 'blue';
    if (score >= 50) return 'purple';
    return 'gray';
  };

  const getQualityLabel = (score) => {
    if (score >= 90) return 'Excellent';
    if (score >= 70) return 'High Quality';
    if (score >= 50) return 'Good';
    return 'Standard';
  };

  return (
    <VStack spacing={4} align="stretch" pb={4}>
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

            <HStack justify="space-between" mb={3}>
              <HStack spacing={2}>
                <Badge colorScheme={getQualityColor(post.qualityScore || 75)} fontSize="xs" px={2} py={1}>
                  <HStack spacing={1}>
                    <Icon as={Star} boxSize={3} />
                    <Text>{post.qualityScore || 75}</Text>
                  </HStack>
                </Badge>
                <Badge colorScheme="purple" fontSize="xs" px={2} py={1} variant="subtle">
                  {getQualityLabel(post.qualityScore || 75)}
                </Badge>
                {post.trending && (
                  <Badge colorScheme="orange" fontSize="xs" px={2} py={1}>
                    <HStack spacing={1}>
                      <Icon as={TrendingUp} boxSize={3} />
                      <Text>Trending</Text>
                    </HStack>
                  </Badge>
                )}
              </HStack>
            </HStack>

            {post.content && (
              <Text fontSize="sm" color="white" mb={3}>
                {post.content}
              </Text>
            )}

            {post.tags && (
              <HStack spacing={2} mb={3} flexWrap="wrap">
                {post.tags.map((tag, idx) => (
                  <Badge key={idx} colorScheme="blue" variant="subtle" fontSize="xs" cursor="pointer" _hover={{ bg: 'blue.900' }}>
                    {tag}
                  </Badge>
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
                <HStack spacing={1}>
                  <Icon as={Eye} boxSize={5} color="gray.400" />
                  <Text fontSize="xs" color="gray.400">
                    {(post.views || Math.floor(Math.random() * 50000) + 5000).toLocaleString()} views
                  </Text>
                </HStack>
                <HStack spacing={1}>
                  <Icon as={Zap} boxSize={5} color="purple.400" />
                  <Text fontSize="xs" color="gray.400">
                    {post.engagement || Math.floor(Math.random() * 100)}% engagement
                  </Text>
                </HStack>
              </HStack>
            </HStack>
          </Box>
        ))
      )}
    </VStack>
  );
};

export default Feed;
