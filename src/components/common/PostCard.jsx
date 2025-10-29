import { Box, VStack, HStack, Text, Icon, Image, Badge } from '@chakra-ui/react';
import { Eye, TrendingUp, Zap, MoreHorizontal, Star } from 'lucide-react';

const PostCard = ({ post, showImage = true }) => {
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

  const qualityScore = post.qualityScore || post.adaptive_score?.overall || 75;
  const views = post.views || post.engagement_metrics?.views || Math.floor(Math.random() * 50000) + 5000;
  const engagement = post.engagement || post.adaptive_score?.overall || Math.floor(Math.random() * 100);

  return (
    <Box
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
          <Box
            w="32px"
            h="32px"
            borderRadius="full"
            bgGradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
            display="flex"
            alignItems="center"
            justifyContent="center"
          >
            <Text fontSize="xs" fontWeight="bold" color="white">
              {post.source ? post.source.charAt(0) : 'A'}
            </Text>
          </Box>
          <VStack align="start" spacing={0}>
            <Text fontSize="sm" fontWeight="bold" color="white">
              {post.source || 'Anonymous'}
            </Text>
            <Text fontSize="xs" color="gray.400">
              {post.time || post.published_at || 'recently'}
            </Text>
          </VStack>
        </HStack>
        <Icon as={MoreHorizontal} boxSize={5} color="gray.400" cursor="pointer" />
      </HStack>

      <HStack justify="space-between" mb={3} flexWrap="wrap">
        <HStack spacing={2}>
          <Badge colorScheme={getQualityColor(qualityScore)} fontSize="xs" px={2} py={1}>
            <HStack spacing={1}>
              <Icon as={Star} boxSize={3} />
              <Text>{Math.round(qualityScore)}</Text>
            </HStack>
          </Badge>
          <Badge colorScheme="purple" fontSize="xs" px={2} py={1} variant="subtle">
            {getQualityLabel(qualityScore)}
          </Badge>
          {(post.trending || post.category === 'trends') && (
            <Badge colorScheme="orange" fontSize="xs" px={2} py={1}>
              <HStack spacing={1}>
                <Icon as={TrendingUp} boxSize={3} />
                <Text>Trending</Text>
              </HStack>
            </Badge>
          )}
        </HStack>
      </HStack>

      {(post.title || post.content) && (
        <Text fontSize="sm" fontWeight="bold" color="white" mb={2}>
          {post.title || post.content}
        </Text>
      )}

      {post.description && (
        <Text fontSize="sm" color="gray.300" mb={3}>
          {post.description}
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

      {post.category && (
        <HStack spacing={2} mb={3}>
          <Badge colorScheme="cyan" variant="subtle" fontSize="xs">
            {post.category}
          </Badge>
        </HStack>
      )}

      {showImage && post.image && (
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

      <HStack justify="space-between" pt={3} borderTop="1px solid" borderColor="whiteAlpha.200">
        <HStack spacing={4}>
          <HStack spacing={1}>
            <Icon as={Eye} boxSize={5} color="gray.400" />
            <Text fontSize="xs" color="gray.400">
              {views.toLocaleString()} views
            </Text>
          </HStack>
          <HStack spacing={1}>
            <Icon as={Zap} boxSize={5} color="purple.400" />
            <Text fontSize="xs" color="gray.400">
              {Math.round(engagement)}% engagement
            </Text>
          </HStack>
        </HStack>
        {post.url && (
          <Text
            as="a"
            href={post.url}
            target="_blank"
            rel="noopener noreferrer"
            fontSize="xs"
            color="purple.400"
            _hover={{ color: 'purple.300', textDecoration: 'underline' }}
          >
            Read more →
          </Text>
        )}
      </HStack>
    </Box>
  );
};

export default PostCard;
