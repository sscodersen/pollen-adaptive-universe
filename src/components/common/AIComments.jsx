import { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Button,
  Icon,
  Spinner,
  Avatar,
  Badge,
  Divider,
  useToast
} from '@chakra-ui/react';
import { MessageCircle, Sparkles, RefreshCw } from 'lucide-react';
import AdvancedReactions from './AdvancedReactions';

const AIComments = ({ post }) => {
  const [comments, setComments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const toast = useToast();

  const generateAIComments = async () => {
    setLoading(true);
    
    try {
      const postContext = `${post?.title || ''} ${post?.description || ''}`;
      const url = `/api/ai/generate-comments?context=${encodeURIComponent(postContext)}&count=3`;
      
      const eventSource = new EventSource(url);
      let generatedComments = [];

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'comment') {
          generatedComments.push({
            id: Date.now() + Math.random(),
            author: data.author || 'AI User',
            text: data.text,
            tone: data.tone || 'thoughtful',
            timestamp: new Date().toISOString()
          });
          setComments([...generatedComments]);
        } else if (data.type === 'complete') {
          eventSource.close();
          setLoading(false);
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        setLoading(false);
        toast({
          title: 'Failed to generate comments',
          status: 'error',
          duration: 3000
        });
      };
    } catch (error) {
      setLoading(false);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 3000
      });
    }
  };

  const handleToggleComments = () => {
    if (!showComments && comments.length === 0) {
      generateAIComments();
    }
    setShowComments(!showComments);
  };

  const getToneColor = (tone) => {
    const tones = {
      'thoughtful': 'purple',
      'enthusiastic': 'orange',
      'critical': 'blue',
      'supportive': 'green',
      'curious': 'yellow'
    };
    return tones[tone] || 'gray';
  };

  return (
    <Box>
      <Button
        leftIcon={<Icon as={MessageCircle} />}
        onClick={handleToggleComments}
        variant="ghost"
        size="sm"
        color="gray.400"
        _hover={{ color: 'purple.400' }}
      >
        {comments.length > 0 ? `${comments.length} Comments` : 'View AI Comments'}
      </Button>

      {showComments && (
        <VStack
          spacing={4}
          align="stretch"
          mt={4}
          pt={4}
          borderTop="1px solid"
          borderColor="whiteAlpha.200"
        >
          <HStack justify="space-between">
            <HStack spacing={2}>
              <Icon as={Sparkles} boxSize={4} color="purple.400" />
              <Text fontSize="sm" fontWeight="bold" color="white">
                AI-Generated Comments
              </Text>
            </HStack>
            {comments.length > 0 && (
              <Button
                leftIcon={<Icon as={RefreshCw} />}
                onClick={generateAIComments}
                size="xs"
                variant="ghost"
                colorScheme="purple"
                isLoading={loading}
              >
                Refresh
              </Button>
            )}
          </HStack>

          {loading && comments.length === 0 && (
            <HStack justify="center" py={4}>
              <Spinner size="sm" color="purple.500" />
              <Text fontSize="sm" color="gray.400">
                Generating thoughtful comments...
              </Text>
            </HStack>
          )}

          {comments.map((comment, index) => (
            <Box
              key={comment.id}
              p={3}
              bg="whiteAlpha.50"
              borderRadius="lg"
              border="1px solid"
              borderColor="whiteAlpha.100"
            >
              <HStack align="start" spacing={3} mb={2}>
                <Avatar
                  size="sm"
                  name={comment.author}
                  bg="purple.600"
                />
                <VStack align="start" spacing={1} flex={1}>
                  <HStack justify="space-between" w="full">
                    <HStack spacing={2}>
                      <Text fontSize="sm" fontWeight="bold" color="white">
                        {comment.author}
                      </Text>
                      <Badge
                        colorScheme={getToneColor(comment.tone)}
                        fontSize="xs"
                        px={2}
                        py={0.5}
                      >
                        {comment.tone}
                      </Badge>
                    </HStack>
                    <Text fontSize="xs" color="gray.500">
                      Just now
                    </Text>
                  </HStack>
                  <Text fontSize="sm" color="gray.300">
                    {comment.text}
                  </Text>
                  <HStack spacing={4} mt={2}>
                    <AdvancedReactions post={{ id: comment.id }} />
                  </HStack>
                </VStack>
              </HStack>
            </Box>
          ))}

          {comments.length > 0 && (
            <Box
              p={3}
              bg="purple.900"
              bgGradient="linear-gradient(135deg, rgba(103, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)"
              borderRadius="lg"
              border="1px solid"
              borderColor="purple.700"
            >
              <Text fontSize="xs" color="gray.300" textAlign="center">
                <Icon as={Sparkles} display="inline" boxSize={3} mr={1} />
                These comments are AI-generated to demonstrate engagement
              </Text>
            </Box>
          )}
        </VStack>
      )}
    </Box>
  );
};

export default AIComments;
