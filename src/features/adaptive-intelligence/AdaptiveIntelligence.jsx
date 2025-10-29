import { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Text,
  VStack,
  HStack,
  Badge,
  Spinner,
  Button,
  Select,
  Progress,
  useToast,
  Icon,
  Grid
} from '@chakra-ui/react';
import { Cpu, Zap, CheckCircle } from 'lucide-react';
import PostCard from '@components/common/PostCard';

const AdaptiveIntelligence = () => {
  const [content, setContent] = useState([]);
  const [loading, setLoading] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [category, setCategory] = useState('all');
  const [minScore, setMinScore] = useState(50);
  const [knowledgeBase, setKnowledgeBase] = useState(null);
  const toast = useToast();

  const loadCuratedContent = async () => {
    setLoading(true);
    setContent([]);
    
    try {
      const categoryParam = category !== 'all' ? `&category=${category}` : '';
      const url = `/api/adaptive-intelligence/curated-feed?min_score=${minScore}&max_results=20${categoryParam}`;
      
      const eventSource = new EventSource(url);
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'status') {
            toast({
              title: data.message,
              status: 'info',
              duration: 2000,
              position: 'top'
            });
          } else if (data.type === 'content') {
            setContent(prev => [...prev, data.data]);
          } else if (data.type === 'complete') {
            toast({
              title: data.message,
              status: 'success',
              duration: 3000,
              position: 'top'
            });
            eventSource.close();
            setLoading(false);
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      };
      
      eventSource.onerror = () => {
        eventSource.close();
        setLoading(false);
      };
      
    } catch (error) {
      toast({
        title: 'Error loading content',
        description: error.message,
        status: 'error',
        duration: 5000
      });
      setLoading(false);
    }
  };

  const trainPollenAI = async () => {
    setTrainingProgress({ progress: 0, message: 'Starting training...' });
    
    try {
      const url = `/api/adaptive-intelligence/train?min_score=70&max_items=50`;
      const eventSource = new EventSource(url);
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'training_progress') {
            setTrainingProgress({
              progress: data.progress,
              message: `Training ${data.current}/${data.total} - ${data.category}`
            });
          } else if (data.type === 'training_complete') {
            setTrainingProgress({
              progress: 100,
              message: 'Training complete!'
            });
            toast({
              title: '🎉 Training Complete!',
              description: `Knowledge base updated with ${data.session.examples_processed} examples`,
              status: 'success',
              duration: 5000
            });
            eventSource.close();
            loadKnowledgeBase();
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      };
      
      eventSource.onerror = () => {
        eventSource.close();
        setTrainingProgress(null);
      };
      
    } catch (error) {
      toast({
        title: 'Training failed',
        description: error.message,
        status: 'error',
        duration: 5000
      });
      setTrainingProgress(null);
    }
  };

  const loadKnowledgeBase = async () => {
    try {
      const response = await fetch('/api/adaptive-intelligence/knowledge-base');
      const data = await response.json();
      setKnowledgeBase(data);
    } catch (error) {
      console.error('Error loading knowledge base:', error);
    }
  };

  useEffect(() => {
    loadCuratedContent();
    loadKnowledgeBase();
  }, []);

  return (
    <Box px={4} py={6}>
      <VStack spacing={6} align="stretch">
        <Box>
          <HStack justify="space-between" align="start" mb={4}>
            <Box>
              <HStack spacing={3} mb={2}>
                <Icon as={Cpu} boxSize={8} color="purple.500" />
                <Heading size="xl" color="white">
                  Adaptive Intelligence Worker Bee
                </Heading>
              </HStack>
              <Text color="gray.400">
                Curated content powered by multidimensional quality analysis for Pollen AI training
              </Text>
            </Box>
            <Button
              colorScheme="purple"
              onClick={trainPollenAI}
              isLoading={trainingProgress !== null}
              leftIcon={<Icon as={Zap} />}
            >
              Train Pollen AI
            </Button>
          </HStack>
        </Box>

        {trainingProgress && (
          <Box
            p={4}
            bg="black"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
          >
            <VStack spacing={3}>
              <HStack w="full" justify="space-between">
                <Text fontWeight="bold" color="white">Training Pollen AI</Text>
                <Text fontSize="sm" color="gray.400">
                  {Math.round(trainingProgress.progress)}%
                </Text>
              </HStack>
              <Progress
                value={trainingProgress.progress}
                w="full"
                colorScheme="purple"
                size="lg"
                borderRadius="full"
              />
              <Text fontSize="sm" color="gray.400">
                {trainingProgress.message}
              </Text>
            </VStack>
          </Box>
        )}

        {knowledgeBase && (
          <Grid templateColumns="repeat(3, 1fr)" gap={4}>
            <Box
              p={4}
              bg="black"
              borderRadius="xl"
              border="1px solid"
              borderColor="whiteAlpha.200"
            >
              <Text fontSize="sm" color="gray.400">Categories</Text>
              <Text fontSize="2xl" fontWeight="bold" color="white">{knowledgeBase.total_categories}</Text>
              <Text fontSize="xs" color="gray.500">Knowledge areas</Text>
            </Box>
            <Box
              p={4}
              bg="black"
              borderRadius="xl"
              border="1px solid"
              borderColor="whiteAlpha.200"
            >
              <Text fontSize="sm" color="gray.400">Training Sessions</Text>
              <Text fontSize="2xl" fontWeight="bold" color="white">{knowledgeBase.training_sessions}</Text>
              <Text fontSize="xs" color="gray.500">Completed</Text>
            </Box>
            <Box
              p={4}
              bg="black"
              borderRadius="xl"
              border="1px solid"
              borderColor="whiteAlpha.200"
            >
              <HStack>
                <Icon as={CheckCircle} color="green.500" boxSize={6} />
                <Box>
                  <Text fontSize="sm" color="gray.400">Status</Text>
                  <Text fontSize="xl" fontWeight="bold" color="white">Active</Text>
                  <Text fontSize="xs" color="gray.500">Operational</Text>
                </Box>
              </HStack>
            </Box>
          </Grid>
        )}

        <HStack spacing={4}>
          <Select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            bg="whiteAlpha.100"
            color="white"
            borderColor="whiteAlpha.300"
            maxW="200px"
          >
            <option value="all" style={{ background: '#1a202c' }}>All Categories</option>
            <option value="technology" style={{ background: '#1a202c' }}>Technology</option>
            <option value="business" style={{ background: '#1a202c' }}>Business</option>
            <option value="science" style={{ background: '#1a202c' }}>Science</option>
            <option value="general" style={{ background: '#1a202c' }}>General</option>
          </Select>
          <Select
            value={minScore}
            onChange={(e) => setMinScore(Number(e.target.value))}
            bg="whiteAlpha.100"
            color="white"
            borderColor="whiteAlpha.300"
            maxW="200px"
          >
            <option value="30" style={{ background: '#1a202c' }}>Low Quality (30+)</option>
            <option value="50" style={{ background: '#1a202c' }}>Medium Quality (50+)</option>
            <option value="70" style={{ background: '#1a202c' }}>High Quality (70+)</option>
          </Select>
          <Button
            onClick={loadCuratedContent}
            colorScheme="blue"
            isLoading={loading}
          >
            Refresh Content
          </Button>
        </HStack>

        {loading && content.length === 0 && (
          <Box textAlign="center" py={12}>
            <Spinner size="xl" color="purple.500" mb={4} />
            <Text color="gray.400">Analyzing content sources...</Text>
          </Box>
        )}

        <VStack spacing={4} align="stretch">
          {content.map((item, index) => (
            <PostCard key={index} post={item} showImage={false} />
          ))}
        </VStack>

        {!loading && content.length === 0 && (
          <Box
            w="100%"
            p={8}
            textAlign="center"
            bg="whiteAlpha.50"
            borderRadius="xl"
            border="1px dashed"
            borderColor="whiteAlpha.300"
          >
            <Icon as={Cpu} boxSize={12} color="gray.400" mb={4} />
            <Text color="gray.500" fontSize="lg">
              No content found matching your criteria
            </Text>
            <Text color="gray.600" fontSize="sm">
              Try adjusting the filters or minimum quality score
            </Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default AdaptiveIntelligence;
