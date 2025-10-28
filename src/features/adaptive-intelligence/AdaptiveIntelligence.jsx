import { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  HStack,
  Badge,
  Card,
  CardBody,
  CardHeader,
  Spinner,
  Button,
  Select,
  Progress,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Grid,
  GridItem,
  useToast,
  Link,
  Icon,
  Tooltip
} from '@chakra-ui/react';
import { TrendingUp, Award, Target, Zap, CheckCircle, AlertCircle, Cpu } from 'lucide-react';

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

  const getQualityColor = (tier) => {
    switch (tier) {
      case 'HIGH QUALITY':
        return 'green';
      case 'MEDIUM QUALITY':
        return 'blue';
      default:
        return 'gray';
    }
  };

  const ScoreBar = ({ label, value, color }) => (
    <Box mb={2}>
      <HStack justify="space-between" mb={1}>
        <Text fontSize="xs" fontWeight="medium">{label}</Text>
        <Text fontSize="xs" fontWeight="bold">{value}</Text>
      </HStack>
      <Progress value={value} colorScheme={color} size="sm" borderRadius="full" />
    </Box>
  );

  return (
    <Container maxW="container.xl" py={8}>
      <VStack spacing={6} align="stretch">
        {/* Header */}
        <Box>
          <HStack justify="space-between" align="start">
            <Box>
              <HStack spacing={3} mb={2}>
                <Icon as={Cpu} boxSize={8} color="purple.500" />
                <Heading size="xl">
                  Adaptive Intelligence Worker Bee
                </Heading>
              </HStack>
              <Text color="gray.600">
                Curated content powered by multidimensional quality analysis for Pollen AI training
              </Text>
            </Box>
            <VStack align="stretch">
              <Button
                colorScheme="purple"
                onClick={trainPollenAI}
                isLoading={trainingProgress !== null}
                leftIcon={<Icon as={Zap} />}
              >
                Train Pollen AI
              </Button>
            </VStack>
          </HStack>
        </Box>

        {/* Training Progress */}
        {trainingProgress && (
          <Card>
            <CardBody>
              <VStack spacing={3}>
                <HStack w="full" justify="space-between">
                  <Text fontWeight="bold">Training Pollen AI</Text>
                  <Text fontSize="sm" color="gray.600">
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
                <Text fontSize="sm" color="gray.600">
                  {trainingProgress.message}
                </Text>
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Knowledge Base Stats */}
        {knowledgeBase && (
          <Grid templateColumns="repeat(3, 1fr)" gap={4}>
            <Card>
              <CardBody>
                <Stat>
                  <StatLabel>Categories</StatLabel>
                  <StatNumber>{knowledgeBase.total_categories}</StatNumber>
                  <StatHelpText>Knowledge areas</StatHelpText>
                </Stat>
              </CardBody>
            </Card>
            <Card>
              <CardBody>
                <Stat>
                  <StatLabel>Training Sessions</StatLabel>
                  <StatNumber>{knowledgeBase.training_sessions}</StatNumber>
                  <StatHelpText>Completed</StatHelpText>
                </Stat>
              </CardBody>
            </Card>
            <Card>
              <CardBody>
                <Stat>
                  <StatLabel>Worker Bee Status</StatLabel>
                  <StatNumber>
                    <HStack>
                      <Icon as={CheckCircle} color="green.500" />
                      <Text>Active</Text>
                    </HStack>
                  </StatNumber>
                  <StatHelpText>System operational</StatHelpText>
                </Stat>
              </CardBody>
            </Card>
          </Grid>
        )}

        {/* Filters */}
        <HStack spacing={4}>
          <Select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            maxW="200px"
          >
            <option value="all">All Categories</option>
            <option value="technology">Technology</option>
            <option value="business">Business</option>
            <option value="science">Science</option>
            <option value="general">General</option>
          </Select>
          <Select
            value={minScore}
            onChange={(e) => setMinScore(Number(e.target.value))}
            maxW="200px"
          >
            <option value="30">Low Quality (30+)</option>
            <option value="50">Medium Quality (50+)</option>
            <option value="70">High Quality (70+)</option>
          </Select>
          <Button
            onClick={loadCuratedContent}
            colorScheme="blue"
            isLoading={loading}
          >
            Refresh Content
          </Button>
        </HStack>

        {/* Loading State */}
        {loading && content.length === 0 && (
          <Box textAlign="center" py={12}>
            <Spinner size="xl" color="purple.500" mb={4} />
            <Text color="gray.600">Analyzing content sources...</Text>
          </Box>
        )}

        {/* Content Grid */}
        <VStack spacing={4} align="stretch">
          {content.map((item, index) => {
            const score = item.adaptive_score || {};
            return (
              <Card key={index} variant="outline" _hover={{ shadow: 'md', borderColor: 'purple.300' }}>
                <CardHeader pb={2}>
                  <HStack justify="space-between" align="start">
                    <Box flex={1}>
                      <Link
                        href={item.url}
                        isExternal
                        _hover={{ textDecoration: 'none' }}
                      >
                        <Heading size="md" mb={2} _hover={{ color: 'purple.600' }}>
                          {item.title}
                        </Heading>
                      </Link>
                      <HStack spacing={2} mb={2}>
                        <Badge colorScheme={getQualityColor(score.quality_tier)}>
                          {score.quality_tier}
                        </Badge>
                        <Badge colorScheme="purple">
                          Score: {score.overall}
                        </Badge>
                        <Badge colorScheme="gray">{item.source}</Badge>
                        <Badge colorScheme="cyan">{item.category}</Badge>
                      </HStack>
                    </Box>
                    <Tooltip label="Overall Quality Score">
                      <Box
                        w="60px"
                        h="60px"
                        borderRadius="full"
                        bg={`${getQualityColor(score.quality_tier)}.100`}
                        border="4px solid"
                        borderColor={`${getQualityColor(score.quality_tier)}.400`}
                        display="flex"
                        alignItems="center"
                        justifyContent="center"
                      >
                        <Text fontWeight="bold" fontSize="lg" color={`${getQualityColor(score.quality_tier)}.700`}>
                          {Math.round(score.overall)}
                        </Text>
                      </Box>
                    </Tooltip>
                  </HStack>
                </CardHeader>
                <CardBody pt={0}>
                  <Text color="gray.600" mb={4}>
                    {item.description}
                  </Text>

                  {/* Score Breakdown */}
                  <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                    <Box>
                      <ScoreBar label="📊 Scope" value={score.scope} color="blue" />
                      <ScoreBar label="⚡ Intensity" value={score.intensity} color="orange" />
                      <ScoreBar label="✨ Originality" value={score.originality} color="purple" />
                      <ScoreBar label="⏱️ Immediacy" value={score.immediacy} color="cyan" />
                    </Box>
                    <Box>
                      <ScoreBar label="🎯 Practicability" value={score.practicability} color="teal" />
                      <ScoreBar label="😊 Positivity" value={score.positivity} color="pink" />
                      <ScoreBar label="🛡️ Credibility" value={score.credibility} color="green" />
                    </Box>
                  </Grid>
                </CardBody>
              </Card>
            );
          })}
        </VStack>

        {/* Empty State */}
        {!loading && content.length === 0 && (
          <Box textAlign="center" py={12}>
            <Icon as={AlertCircle} boxSize={12} color="gray.400" mb={4} />
            <Text color="gray.600" fontSize="lg">
              No content found matching your criteria
            </Text>
            <Text color="gray.500" fontSize="sm">
              Try adjusting the filters or minimum quality score
            </Text>
          </Box>
        )}
      </VStack>
    </Container>
  );
};

export default AdaptiveIntelligence;
