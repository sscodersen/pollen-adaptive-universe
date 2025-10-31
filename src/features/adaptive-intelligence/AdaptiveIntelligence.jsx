import { useState, useEffect, useRef } from 'react';
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
  Grid,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel
} from '@chakra-ui/react';
import { Cpu, Zap, CheckCircle, Newspaper, Calendar, Package, TrendingUp, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';
import PostCard from '@components/common/PostCard';
import { API_BASE_URL } from '@utils/constants';

const AdaptiveIntelligence = () => {
  const [content, setContent] = useState([]);
  const [news, setNews] = useState([]);
  const [events, setEvents] = useState([]);
  const [products, setProducts] = useState([]);
  const [trendingTopics, setTrendingTopics] = useState([]);
  
  const [loading, setLoading] = useState(false);
  const [newsLoading, setNewsLoading] = useState(false);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [productsLoading, setProductsLoading] = useState(false);
  const [trendsLoading, setTrendsLoading] = useState(false);
  
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [category, setCategory] = useState('all');
  const [minScore, setMinScore] = useState(50);
  const [knowledgeBase, setKnowledgeBase] = useState(null);
  
  const [newsCategories, setNewsCategories] = useState([]);
  const [selectedNewsCategory, setSelectedNewsCategory] = useState('');
  const [newsStatus, setNewsStatus] = useState('');
  
  const [eventsCategories, setEventsCategories] = useState([]);
  const [selectedEventsCategory, setSelectedEventsCategory] = useState('');
  const [eventsStatus, setEventsStatus] = useState('');
  
  const [productsCategories, setProductsCategories] = useState([]);
  const [selectedProductsCategory, setSelectedProductsCategory] = useState('');
  const [productsStatus, setProductsStatus] = useState('');
  
  const eventSourceRef = useRef(null);
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

  const loadNews = () => {
    setNewsLoading(true);
    setNews([]);
    setNewsStatus('Fetching latest news...');

    const params = new URLSearchParams();
    if (selectedNewsCategory) params.append('category', selectedNewsCategory.toLowerCase());
    params.append('min_score', '0');
    params.append('max_results', '20');

    const eventSource = new EventSource(`${API_BASE_URL}/api/news/fetch?${params}`);
    
    const timeout = setTimeout(() => {
      eventSource.close();
      setNewsLoading(false);
      setNewsStatus('');
      if (news.length === 0) {
        toast({
          title: 'Loading timeout',
          description: 'News feed took too long to load',
          status: 'warning',
          duration: 3000,
        });
      }
    }, 15000);

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        
        if (parsed.type === 'status') {
          setNewsStatus(parsed.message);
        } else if (parsed.type === 'content') {
          setNews(prev => [...prev, parsed.data]);
        } else if (parsed.type === 'complete') {
          clearTimeout(timeout);
          setNewsStatus('');
          setNewsLoading(false);
          eventSource.close();
        } else if (parsed.type === 'error') {
          clearTimeout(timeout);
          setNewsStatus('');
          setNewsLoading(false);
          eventSource.close();
        }
      } catch (e) {}
    };

    eventSource.onerror = () => {
      clearTimeout(timeout);
      eventSource.close();
      setNewsLoading(false);
      setNewsStatus('');
    };
  };

  const loadEvents = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setEventsLoading(true);
    setEvents([]);
    setEventsStatus('Loading events...');

    const params = new URLSearchParams();
    if (selectedEventsCategory) params.append('category', selectedEventsCategory);
    params.append('max_results', '20');

    const eventSource = new EventSource(`${API_BASE_URL}/api/events/upcoming?${params}`);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        
        if (parsed.type === 'status') {
          setEventsStatus(parsed.message);
        } else if (parsed.type === 'event') {
          setEvents(prev => [...prev, parsed.data]);
        } else if (parsed.type === 'complete') {
          setEventsStatus('');
          setEventsLoading(false);
          eventSource.close();
          eventSourceRef.current = null;
        } else if (parsed.type === 'error') {
          setEventsStatus('');
          setEventsLoading(false);
          eventSource.close();
          eventSourceRef.current = null;
        }
      } catch (e) {}
    };

    eventSource.onerror = () => {
      eventSource.close();
      eventSourceRef.current = null;
      setEventsLoading(false);
      setEventsStatus('');
    };
  };

  const loadProducts = () => {
    setProductsLoading(true);
    setProducts([]);
    setProductsStatus('Discovering products...');

    const params = new URLSearchParams();
    if (selectedProductsCategory) params.append('category', selectedProductsCategory);
    params.append('min_score', '20');
    params.append('max_results', '20');

    const eventSource = new EventSource(`${API_BASE_URL}/api/products/discover?${params}`);

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        
        if (parsed.type === 'status') {
          setProductsStatus(parsed.message);
        } else if (parsed.type === 'product') {
          setProducts(prev => [...prev, parsed.data]);
        } else if (parsed.type === 'complete') {
          setProductsStatus('');
          setProductsLoading(false);
        } else if (parsed.type === 'error') {
          setProductsStatus('');
          setProductsLoading(false);
        }
      } catch (e) {}
    };

    eventSource.onerror = () => {
      eventSource.close();
      setProductsLoading(false);
      setProductsStatus('');
    };
  };

  const loadTrendingTopics = async () => {
    setTrendsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/feed/trending`);
      const data = await response.json();
      setTrendingTopics(data.trends || []);
    } catch (error) {
      console.error('Error loading trending topics:', error);
    } finally {
      setTrendsLoading(false);
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

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        year: 'numeric' 
      });
    } catch {
      return dateString;
    }
  };

  useEffect(() => {
    loadCuratedContent();
    loadKnowledgeBase();
    loadTrendingTopics();
    
    fetch(`${API_BASE_URL}/api/news/categories`)
      .then(res => res.json())
      .then(data => setNewsCategories(data.categories || []))
      .catch(() => {});
    
    fetch(`${API_BASE_URL}/api/events/categories`)
      .then(res => res.json())
      .then(data => setEventsCategories(data.categories || []))
      .catch(() => {});
    
    fetch(`${API_BASE_URL}/api/products/categories`)
      .then(res => res.json())
      .then(data => setProductsCategories(data.categories || []))
      .catch(() => {});
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
                  AI Worker Bee
                </Heading>
              </HStack>
              <Text color="gray.400">
                Curated content powered by multidimensional quality analysis
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

        <Tabs colorScheme="purple" variant="soft-rounded">
          <TabList overflowX="auto" overflowY="hidden" flexWrap="nowrap" pb={2}>
            <Tab color="gray.400" _selected={{ color: 'white', bg: 'purple.500' }}>
              <Icon as={Cpu} mr={2} />
              Curated Content
            </Tab>
            <Tab color="gray.400" _selected={{ color: 'white', bg: 'pink.500' }}>
              <Icon as={Newspaper} mr={2} />
              News
            </Tab>
            <Tab color="gray.400" _selected={{ color: 'white', bg: 'blue.500' }}>
              <Icon as={Calendar} mr={2} />
              Events
            </Tab>
            <Tab color="gray.400" _selected={{ color: 'white', bg: 'green.500' }}>
              <Icon as={Package} mr={2} />
              Products
            </Tab>
            <Tab color="gray.400" _selected={{ color: 'white', bg: 'orange.500' }}>
              <Icon as={TrendingUp} mr={2} />
              Trending
            </Tab>
          </TabList>

          <TabPanels>
            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
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
                    leftIcon={<RefreshCw size={16} />}
                  >
                    Refresh
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
            </TabPanel>

            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                <HStack w="100%" spacing={3}>
                  <Select
                    value={selectedNewsCategory}
                    onChange={(e) => setSelectedNewsCategory(e.target.value)}
                    bg="whiteAlpha.100"
                    border="1px solid"
                    borderColor="whiteAlpha.300"
                    color="white"
                    borderRadius="xl"
                  >
                    <option value="" style={{ background: '#1a202c' }}>All Categories</option>
                    {newsCategories.map(cat => (
                      <option key={cat} value={cat} style={{ background: '#1a202c' }}>
                        {cat}
                      </option>
                    ))}
                  </Select>
                  <Button
                    onClick={loadNews}
                    isLoading={newsLoading}
                    colorScheme="pink"
                    leftIcon={<RefreshCw size={16} />}
                  >
                    Refresh
                  </Button>
                </HStack>

                {newsStatus && (
                  <HStack w="100%" justify="center" p={4}>
                    <Spinner size="sm" color="pink.400" />
                    <Text fontSize="sm" color="gray.400">{newsStatus}</Text>
                  </HStack>
                )}

                {news.length > 0 && (
                  <VStack spacing={4} w="100%">
                    {news.map((article, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        transition={{
                          duration: 0.4,
                          delay: idx * 0.1,
                          ease: [0.4, 0, 0.2, 1]
                        }}
                        style={{ width: '100%' }}
                      >
                        <PostCard post={article} showImage={true} />
                      </motion.div>
                    ))}
                  </VStack>
                )}

                {!newsLoading && news.length === 0 && (
                  <Box
                    w="100%"
                    p={8}
                    textAlign="center"
                    bg="whiteAlpha.50"
                    borderRadius="xl"
                    border="1px dashed"
                    borderColor="whiteAlpha.300"
                  >
                    <Newspaper size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
                    <Text color="gray.500">No news found. Click Refresh to load articles.</Text>
                  </Box>
                )}
              </VStack>
            </TabPanel>

            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                <HStack w="100%" spacing={3}>
                  <Select
                    value={selectedEventsCategory}
                    onChange={(e) => setSelectedEventsCategory(e.target.value)}
                    bg="whiteAlpha.100"
                    border="1px solid"
                    borderColor="whiteAlpha.300"
                    color="white"
                    borderRadius="xl"
                  >
                    <option value="" style={{ background: '#1a202c' }}>All Categories</option>
                    {eventsCategories.map(cat => (
                      <option key={cat} value={cat.toLowerCase()} style={{ background: '#1a202c' }}>
                        {cat}
                      </option>
                    ))}
                  </Select>
                  <Button
                    onClick={loadEvents}
                    isLoading={eventsLoading}
                    colorScheme="blue"
                    leftIcon={<RefreshCw size={16} />}
                  >
                    Refresh
                  </Button>
                </HStack>

                {eventsStatus && (
                  <HStack w="100%" justify="center" p={4}>
                    <Spinner size="sm" color="blue.400" />
                    <Text fontSize="sm" color="gray.400">{eventsStatus}</Text>
                  </HStack>
                )}

                {events.length > 0 && (
                  <VStack spacing={4} w="100%">
                    {events.map((event, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        transition={{
                          duration: 0.4,
                          delay: idx * 0.1,
                          ease: [0.4, 0, 0.2, 1]
                        }}
                        style={{ width: '100%' }}
                      >
                        <PostCard 
                          post={{
                            ...event,
                            content: event.title,
                            tags: [
                              event.category,
                              event.date && `📅 ${formatDate(event.date)}`,
                              event.location && `📍 ${event.location}`
                            ].filter(Boolean)
                          }} 
                          showImage={true} 
                        />
                      </motion.div>
                    ))}
                  </VStack>
                )}

                {!eventsLoading && events.length === 0 && (
                  <Box
                    w="100%"
                    p={8}
                    textAlign="center"
                    bg="whiteAlpha.50"
                    borderRadius="xl"
                    border="1px dashed"
                    borderColor="whiteAlpha.300"
                  >
                    <Calendar size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
                    <Text color="gray.500">No events found. Click Refresh to discover events.</Text>
                  </Box>
                )}
              </VStack>
            </TabPanel>

            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                <HStack w="100%" spacing={3}>
                  <Select
                    value={selectedProductsCategory}
                    onChange={(e) => setSelectedProductsCategory(e.target.value)}
                    bg="whiteAlpha.100"
                    border="1px solid"
                    borderColor="whiteAlpha.300"
                    color="white"
                    borderRadius="xl"
                  >
                    <option value="" style={{ background: '#1a202c' }}>All Categories</option>
                    {productsCategories.map(cat => (
                      <option key={cat} value={cat} style={{ background: '#1a202c' }}>
                        {cat}
                      </option>
                    ))}
                  </Select>
                  <Button
                    onClick={loadProducts}
                    isLoading={productsLoading}
                    colorScheme="green"
                    leftIcon={<RefreshCw size={16} />}
                  >
                    Refresh
                  </Button>
                </HStack>

                {productsStatus && (
                  <HStack w="100%" justify="center" p={4}>
                    <Spinner size="sm" color="green.400" />
                    <Text fontSize="sm" color="gray.400">{productsStatus}</Text>
                  </HStack>
                )}

                {products.length > 0 && (
                  <VStack spacing={4} w="100%">
                    {products.map((product, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        transition={{
                          duration: 0.4,
                          delay: idx * 0.1,
                          ease: [0.4, 0, 0.2, 1]
                        }}
                        style={{ width: '100%' }}
                      >
                        <PostCard 
                          post={{
                            ...product,
                            content: product.title,
                            tags: product.price ? [`💰 ${product.price}`] : undefined
                          }} 
                          showImage={false} 
                        />
                      </motion.div>
                    ))}
                  </VStack>
                )}

                {!productsLoading && products.length === 0 && (
                  <Box
                    w="100%"
                    p={8}
                    textAlign="center"
                    bg="whiteAlpha.50"
                    borderRadius="xl"
                    border="1px dashed"
                    borderColor="whiteAlpha.300"
                  >
                    <Package size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
                    <Text color="gray.500">No products found. Click Refresh to discover products.</Text>
                  </Box>
                )}
              </VStack>
            </TabPanel>

            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                <HStack justify="space-between">
                  <Text fontSize="lg" fontWeight="bold" color="white">
                    Trending Topics
                  </Text>
                  <Button
                    onClick={loadTrendingTopics}
                    isLoading={trendsLoading}
                    colorScheme="orange"
                    size="sm"
                    leftIcon={<RefreshCw size={16} />}
                  >
                    Refresh
                  </Button>
                </HStack>

                {trendsLoading && (
                  <Box textAlign="center" py={8}>
                    <Spinner size="lg" color="orange.500" mb={4} />
                    <Text color="gray.400">Loading trending topics...</Text>
                  </Box>
                )}

                {!trendsLoading && trendingTopics.length > 0 && (
                  <Grid templateColumns="repeat(auto-fill, minmax(250px, 1fr))" gap={4}>
                    {trendingTopics.map((topic, idx) => (
                      <motion.div
                        key={topic.id}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: idx * 0.1 }}
                      >
                        <Box
                          p={4}
                          bg="whiteAlpha.100"
                          borderRadius="xl"
                          border="1px solid"
                          borderColor="whiteAlpha.200"
                          _hover={{ bg: 'whiteAlpha.150', borderColor: 'orange.400' }}
                          cursor="pointer"
                          transition="all 0.2s"
                        >
                          <VStack align="start" spacing={2}>
                            <Text fontSize="lg" fontWeight="bold" color="white">
                              {topic.tag}
                            </Text>
                            <HStack>
                              <Badge colorScheme="orange">{topic.trend}</Badge>
                              <Text fontSize="sm" color="gray.400">
                                {topic.posts} posts
                              </Text>
                            </HStack>
                          </VStack>
                        </Box>
                      </motion.div>
                    ))}
                  </Grid>
                )}

                {!trendsLoading && trendingTopics.length === 0 && (
                  <Box
                    w="100%"
                    p={8}
                    textAlign="center"
                    bg="whiteAlpha.50"
                    borderRadius="xl"
                    border="1px dashed"
                    borderColor="whiteAlpha.300"
                  >
                    <TrendingUp size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
                    <Text color="gray.500">No trending topics available. Click Refresh to load.</Text>
                  </Box>
                )}
              </VStack>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
    </Box>
  );
};

export default AdaptiveIntelligence;
