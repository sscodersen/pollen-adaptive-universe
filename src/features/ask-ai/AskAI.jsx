import { useState, useRef, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Textarea,
  Button,
  Text,
  Icon,
  Grid,
  Badge,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Image,
  useToast,
  Collapse,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from '@chakra-ui/react';
import {
  MessageCircle,
  Code,
  ShoppingBag,
  Plane,
  Heart,
  GraduationCap,
  TrendingUp,
  Image as ImageIcon,
  Video,
  Mic,
  Music,
  Brain,
  Zap,
  Send,
  Sparkles,
  ChevronDown,
  Trash2,
  Clock,
} from 'lucide-react';
import useAskAIStore from '@store/askAIStore';
import { format } from 'date-fns';

const iconMap = {
  MessageCircle,
  Code,
  ShoppingBag,
  Plane,
  Heart,
  GraduationCap,
  TrendingUp,
  Image: ImageIcon,
  Video,
  Mic,
  Music,
  Brain,
  Zap,
};

const AskAI = () => {
  const toast = useToast();
  const [input, setInput] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const messagesEndRef = useRef(null);
  
  const {
    currentMode,
    modes,
    sessions,
    currentSession,
    isStreaming,
    streamingData,
    setMode,
    createSession,
    addMessage,
    setStreaming,
    appendStreamingData,
    clearStreamingData,
    clearSession,
  } = useAskAIStore();
  
  const currentModeData = modes.find((m) => m.id === currentMode);
  const currentSessionData = sessions.find((s) => s.id === currentSession);
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSessionData?.messages, streamingData]);
  
  const handleSubmit = async () => {
    if (!input.trim() || isStreaming) return;
    
    let sessionId = currentSession;
    if (!sessionId) {
      sessionId = createSession(currentMode);
    }
    
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };
    
    addMessage(sessionId, userMessage);
    setInput('');
    setStreaming(true);
    clearStreamingData();
    
    try {
      const endpoint = getEndpointForMode(currentMode);
      const eventSource = new EventSource(
        `${endpoint}?${new URLSearchParams({ query: input, mode: currentMode })}`
      );
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.text) {
            appendStreamingData(data.text);
          } else if (data.type === 'complete') {
            const aiMessage = {
              role: 'assistant',
              content: useAskAIStore.getState().streamingData,
              timestamp: new Date().toISOString(),
              mode: currentMode,
            };
            addMessage(sessionId, aiMessage);
            clearStreamingData();
            setStreaming(false);
            eventSource.close();
          }
        } catch (e) {
          console.error('Error parsing SSE data:', e);
        }
      };
      
      eventSource.onerror = () => {
        setStreaming(false);
        eventSource.close();
        toast({
          title: 'Connection error',
          description: 'Failed to connect to AI service',
          status: 'error',
          duration: 3000,
        });
      };
    } catch (error) {
      setStreaming(false);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 3000,
      });
    }
  };
  
  const getEndpointForMode = (mode) => {
    const endpoints = {
      chat: '/api/playground/chat',
      code: '/api/playground/code-assist',
      shopping: '/api/shopping/search',
      travel: '/api/travel/plan',
      health: '/api/health/advice',
      education: '/api/education/learn',
      finance: '/api/finance/advice',
      image: '/api/playground/generate-image',
      video: '/api/playground/generate-video',
      audio: '/api/playground/voice-to-text',
      music: '/api/playground/generate-music',
      react: '/api/playground/react-mode',
      automation: '/api/playground/automate-task',
    };
    return endpoints[mode] || '/api/playground/chat';
  };
  
  return (
    <Box h="100vh" display="flex" flexDirection="column" p={4}>
      <VStack spacing={4} align="stretch" flex={1} overflow="hidden">
        <Box
          p={6}
          bg="black"
          borderRadius="2xl"
          border="1px solid"
          borderColor="whiteAlpha.200"
          bgGradient="linear(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)"
        >
          <HStack justify="space-between" align="center">
            <HStack spacing={4}>
              <Box
                p={3}
                borderRadius="lg"
                bgGradient="linear(to-br, purple.500, pink.500)"
              >
                <Icon as={iconMap[currentModeData?.icon]} boxSize={6} color="white" />
              </Box>
              <VStack align="start" spacing={0}>
                <HStack>
                  <Text fontSize="2xl" fontWeight="bold" color="white">
                    Ask AI
                  </Text>
                  <Badge colorScheme="purple" fontSize="xs">
                    {currentModeData?.name}
                  </Badge>
                </HStack>
                <Text fontSize="sm" color="gray.400">
                  {currentModeData?.description}
                </Text>
              </VStack>
            </HStack>
            
            <HStack spacing={2}>
              <IconButton
                icon={<Clock size={18} />}
                variant="ghost"
                colorScheme="purple"
                onClick={() => setShowHistory(!showHistory)}
                aria-label="Toggle history"
              />
              <Menu>
                <MenuButton
                  as={Button}
                  rightIcon={<ChevronDown size={16} />}
                  variant="outline"
                  colorScheme="purple"
                  size="sm"
                >
                  Switch Mode
                </MenuButton>
                <MenuList bg="gray.900" borderColor="whiteAlpha.200">
                  <Grid templateColumns="repeat(2, 1fr)" gap={1} p={2}>
                    {modes.map((mode) => (
                      <MenuItem
                        key={mode.id}
                        onClick={() => setMode(mode.id)}
                        bg={currentMode === mode.id ? 'purple.900' : 'transparent'}
                        _hover={{ bg: 'whiteAlpha.100' }}
                        borderRadius="md"
                      >
                        <HStack spacing={2}>
                          <Icon as={iconMap[mode.icon]} boxSize={4} />
                          <Text fontSize="sm">{mode.name}</Text>
                        </HStack>
                      </MenuItem>
                    ))}
                  </Grid>
                </MenuList>
              </Menu>
            </HStack>
          </HStack>
        </Box>
        
        <Box
          flex={1}
          bg="black"
          borderRadius="2xl"
          border="1px solid"
          borderColor="whiteAlpha.200"
          p={4}
          overflowY="auto"
          css={{
            '&::-webkit-scrollbar': {
              width: '6px',
            },
            '&::-webkit-scrollbar-track': {
              background: 'transparent',
            },
            '&::-webkit-scrollbar-thumb': {
              background: '#333',
              borderRadius: '3px',
            },
          }}
        >
          {currentSessionData?.messages.length === 0 && !streamingData ? (
            <VStack spacing={6} align="center" justify="center" h="100%" color="gray.500">
              <Icon as={Sparkles} boxSize={12} />
              <VStack spacing={2}>
                <Text fontSize="xl" fontWeight="bold">
                  Ready to assist you
                </Text>
                <Text fontSize="sm" textAlign="center" maxW="400px">
                  Ask me anything! I can help with coding, shopping, travel planning, learning, and much more.
                </Text>
              </VStack>
              <Grid templateColumns="repeat(3, 1fr)" gap={3} w="100%" maxW="600px" pt={4}>
                {modes.slice(0, 6).map((mode) => (
                  <Box
                    key={mode.id}
                    p={3}
                    bg="whiteAlpha.50"
                    borderRadius="lg"
                    cursor="pointer"
                    _hover={{ bg: 'whiteAlpha.100', transform: 'translateY(-2px)' }}
                    transition="all 0.2s"
                    onClick={() => setMode(mode.id)}
                  >
                    <VStack spacing={2}>
                      <Icon as={iconMap[mode.icon]} boxSize={6} color="purple.400" />
                      <Text fontSize="xs" fontWeight="bold" color="white">
                        {mode.name}
                      </Text>
                    </VStack>
                  </Box>
                ))}
              </Grid>
            </VStack>
          ) : (
            <VStack spacing={4} align="stretch">
              {currentSessionData?.messages.map((message, idx) => (
                <Box
                  key={idx}
                  alignSelf={message.role === 'user' ? 'flex-end' : 'flex-start'}
                  maxW="80%"
                >
                  <Box
                    p={4}
                    borderRadius="lg"
                    bg={message.role === 'user' ? 'purple.600' : 'whiteAlpha.100'}
                    color="white"
                  >
                    <Text fontSize="sm" whiteSpace="pre-wrap">
                      {message.content}
                    </Text>
                    <Text fontSize="xs" color="whiteAlpha.600" mt={2}>
                      {format(new Date(message.timestamp), 'HH:mm')}
                    </Text>
                  </Box>
                </Box>
              ))}
              
              {streamingData && (
                <Box alignSelf="flex-start" maxW="80%">
                  <Box p={4} borderRadius="lg" bg="whiteAlpha.100" color="white">
                    <HStack spacing={2} mb={2}>
                      <Icon as={Sparkles} boxSize={4} color="purple.400" />
                      <Text fontSize="xs" color="purple.400">
                        AI is typing...
                      </Text>
                    </HStack>
                    <Text fontSize="sm" whiteSpace="pre-wrap">
                      {streamingData}
                    </Text>
                  </Box>
                </Box>
              )}
              <div ref={messagesEndRef} />
            </VStack>
          )}
        </Box>
        
        <Box
          p={4}
          bg="black"
          borderRadius="2xl"
          border="1px solid"
          borderColor="whiteAlpha.200"
        >
          <HStack spacing={3}>
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={`Ask ${currentModeData?.name} AI anything...`}
              size="lg"
              minH="60px"
              maxH="200px"
              bg="whiteAlpha.100"
              border="none"
              color="white"
              resize="none"
              _placeholder={{ color: 'gray.500' }}
              _focus={{ bg: 'whiteAlpha.150' }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
            <Button
              colorScheme="purple"
              size="lg"
              h="60px"
              px={8}
              onClick={handleSubmit}
              isLoading={isStreaming}
              isDisabled={!input.trim()}
              leftIcon={<Send size={20} />}
            >
              Send
            </Button>
          </HStack>
        </Box>
      </VStack>
    </Box>
  );
};

export default AskAI;
