import { Box, VStack, HStack, Text, Avatar, Icon, Badge, Divider } from '@chakra-ui/react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Home, 
  Compass, 
  Bell, 
  MessageCircle, 
  User, 
  Settings,
  Sparkles,
  Zap,
  Award
} from 'lucide-react';

const LeftSidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const anonymousStats = {
    activeUsers: '2.4M',
    postsToday: '847K',
    qualityScore: 94
  };

  const menuItems = [
    { id: 'feed', label: 'Feed', icon: Home, path: '/', badge: null },
    { id: 'ask-ai', label: 'Ask AI', icon: Zap, path: '/ask-ai', badge: 'New' },
    { id: 'ai-worker-bee', label: 'AI Worker Bee', icon: Award, path: '/adaptive-intelligence', badge: null },
    { id: 'explore', label: 'Explore', icon: Compass, path: '/explore', badge: null },
    { id: 'activity', label: 'Activity', icon: Bell, path: '/activity', badge: 3 },
    { id: 'messages', label: 'Messages', icon: MessageCircle, path: '/messages', badge: 8 },
    { id: 'profile', label: 'Profile', icon: User, path: '/profile', badge: null },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <VStack spacing={4} align="stretch">
      <Box
        p={4}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
        bgGradient="linear(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)"
      >
        <VStack spacing={3} align="center">
          <Box
            p={3}
            borderRadius="lg"
            bgGradient="linear(to-br, purple.500, pink.500)"
          >
            <Icon as={Sparkles} boxSize={6} color="white" />
          </Box>
          <Text fontSize="md" fontWeight="bold" color="white" textAlign="center">
            100% Anonymous Platform
          </Text>
          <Text fontSize="xs" color="gray.400" textAlign="center">
            Privacy-first, AI-powered community
          </Text>
        </VStack>

        <HStack spacing={4} justify="space-around" pt={3} mt={3} borderTop="1px solid" borderColor="whiteAlpha.200">
          <VStack spacing={0}>
            <Text fontSize="sm" fontWeight="bold" color="purple.400">{anonymousStats.activeUsers}</Text>
            <Text fontSize="xs" color="gray.400">Active</Text>
          </VStack>
          <VStack spacing={0}>
            <Text fontSize="sm" fontWeight="bold" color="green.400">{anonymousStats.postsToday}</Text>
            <Text fontSize="xs" color="gray.400">Today</Text>
          </VStack>
          <VStack spacing={0}>
            <Text fontSize="sm" fontWeight="bold" color="blue.400">{anonymousStats.qualityScore}</Text>
            <Text fontSize="xs" color="gray.400">Quality</Text>
          </VStack>
        </HStack>
      </Box>

      <Box
        p={3}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
      >
        <VStack spacing={1} align="stretch">
          {menuItems.map((item) => (
            <HStack
              key={item.id}
              spacing={3}
              p={3}
              borderRadius="lg"
              cursor="pointer"
              bg={isActive(item.path) ? 'whiteAlpha.100' : 'transparent'}
              borderLeft={isActive(item.path) ? '3px solid' : '3px solid transparent'}
              borderColor={isActive(item.path) ? 'purple.500' : 'transparent'}
              _hover={{ bg: 'whiteAlpha.100' }}
              transition="all 0.2s"
              onClick={() => navigate(item.path)}
            >
              <Icon as={item.icon} boxSize={5} color={isActive(item.path) ? 'purple.400' : 'gray.400'} />
              <Text 
                fontSize="sm" 
                fontWeight={isActive(item.path) ? 'bold' : 'medium'} 
                color="white"
                flex={1}
              >
                {item.label}
              </Text>
              {item.badge && (
                <Badge colorScheme="purple" borderRadius="full" fontSize="xs">
                  {item.badge}
                </Badge>
              )}
            </HStack>
          ))}
        </VStack>
      </Box>

      <HStack
        spacing={3}
        p={3}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
        cursor="pointer"
        _hover={{ bg: 'whiteAlpha.50' }}
        transition="all 0.2s"
        onClick={() => navigate('/settings')}
      >
        <Icon as={Settings} boxSize={5} color="gray.400" />
        <Text fontSize="sm" color="white">Settings</Text>
      </HStack>
    </VStack>
  );
};

export default LeftSidebar;
