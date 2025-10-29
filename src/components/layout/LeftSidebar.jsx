import { Box, VStack, HStack, Text, Avatar, Icon, Badge, Divider } from '@chakra-ui/react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Home, 
  Compass, 
  Newspaper,
  Calendar,
  Package,
  Bell, 
  MessageCircle, 
  User, 
  Settings,
  Sparkles,
  ShoppingBag,
  Plane,
  Heart,
  GraduationCap,
  TrendingUp,
  Code
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
    { id: 'explore', label: 'Explore', icon: Compass, path: '/explore', badge: null },
    { id: 'news', label: 'News', icon: Newspaper, path: '/news', badge: null },
    { id: 'events', label: 'Events', icon: Calendar, path: '/events', badge: null },
    { id: 'products', label: 'Products', icon: Package, path: '/products', badge: null },
    { id: 'activity', label: 'Activity', icon: Bell, path: '/activity', badge: 3 },
    { id: 'messages', label: 'Messages', icon: MessageCircle, path: '/messages', badge: 8 },
    { id: 'profile', label: 'Profile', icon: User, path: '/profile', badge: null },
  ];

  const quickAccess = [
    { id: 'shopping', label: 'Shopping', icon: ShoppingBag, path: '/app/shopping', gradient: 'linear(to-br, purple.400, purple.600)' },
    { id: 'travel', label: 'Travel', icon: Plane, path: '/app/travel', gradient: 'linear(to-br, cyan.400, blue.500)' },
    { id: 'health', label: 'Wellness', icon: Heart, path: '/app/health', gradient: 'linear(to-br, pink.400, red.400)' },
    { id: 'education', label: 'Learn', icon: GraduationCap, path: '/app/education', gradient: 'linear(to-br, orange.400, yellow.500)' },
    { id: 'finance', label: 'Finance', icon: TrendingUp, path: '/app/finance', gradient: 'linear(to-br, green.400, teal.500)' },
    { id: 'code', label: 'Code', icon: Code, path: '/app/code', gradient: 'linear(to-br, red.400, pink.500)' },
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

      <Box
        p={3}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
      >
        <HStack spacing={2} mb={3}>
          <Icon as={Sparkles} color="purple.400" boxSize={4} />
          <Text fontSize="xs" fontWeight="bold" color="white" textTransform="uppercase">
            Quick Access
          </Text>
        </HStack>
        <VStack spacing={2} align="stretch">
          {quickAccess.map((item) => (
            <HStack
              key={item.id}
              spacing={3}
              p={2}
              borderRadius="lg"
              cursor="pointer"
              _hover={{ bg: 'whiteAlpha.50' }}
              transition="all 0.2s"
              onClick={() => navigate(item.path)}
            >
              <Box
                p={2}
                borderRadius="md"
                bgGradient={item.gradient}
              >
                <Icon as={item.icon} boxSize={4} color="white" />
              </Box>
              <Text fontSize="xs" color="gray.300">
                {item.label}
              </Text>
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
