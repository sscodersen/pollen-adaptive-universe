import { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Avatar,
  Button,
  Icon,
  Switch,
  Divider,
  Badge,
  SimpleGrid,
  Progress
} from '@chakra-ui/react';
import {
  User,
  Settings,
  History,
  Bookmark,
  Download,
  Bell,
  Moon,
  Sun,
  Trash2,
  LogOut,
  Shield,
  Zap
} from 'lucide-react';

const Profile = () => {
  const [notifications, setNotifications] = useState(true);
  const [darkMode, setDarkMode] = useState(true);

  const userStats = [
    { label: 'AI Queries', value: 247, color: 'purple' },
    { label: 'Hours Saved', value: 18, color: 'cyan' },
    { label: 'Favorites', value: 34, color: 'pink' },
    { label: 'Streak Days', value: 12, color: 'orange' },
  ];

  const recentHistory = [
    { title: 'Best laptops under $1000', feature: 'Shopping', time: '2 hours ago' },
    { title: 'Tokyo travel itinerary', feature: 'Travel', time: 'Yesterday' },
    { title: 'Blog post about AI', feature: 'Content', time: '2 days ago' },
    { title: 'Meditation benefits', feature: 'Health', time: '3 days ago' },
  ];

  const savedQueries = [
    { title: 'Weekly tech news roundup', feature: 'News' },
    { title: 'Healthy meal prep ideas', feature: 'Health' },
    { title: 'Python tutorial resources', feature: 'Education' },
  ];

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={6}>
        {/* Profile Header */}
        <Box w="100%">
          <VStack spacing={4}>
            <Box
              w="100%"
              p={6}
              borderRadius="2xl"
              bgGradient="linear(135deg, purple.600, purple.800)"
              position="relative"
              overflow="hidden"
            >
              <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
                <User size={120} />
              </Box>
              <VStack spacing={3} position="relative">
                <Avatar size="xl" bg="purple.500" />
                <VStack spacing={0}>
                  <Text fontSize="2xl" fontWeight="bold" color="white">
                    Anonymous User
                  </Text>
                  <HStack spacing={2}>
                    <Icon as={Shield} boxSize={4} color="green.400" />
                    <Text fontSize="sm" color="whiteAlpha.800">
                      Privacy-First · No Sign-In Required
                    </Text>
                  </HStack>
                </VStack>
                <Badge colorScheme="green" fontSize="sm" px={3} py={1}>
                  <HStack spacing={1}>
                    <Icon as={Shield} boxSize={3} />
                    <Text>All Data Stored Locally</Text>
                  </HStack>
                </Badge>
              </VStack>
            </Box>

            {/* Stats Grid */}
            <SimpleGrid columns={2} spacing={3} w="100%">
              {userStats.map((stat) => (
                <Box
                  key={stat.label}
                  p={4}
                  bg="whiteAlpha.100"
                  backdropFilter="blur(10px)"
                  borderRadius="xl"
                  border="1px solid"
                  borderColor="whiteAlpha.200"
                >
                  <VStack align="start" spacing={1}>
                    <Text fontSize="2xl" fontWeight="bold" color={`${stat.color}.400`}>
                      {stat.value}
                    </Text>
                    <Text fontSize="xs" color="gray.400">
                      {stat.label}
                    </Text>
                  </VStack>
                </Box>
              ))}
            </SimpleGrid>
          </VStack>
        </Box>

        {/* Settings Section */}
        <Box w="100%">
          <HStack spacing={2} mb={4}>
            <Icon as={Settings} color="purple.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Settings
            </Text>
          </HStack>
          <VStack
            spacing={0}
            w="100%"
            bg="whiteAlpha.100"
            backdropFilter="blur(10px)"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
            overflow="hidden"
          >
            <HStack justify="space-between" w="100%" p={4}>
              <HStack spacing={3}>
                <Icon as={Bell} boxSize={5} color="cyan.400" />
                <Text fontSize="sm" color="white">
                  Notifications
                </Text>
              </HStack>
              <Switch
                isChecked={notifications}
                onChange={(e) => setNotifications(e.target.checked)}
                colorScheme="purple"
              />
            </HStack>

            <Divider borderColor="whiteAlpha.200" />

            <HStack justify="space-between" w="100%" p={4}>
              <HStack spacing={3}>
                <Icon as={darkMode ? Moon : Sun} boxSize={5} color="purple.400" />
                <Text fontSize="sm" color="white">
                  Dark Mode
                </Text>
              </HStack>
              <Switch
                isChecked={darkMode}
                onChange={(e) => setDarkMode(e.target.checked)}
                colorScheme="purple"
              />
            </HStack>

            <Divider borderColor="whiteAlpha.200" />

            <HStack justify="space-between" w="100%" p={4} cursor="pointer" _hover={{ bg: 'whiteAlpha.50' }}>
              <HStack spacing={3}>
                <Icon as={Shield} boxSize={5} color="green.400" />
                <Text fontSize="sm" color="white">
                  Privacy & Security
                </Text>
              </HStack>
              <Text fontSize="xs" color="gray.500">
                →
              </Text>
            </HStack>
          </VStack>
        </Box>

        {/* Recent History */}
        <Box w="100%">
          <HStack spacing={2} mb={4}>
            <Icon as={History} color="orange.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Recent Activity
            </Text>
          </HStack>
          <VStack spacing={3} w="100%">
            {recentHistory.map((item, index) => (
              <Box
                key={index}
                w="100%"
                p={4}
                bg="whiteAlpha.100"
                backdropFilter="blur(10px)"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
                cursor="pointer"
                _hover={{ bg: 'whiteAlpha.150' }}
              >
                <VStack align="start" spacing={1}>
                  <Text fontSize="sm" fontWeight="medium" color="white">
                    {item.title}
                  </Text>
                  <HStack spacing={2}>
                    <Badge colorScheme="purple" fontSize="xs">
                      {item.feature}
                    </Badge>
                    <Text fontSize="xs" color="gray.500">
                      {item.time}
                    </Text>
                  </HStack>
                </VStack>
              </Box>
            ))}
          </VStack>
          <Button
            variant="ghost"
            colorScheme="purple"
            size="sm"
            w="100%"
            mt={2}
          >
            View All History
          </Button>
        </Box>

        {/* Saved Queries */}
        <Box w="100%">
          <HStack spacing={2} mb={4}>
            <Icon as={Bookmark} color="pink.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Saved Queries
            </Text>
          </HStack>
          <VStack spacing={3} w="100%">
            {savedQueries.map((item, index) => (
              <Box
                key={index}
                w="100%"
                p={4}
                bg="whiteAlpha.100"
                backdropFilter="blur(10px)"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
                cursor="pointer"
                _hover={{ bg: 'whiteAlpha.150' }}
              >
                <HStack justify="space-between">
                  <VStack align="start" spacing={1}>
                    <Text fontSize="sm" fontWeight="medium" color="white">
                      {item.title}
                    </Text>
                    <Badge colorScheme="blue" fontSize="xs">
                      {item.feature}
                    </Badge>
                  </VStack>
                  <Icon as={Bookmark} boxSize={4} color="pink.400" />
                </HStack>
              </Box>
            ))}
          </VStack>
        </Box>

        {/* Privacy Notice */}
        <Box
          w="100%"
          p={4}
          bg="rgba(34, 197, 94, 0.1)"
          borderRadius="xl"
          border="1px solid"
          borderColor="green.700"
        >
          <HStack spacing={3} mb={2}>
            <Icon as={Shield} boxSize={5} color="green.400" />
            <Text fontSize="sm" fontWeight="semibold" color="green.300">
              Privacy Promise
            </Text>
          </HStack>
          <Text fontSize="xs" color="gray.300" lineHeight="tall">
            All your data is stored locally on your device. No sign-in required. 
            No data sent to external servers. No tracking. You're in complete control.
          </Text>
        </Box>

        {/* Actions */}
        <VStack spacing={3} w="100%" pb={4}>
          <Button
            leftIcon={<Download size={18} />}
            variant="outline"
            colorScheme="cyan"
            w="100%"
            borderColor="whiteAlpha.300"
            color="white"
            _hover={{ bg: 'whiteAlpha.100' }}
          >
            Export My Data (Local)
          </Button>

          <Button
            leftIcon={<Trash2 size={18} />}
            variant="outline"
            colorScheme="orange"
            w="100%"
            borderColor="whiteAlpha.300"
            color="white"
            _hover={{ bg: 'whiteAlpha.100' }}
          >
            Clear Local Data
          </Button>
        </VStack>
      </VStack>
    </Box>
  );
};

export default Profile;
