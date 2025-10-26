import {
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  VStack,
  HStack,
  Text,
  Box,
  Avatar,
  Badge,
  IconButton
} from '@chakra-ui/react';
import { formatDistanceToNow } from 'date-fns';
import { Check, Sparkles, ShoppingBag, Plane, Heart } from 'lucide-react';

const NotificationsPanel = ({ isOpen, onClose }) => {
  const notifications = [
    {
      id: 1,
      type: 'ai_response',
      title: 'AI Response Ready',
      message: 'Your shopping recommendations for "best laptops" are ready',
      icon: ShoppingBag,
      color: 'purple.400',
      time: new Date(Date.now() - 120000),
      read: false
    },
    {
      id: 2,
      type: 'ai_response',
      title: 'Travel Plan Complete',
      message: 'Your trip to Tokyo itinerary has been generated',
      icon: Plane,
      color: 'cyan.400',
      time: new Date(Date.now() - 1800000),
      read: false
    },
    {
      id: 3,
      type: 'system',
      title: 'New Feature Available',
      message: 'Check out the new Activity Feed to see all your AI interactions',
      icon: Sparkles,
      color: 'pink.400',
      time: new Date(Date.now() - 3600000),
      read: false
    },
    {
      id: 4,
      type: 'ai_response',
      title: 'Health Tips Ready',
      message: 'Your personalized wellness plan is ready to view',
      icon: Heart,
      color: 'red.400',
      time: new Date(Date.now() - 7200000),
      read: true
    },
  ];

  const markAsRead = (id) => {
    console.log('Mark as read:', id);
  };

  const markAllAsRead = () => {
    console.log('Mark all as read');
  };

  return (
    <Drawer isOpen={isOpen} placement="right" onClose={onClose} size="sm">
      <DrawerOverlay backdropFilter="blur(4px)" />
      <DrawerContent
        bg="gray.900"
        borderLeft="1px solid"
        borderColor="whiteAlpha.200"
      >
        <DrawerCloseButton color="gray.300" />
        <DrawerHeader
          borderBottomWidth="1px"
          borderColor="whiteAlpha.200"
          color="white"
        >
          <HStack justify="space-between">
            <Text fontSize="lg" fontWeight="bold">Notifications</Text>
            <Text
              fontSize="xs"
              color="purple.400"
              cursor="pointer"
              _hover={{ color: 'purple.300' }}
              onClick={markAllAsRead}
            >
              Mark all read
            </Text>
          </HStack>
        </DrawerHeader>

        <DrawerBody p={0}>
          <VStack spacing={0} align="stretch">
            {notifications.map((notif) => {
              const IconComponent = notif.icon;
              
              return (
                <Box
                  key={notif.id}
                  p={4}
                  borderBottom="1px solid"
                  borderColor="whiteAlpha.100"
                  bg={notif.read ? 'transparent' : 'whiteAlpha.50'}
                  cursor="pointer"
                  transition="all 0.2s"
                  _hover={{ bg: 'whiteAlpha.100' }}
                  onClick={() => markAsRead(notif.id)}
                >
                  <HStack align="start" spacing={3}>
                    <Box
                      p={2}
                      borderRadius="lg"
                      bg={notif.color}
                      color="white"
                      flexShrink={0}
                    >
                      <IconComponent size={18} />
                    </Box>
                    
                    <VStack align="start" spacing={1} flex="1">
                      <HStack justify="space-between" w="100%">
                        <Text
                          fontSize="sm"
                          fontWeight="semibold"
                          color="white"
                        >
                          {notif.title}
                        </Text>
                        {!notif.read && (
                          <Box
                            w="8px"
                            h="8px"
                            borderRadius="full"
                            bg="purple.400"
                          />
                        )}
                      </HStack>
                      
                      <Text fontSize="xs" color="gray.400" lineHeight="tall">
                        {notif.message}
                      </Text>
                      
                      <Text fontSize="xs" color="gray.500">
                        {formatDistanceToNow(notif.time, { addSuffix: true })}
                      </Text>
                    </VStack>
                  </HStack>
                </Box>
              );
            })}
          </VStack>
        </DrawerBody>
      </DrawerContent>
    </Drawer>
  );
};

export default NotificationsPanel;
