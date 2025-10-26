import { useState, useEffect } from 'react';
import { 
  Box, 
  Flex, 
  Text, 
  Avatar, 
  IconButton,
  useDisclosure,
  Badge
} from '@chakra-ui/react';
import { Bell, Menu } from 'lucide-react';
import NavigationSidebar from './NavigationSidebar';
import NotificationsPanel from './NotificationsPanel';

const Header = () => {
  const userName = 'Jane';
  const [greeting, setGreeting] = useState('');
  const { isOpen: isMenuOpen, onOpen: onMenuOpen, onClose: onMenuClose } = useDisclosure();
  const { isOpen: isNotifOpen, onOpen: onNotifOpen, onClose: onNotifClose } = useDisclosure();

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting('Good morning');
    else if (hour < 18) setGreeting('Good afternoon');
    else setGreeting('Good evening');
  }, []);

  return (
    <>
      <Box
        px={4}
        pt={6}
        pb={4}
        position="sticky"
        top={0}
        zIndex={10}
        bg="#1a1a1a"
        borderBottom="1px solid"
        borderColor="whiteAlpha.100"
      >
        <Flex justify="space-between" align="center">
          <IconButton
            icon={<Menu size={24} />}
            variant="ghost"
            color="gray.300"
            aria-label="Menu"
            size="sm"
            onClick={onMenuOpen}
            _hover={{ bg: 'whiteAlpha.200', color: 'white' }}
          />
          
          <Flex align="center" gap={3}>
            <Box textAlign="right" display={{ base: 'none', sm: 'block' }}>
              <Text fontSize="sm" fontWeight="600" color="white">
                {greeting}, {userName}!
              </Text>
            </Box>
            <Box position="relative">
              <IconButton
                icon={<Bell size={20} />}
                variant="ghost"
                color="gray.300"
                aria-label="Notifications"
                size="sm"
                onClick={onNotifOpen}
                _hover={{ bg: 'whiteAlpha.200', color: 'white' }}
              />
              <Badge
                position="absolute"
                top="-1"
                right="-1"
                colorScheme="red"
                borderRadius="full"
                fontSize="xs"
                px={1.5}
              >
                3
              </Badge>
            </Box>
            <Avatar size="sm" name={userName} bg="purple.500" cursor="pointer" />
          </Flex>
        </Flex>
      </Box>

      <NavigationSidebar isOpen={isMenuOpen} onClose={onMenuClose} />
      <NotificationsPanel isOpen={isNotifOpen} onClose={onNotifClose} />
    </>
  );
};

export default Header;