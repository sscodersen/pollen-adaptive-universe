import { useState, useEffect } from 'react';
import { Box, Flex, Text, Avatar, IconButton } from '@chakra-ui/react';
import { Bell, Menu } from 'lucide-react';

const Header = () => {
  const userName = 'Jane';
  const [greeting, setGreeting] = useState('');

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting('Good morning');
    else if (hour < 18) setGreeting('Good afternoon');
    else setGreeting('Good evening');
  }, []);

  return (
    <Box
      px={4}
      pt={6}
      pb={4}
      position="sticky"
      top={0}
      zIndex={10}
      backdropFilter="blur(10px)"
      bg="whiteAlpha.300"
    >
      <Flex justify="space-between" align="center">
        <IconButton
          icon={<Menu size={24} />}
          variant="ghost"
          color="gray.700"
          aria-label="Menu"
          size="sm"
        />
        
        <Flex align="center" gap={3}>
          <Box textAlign="right" display={{ base: 'none', sm: 'block' }}>
            <Text fontSize="sm" fontWeight="600" color="gray.800">
              {greeting}, {userName}!
            </Text>
          </Box>
          <IconButton
            icon={<Bell size={20} />}
            variant="ghost"
            color="gray.700"
            aria-label="Notifications"
            size="sm"
          />
          <Avatar size="sm" name={userName} bg="purple.500" />
        </Flex>
      </Flex>
    </Box>
  );
};

export default Header;