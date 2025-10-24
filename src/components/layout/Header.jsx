import { Box, Flex, Text, Avatar, IconButton } from '@chakra-ui/react';
import { Bell, Menu } from 'lucide-react';
import { format } from 'date-fns';

const Header = () => {
  const userName = 'Jane';
  const currentTime = format(new Date(), 'EEEE, MMMM d');

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
      <Flex justify="space-between" align="center" mb={4}>
        <IconButton
          icon={<Menu size={24} />}
          variant="ghost"
          color="gray.700"
          aria-label="Menu"
          size="sm"
        />
        
        <Flex gap={2}>
          <IconButton
            icon={<Bell size={20} />}
            variant="ghost"
            color="gray.700"
            aria-label="Notifications"
            size="sm"
          />
          <Avatar size="sm" name={userName} bg="brand.500" />
        </Flex>
      </Flex>

      <Box>
        <Text fontSize="2xl" fontWeight="bold" color="gray.800" mb={1}>
          Hey {userName},
        </Text>
        <Text fontSize="md" color="gray.600">
          Welcome back!
        </Text>
      </Box>
    </Box>
  );
};

export default Header;
