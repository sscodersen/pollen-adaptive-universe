import { Box, Flex, IconButton, Text, VStack } from '@chakra-ui/react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Home, Compass, Newspaper, User } from 'lucide-react';

const BottomNavigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { icon: Home, label: 'Home', path: '/' },
    { icon: Compass, label: 'Explore', path: '/explore' },
    { icon: Newspaper, label: 'News', path: '/news' },
    { icon: User, label: 'Profile', path: '/profile' },
  ];

  return (
    <Box
      position="fixed"
      bottom={0}
      left="50%"
      transform="translateX(-50%)"
      width="100%"
      maxW="480px"
      bg="whiteAlpha.800"
      backdropFilter="blur(20px)"
      borderTopRadius="2xl"
      boxShadow="0 -4px 20px rgba(0,0,0,0.1)"
      borderTop="1px solid"
      borderColor="whiteAlpha.400"
      px={4}
      py={2}
      zIndex={100}
    >
      <Flex justify="space-around" align="center">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          
          return (
            <VStack
              key={item.path}
              spacing={0}
              onClick={() => navigate(item.path)}
              cursor="pointer"
              color={isActive ? 'brand.500' : 'gray.600'}
              transition="all 0.2s"
              _hover={{ color: 'brand.500' }}
            >
              <IconButton
                icon={<Icon size={22} />}
                variant="ghost"
                size="sm"
                aria-label={item.label}
                color="inherit"
              />
              <Text fontSize="xs" fontWeight={isActive ? 'bold' : 'normal'}>
                {item.label}
              </Text>
            </VStack>
          );
        })}
      </Flex>
    </Box>
  );
};

export default BottomNavigation;
