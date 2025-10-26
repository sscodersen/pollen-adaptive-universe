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
  Icon,
  Box,
  Badge
} from '@chakra-ui/react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Home, 
  Compass, 
  Newspaper, 
  User, 
  ShoppingBag,
  Plane,
  Sparkles,
  Home as HomeIcon,
  Heart,
  GraduationCap,
  Activity,
  DollarSign,
  Code
} from 'lucide-react';

const NavigationSidebar = ({ isOpen, onClose }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { icon: Home, label: 'Home', path: '/', category: 'Main' },
    { icon: Compass, label: 'Explore', path: '/explore', category: 'Main' },
    { icon: Activity, label: 'Activity', path: '/activity', category: 'Main', badge: 'New' },
    { icon: User, label: 'Profile', path: '/profile', category: 'Main' },
    { icon: ShoppingBag, label: 'Shopping', path: '/shopping', category: 'Features' },
    { icon: Plane, label: 'Travel', path: '/travel', category: 'Features' },
    { icon: Newspaper, label: 'News', path: '/news', category: 'Features' },
    { icon: Sparkles, label: 'Content', path: '/content', category: 'Features' },
    { icon: HomeIcon, label: 'Smart Home', path: '/smarthome', category: 'Features' },
    { icon: Heart, label: 'Wellness', path: '/health', category: 'Features' },
    { icon: GraduationCap, label: 'Learn', path: '/education', category: 'Features' },
    { icon: DollarSign, label: 'Finance', path: '/finance', category: 'Features', badge: 'New' },
    { icon: Code, label: 'Code Helper', path: '/code', category: 'Features', badge: 'New' },
  ];

  const handleNavigation = (path) => {
    navigate(path);
    onClose();
  };

  const groupedItems = navItems.reduce((acc, item) => {
    if (!acc[item.category]) {
      acc[item.category] = [];
    }
    acc[item.category].push(item);
    return acc;
  }, {});

  return (
    <Drawer isOpen={isOpen} placement="left" onClose={onClose} size="xs">
      <DrawerOverlay backdropFilter="blur(4px)" />
      <DrawerContent 
        bg="gray.900" 
        borderRight="1px solid"
        borderColor="whiteAlpha.200"
      >
        <DrawerCloseButton color="gray.300" />
        <DrawerHeader 
          borderBottomWidth="1px" 
          borderColor="whiteAlpha.200"
          color="white"
        >
          <HStack spacing={2}>
            <Box
              w="32px"
              h="32px"
              borderRadius="lg"
              bgGradient="linear(to-br, purple.400, pink.400)"
              display="flex"
              alignItems="center"
              justifyContent="center"
            >
              <Sparkles size={18} />
            </Box>
            <Text fontSize="lg" fontWeight="bold">Pollen AI</Text>
          </HStack>
        </DrawerHeader>

        <DrawerBody p={4}>
          <VStack spacing={6} align="stretch">
            {Object.entries(groupedItems).map(([category, items]) => (
              <Box key={category}>
                <Text
                  fontSize="xs"
                  fontWeight="bold"
                  color="gray.500"
                  mb={2}
                  px={2}
                  letterSpacing="wide"
                >
                  {category.toUpperCase()}
                </Text>
                <VStack spacing={1} align="stretch">
                  {items.map((item) => {
                    const IconComponent = item.icon;
                    const isActive = location.pathname === item.path;
                    
                    return (
                      <HStack
                        key={item.path}
                        p={3}
                        borderRadius="lg"
                        cursor="pointer"
                        bg={isActive ? 'whiteAlpha.200' : 'transparent'}
                        color={isActive ? 'purple.300' : 'gray.300'}
                        transition="all 0.2s"
                        _hover={{
                          bg: 'whiteAlpha.100',
                          color: 'white',
                          transform: 'translateX(4px)'
                        }}
                        onClick={() => handleNavigation(item.path)}
                      >
                        <Icon as={IconComponent} boxSize={5} />
                        <Text fontSize="sm" fontWeight="medium" flex="1">
                          {item.label}
                        </Text>
                        {item.badge && (
                          <Badge
                            colorScheme="purple"
                            fontSize="xs"
                            px={2}
                          >
                            {item.badge}
                          </Badge>
                        )}
                      </HStack>
                    );
                  })}
                </VStack>
              </Box>
            ))}
          </VStack>
        </DrawerBody>
      </DrawerContent>
    </Drawer>
  );
};

export default NavigationSidebar;
