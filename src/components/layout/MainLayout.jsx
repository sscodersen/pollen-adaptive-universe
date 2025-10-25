import { Box, Container } from '@chakra-ui/react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import BottomNavigation from './BottomNavigation';
import UnifiedSearchBar from '../common/UnifiedSearchBar';

const MainLayout = () => {
  return (
    <Box
      minH="100vh"
      bgGradient="linear(to-br, gray.900, purple.900, gray.900)"
      position="relative"
      overflow="hidden"
    >
      <Container
        maxW="480px"
        h="100vh"
        p={0}
        display="flex"
        flexDirection="column"
        bg="transparent"
      >
        <Header />
        
        <Box
          flex="1"
          overflowY="auto"
          overflowX="hidden"
          pb="80px"
          css={{
            '&::-webkit-scrollbar': {
              display: 'none',
            },
            scrollbarWidth: 'none',
          }}
        >
          <Outlet />
        </Box>

        <BottomNavigation />
        <UnifiedSearchBar variant="floating" />
      </Container>
    </Box>
  );
};

export default MainLayout;
