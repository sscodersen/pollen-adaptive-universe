import { Box, Container } from '@chakra-ui/react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import UnifiedSearchBar from '../common/UnifiedSearchBar';

const MainLayout = () => {
  return (
    <Box
      minH="100vh"
      bg="#1a1a1a"
      position="relative"
      overflow="hidden"
    >
      <Container
        maxW="800px"
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
          pb="24px"
          css={{
            '&::-webkit-scrollbar': {
              display: 'none',
            },
            scrollbarWidth: 'none',
          }}
        >
          <Outlet />
        </Box>

        <UnifiedSearchBar variant="floating" />
      </Container>
    </Box>
  );
};

export default MainLayout;
