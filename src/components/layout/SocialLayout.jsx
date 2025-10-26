import { Box, Container, Grid, GridItem } from '@chakra-ui/react';
import { Outlet } from 'react-router-dom';
import LeftSidebar from './LeftSidebar';
import RightSidebar from './RightSidebar';
import UnifiedSearchBar from '../common/UnifiedSearchBar';

const SocialLayout = () => {
  return (
    <Box
      minH="100vh"
      bg="#1a1a1a"
      position="relative"
      overflow="hidden"
    >
      <Container
        maxW="1400px"
        h="100vh"
        p={0}
        display="flex"
        flexDirection="column"
      >
        <Grid
          templateColumns={{ base: '1fr', lg: '280px 1fr 340px' }}
          gap={4}
          h="100vh"
          p={4}
          pt={2}
        >
          <GridItem
            display={{ base: 'none', lg: 'block' }}
            overflowY="auto"
            css={{
              '&::-webkit-scrollbar': {
                width: '6px',
              },
              '&::-webkit-scrollbar-track': {
                background: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                background: '#333',
                borderRadius: '3px',
              },
              '&::-webkit-scrollbar-thumb:hover': {
                background: '#444',
              },
            }}
          >
            <LeftSidebar />
          </GridItem>

          <GridItem
            overflowY="auto"
            css={{
              '&::-webkit-scrollbar': {
                width: '6px',
              },
              '&::-webkit-scrollbar-track': {
                background: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                background: '#333',
                borderRadius: '3px',
              },
              '&::-webkit-scrollbar-thumb:hover': {
                background: '#444',
              },
            }}
          >
            <Outlet />
          </GridItem>

          <GridItem
            display={{ base: 'none', lg: 'block' }}
            overflowY="auto"
            css={{
              '&::-webkit-scrollbar': {
                width: '6px',
              },
              '&::-webkit-scrollbar-track': {
                background: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                background: '#333',
                borderRadius: '3px',
              },
              '&::-webkit-scrollbar-thumb:hover': {
                background: '#444',
              },
            }}
          >
            <RightSidebar />
          </GridItem>
        </Grid>

        <UnifiedSearchBar variant="floating" />
      </Container>
    </Box>
  );
};

export default SocialLayout;
