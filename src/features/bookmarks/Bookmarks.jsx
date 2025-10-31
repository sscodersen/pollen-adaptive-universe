import { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  Heading,
  Text,
  HStack,
  Icon,
  Button,
  Select,
  Input,
  InputGroup,
  InputLeftElement,
  Badge
} from '@chakra-ui/react';
import { Bookmark, Search, Trash2, Filter } from 'lucide-react';
import PostCard from '@components/common/PostCard';

const Bookmarks = () => {
  const [bookmarks, setBookmarks] = useState([]);
  const [filteredBookmarks, setFilteredBookmarks] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('recent');

  useEffect(() => {
    loadBookmarks();
  }, []);

  useEffect(() => {
    filterAndSortBookmarks();
  }, [bookmarks, searchQuery, sortBy]);

  const loadBookmarks = () => {
    const stored = JSON.parse(localStorage.getItem('bookmarkedPosts') || '[]');
    setBookmarks(stored);
  };

  const filterAndSortBookmarks = () => {
    let filtered = bookmarks;

    if (searchQuery) {
      filtered = bookmarks.filter(b =>
        b.post?.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        b.post?.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        b.post?.category?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    switch (sortBy) {
      case 'recent':
        filtered.sort((a, b) => new Date(b.savedAt) - new Date(a.savedAt));
        break;
      case 'quality':
        filtered.sort((a, b) => {
          const scoreA = a.post?.qualityScore || a.post?.adaptive_score?.overall || 0;
          const scoreB = b.post?.qualityScore || b.post?.adaptive_score?.overall || 0;
          return scoreB - scoreA;
        });
        break;
      case 'oldest':
        filtered.sort((a, b) => new Date(a.savedAt) - new Date(b.savedAt));
        break;
    }

    setFilteredBookmarks(filtered);
  };

  const removeBookmark = (id) => {
    const updated = bookmarks.filter(b => b.id !== id);
    localStorage.setItem('bookmarkedPosts', JSON.stringify(updated));
    setBookmarks(updated);
  };

  const clearAllBookmarks = () => {
    if (window.confirm('Are you sure you want to remove all bookmarks?')) {
      localStorage.setItem('bookmarkedPosts', JSON.stringify([]));
      setBookmarks([]);
    }
  };

  return (
    <Box px={4} py={6}>
      <VStack spacing={6} align="stretch">
        <HStack justify="space-between" align="start">
          <Box>
            <HStack spacing={3} mb={2}>
              <Icon as={Bookmark} boxSize={8} color="purple.500" />
              <Heading size="xl" color="white">
                Bookmarks
              </Heading>
            </HStack>
            <Text color="gray.400">
              {bookmarks.length} saved {bookmarks.length === 1 ? 'post' : 'posts'}
            </Text>
          </Box>
          {bookmarks.length > 0 && (
            <Button
              leftIcon={<Icon as={Trash2} />}
              onClick={clearAllBookmarks}
              variant="ghost"
              colorScheme="red"
              size="sm"
            >
              Clear All
            </Button>
          )}
        </HStack>

        {bookmarks.length > 0 && (
          <HStack spacing={4}>
            <InputGroup>
              <InputLeftElement pointerEvents="none">
                <Icon as={Search} color="gray.400" />
              </InputLeftElement>
              <Input
                placeholder="Search bookmarks..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                bg="whiteAlpha.100"
                border="1px solid"
                borderColor="whiteAlpha.300"
                color="white"
                _focus={{ borderColor: 'purple.500' }}
              />
            </InputGroup>
            <Select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              bg="whiteAlpha.100"
              color="white"
              borderColor="whiteAlpha.300"
              maxW="200px"
              icon={<Icon as={Filter} />}
            >
              <option value="recent" style={{ background: '#1a202c' }}>Most Recent</option>
              <option value="quality" style={{ background: '#1a202c' }}>Highest Quality</option>
              <option value="oldest" style={{ background: '#1a202c' }}>Oldest First</option>
            </Select>
          </HStack>
        )}

        {filteredBookmarks.length === 0 && bookmarks.length > 0 && searchQuery && (
          <Box
            w="100%"
            p={8}
            textAlign="center"
            bg="whiteAlpha.50"
            borderRadius="xl"
            border="1px dashed"
            borderColor="whiteAlpha.300"
          >
            <Icon as={Search} boxSize={12} color="gray.400" mb={4} />
            <Text color="gray.500" fontSize="lg">
              No bookmarks match your search
            </Text>
          </Box>
        )}

        {bookmarks.length === 0 && (
          <Box
            w="100%"
            p={8}
            textAlign="center"
            bg="whiteAlpha.50"
            borderRadius="xl"
            border="1px dashed"
            borderColor="whiteAlpha.300"
          >
            <Icon as={Bookmark} boxSize={12} color="gray.400" mb={4} />
            <Text color="gray.500" fontSize="lg" mb={2}>
              No bookmarks yet
            </Text>
            <Text color="gray.600" fontSize="sm">
              Bookmark posts to save them for later
            </Text>
          </Box>
        )}

        <VStack spacing={4} align="stretch">
          {filteredBookmarks.map((bookmark) => (
            <Box key={bookmark.id} position="relative">
              <PostCard post={bookmark.post} />
              <Badge
                position="absolute"
                top={4}
                right={4}
                colorScheme="purple"
                fontSize="xs"
                px={2}
                py={1}
              >
                Saved {new Date(bookmark.savedAt).toLocaleDateString()}
              </Badge>
            </Box>
          ))}
        </VStack>
      </VStack>
    </Box>
  );
};

export default Bookmarks;
