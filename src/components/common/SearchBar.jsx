import { InputGroup, Input, InputLeftElement, Box } from '@chakra-ui/react';
import { Search } from 'lucide-react';

const SearchBar = ({ placeholder = 'Search anything...', onSearch }) => {
  return (
    <Box
      px={4}
      mb={4}
      position="sticky"
      top="120px"
      zIndex={9}
    >
      <InputGroup>
        <InputLeftElement pointerEvents="none" color="gray.500">
          <Search size={20} />
        </InputLeftElement>
        <Input
          placeholder={placeholder}
          bg="whiteAlpha.800"
          backdropFilter="blur(10px)"
          border="1px solid"
          borderColor="whiteAlpha.500"
          borderRadius="full"
          _placeholder={{ color: 'gray.500' }}
          _focus={{
            bg: 'white',
            borderColor: 'brand.400',
            boxShadow: '0 0 0 1px var(--chakra-colors-brand-400)',
          }}
          fontSize="sm"
        />
      </InputGroup>
    </Box>
  );
};

export default SearchBar;
