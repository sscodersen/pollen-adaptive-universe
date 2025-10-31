import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Button,
  VStack,
  HStack,
  Text,
  Switch,
  Select,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Wrap,
  WrapItem,
  Badge,
  Icon,
  Box,
  Divider,
  Heading,
  useToast
} from '@chakra-ui/react';
import { Settings, Star, Zap, Eye, RefreshCw } from 'lucide-react';
import { usePersonalization } from '@hooks/usePersonalization';

const AVAILABLE_INTERESTS = [
  'Technology', 'Business', 'Science', 'Health', 'Finance',
  'Education', 'Entertainment', 'Sports', 'Politics', 'Environment',
  'Art', 'Travel', 'Food', 'Gaming', 'Music', 'Fashion'
];

const PersonalizationSettings = ({ isOpen, onClose }) => {
  const {
    interests,
    preferences,
    toggleInterest,
    updatePreference
  } = usePersonalization();
  
  const toast = useToast();

  const handleSave = () => {
    toast({
      title: 'Settings saved',
      description: 'Your preferences have been updated',
      status: 'success',
      duration: 2000,
      position: 'top'
    });
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl" isCentered scrollBehavior="inside">
      <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(10px)" />
      <ModalContent
        bg="gray.900"
        border="1px solid"
        borderColor="whiteAlpha.200"
        borderRadius="2xl"
        maxH="90vh"
      >
        <ModalHeader color="white">
          <HStack>
            <Icon as={Settings} boxSize={6} color="purple.400" />
            <Text>Personalization Settings</Text>
          </HStack>
        </ModalHeader>
        <ModalCloseButton color="gray.400" />
        
        <ModalBody>
          <VStack spacing={6} align="stretch">
            <Box>
              <HStack mb={3}>
                <Icon as={Star} boxSize={5} color="purple.400" />
                <Heading size="sm" color="white">
                  Your Interests
                </Heading>
              </HStack>
              <Text fontSize="sm" color="gray.400" mb={3}>
                Select topics you're interested in to personalize your feed
              </Text>
              <Wrap spacing={2}>
                {AVAILABLE_INTERESTS.map((interest) => {
                  const isSelected = interests.includes(interest.toLowerCase());
                  return (
                    <WrapItem key={interest}>
                      <Badge
                        px={3}
                        py={2}
                        borderRadius="full"
                        cursor="pointer"
                        fontSize="sm"
                        colorScheme={isSelected ? 'purple' : 'gray'}
                        bg={isSelected ? 'purple.600' : 'whiteAlpha.100'}
                        border="1px solid"
                        borderColor={isSelected ? 'purple.500' : 'whiteAlpha.200'}
                        onClick={() => toggleInterest(interest.toLowerCase())}
                        _hover={{
                          bg: isSelected ? 'purple.700' : 'whiteAlpha.200',
                          transform: 'scale(1.05)'
                        }}
                        transition="all 0.2s"
                      >
                        {interest}
                      </Badge>
                    </WrapItem>
                  );
                })}
              </Wrap>
            </Box>

            <Divider borderColor="whiteAlpha.200" />

            <Box>
              <HStack mb={3}>
                <Icon as={Zap} boxSize={5} color="purple.400" />
                <Heading size="sm" color="white">
                  Feed Algorithm
                </Heading>
              </HStack>
              <Select
                value={preferences.feedAlgorithm}
                onChange={(e) => updatePreference('feedAlgorithm', e.target.value)}
                bg="whiteAlpha.100"
                color="white"
                borderColor="whiteAlpha.300"
                _focus={{ borderColor: 'purple.500' }}
              >
                <option value="personalized" style={{ background: '#1a202c' }}>
                  Personalized (Recommended)
                </option>
                <option value="quality" style={{ background: '#1a202c' }}>
                  Highest Quality First
                </option>
                <option value="chronological" style={{ background: '#1a202c' }}>
                  Most Recent First
                </option>
              </Select>
            </Box>

            <Box>
              <HStack mb={3} justify="space-between">
                <HStack>
                  <Icon as={Eye} boxSize={5} color="purple.400" />
                  <Heading size="sm" color="white">
                    Minimum Quality Score
                  </Heading>
                </HStack>
                <Badge colorScheme="purple" fontSize="md" px={3} py={1}>
                  {preferences.minQualityScore}+
                </Badge>
              </HStack>
              <Slider
                value={preferences.minQualityScore}
                onChange={(val) => updatePreference('minQualityScore', val)}
                min={0}
                max={100}
                step={10}
                colorScheme="purple"
              >
                <SliderTrack bg="whiteAlpha.200">
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb boxSize={6} bg="purple.500" />
              </Slider>
              <HStack justify="space-between" mt={1}>
                <Text fontSize="xs" color="gray.500">Show all</Text>
                <Text fontSize="xs" color="gray.500">High quality only</Text>
              </HStack>
            </Box>

            <Divider borderColor="whiteAlpha.200" />

            <Box>
              <Heading size="sm" color="white" mb={4}>
                Display Options
              </Heading>
              <VStack spacing={4} align="stretch">
                <HStack justify="space-between">
                  <VStack align="start" spacing={0}>
                    <Text fontSize="sm" fontWeight="medium" color="white">
                      Show Trending Badge
                    </Text>
                    <Text fontSize="xs" color="gray.400">
                      Highlight trending content
                    </Text>
                  </VStack>
                  <Switch
                    isChecked={preferences.showTrending}
                    onChange={(e) => updatePreference('showTrending', e.target.checked)}
                    colorScheme="purple"
                    size="lg"
                  />
                </HStack>

                <HStack justify="space-between">
                  <VStack align="start" spacing={0}>
                    <HStack>
                      <Icon as={RefreshCw} boxSize={4} color="gray.400" />
                      <Text fontSize="sm" fontWeight="medium" color="white">
                        Auto-refresh Feed
                      </Text>
                    </HStack>
                    <Text fontSize="xs" color="gray.400">
                      Automatically update content
                    </Text>
                  </VStack>
                  <Switch
                    isChecked={preferences.autoRefresh}
                    onChange={(e) => updatePreference('autoRefresh', e.target.checked)}
                    colorScheme="purple"
                    size="lg"
                  />
                </HStack>

                <HStack justify="space-between">
                  <VStack align="start" spacing={0}>
                    <Text fontSize="sm" fontWeight="medium" color="white">
                      Compact View
                    </Text>
                    <Text fontSize="xs" color="gray.400">
                      Show more posts at once
                    </Text>
                  </VStack>
                  <Switch
                    isChecked={preferences.compactView}
                    onChange={(e) => updatePreference('compactView', e.target.checked)}
                    colorScheme="purple"
                    size="lg"
                  />
                </HStack>
              </VStack>
            </Box>

            <Box
              p={4}
              bg="purple.900"
              bgGradient="linear-gradient(135deg, rgba(103, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)"
              borderRadius="xl"
              border="1px solid"
              borderColor="purple.700"
            >
              <Text fontSize="sm" color="gray.300" textAlign="center">
                <Icon as={Star} display="inline" boxSize={4} mr={1} color="purple.300" />
                Your preferences are stored locally and never leave your device
              </Text>
            </Box>
          </VStack>
        </ModalBody>

        <ModalFooter>
          <Button variant="ghost" onClick={onClose} mr={3} color="gray.400">
            Cancel
          </Button>
          <Button colorScheme="purple" onClick={handleSave}>
            Save Settings
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default PersonalizationSettings;
