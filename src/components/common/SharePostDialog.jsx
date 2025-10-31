import { useState } from 'react';
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
  Icon,
  Box,
  Input,
  useToast,
  Image,
  Badge,
  Divider,
  IconButton,
  Tooltip
} from '@chakra-ui/react';
import {
  Copy,
  Check,
  Link2,
  Facebook,
  Twitter,
  Linkedin,
  Mail,
  MessageCircle,
  Star,
  TrendingUp
} from 'lucide-react';

const SharePostDialog = ({ isOpen, onClose, post }) => {
  const [copied, setCopied] = useState(false);
  const toast = useToast();

  const shareUrl = post?.url || `${window.location.origin}/post/${post?.id || 'shared'}`;
  const shareTitle = post?.title || 'Check out this post on New Frontier AI Platform';
  const shareDescription = post?.description || 'Discover quality content on our privacy-first AI platform';

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      toast({
        title: 'Link copied!',
        description: 'Share link copied to clipboard',
        status: 'success',
        duration: 2000,
        position: 'top'
      });
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast({
        title: 'Failed to copy',
        description: error.message,
        status: 'error',
        duration: 3000
      });
    }
  };

  const shareOptions = [
    {
      name: 'Twitter',
      icon: Twitter,
      color: 'twitter',
      url: `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareTitle)}&url=${encodeURIComponent(shareUrl)}`,
      bg: '#1DA1F2'
    },
    {
      name: 'Facebook',
      icon: Facebook,
      color: 'facebook',
      url: `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`,
      bg: '#4267B2'
    },
    {
      name: 'LinkedIn',
      icon: Linkedin,
      color: 'linkedin',
      url: `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`,
      bg: '#0077B5'
    },
    {
      name: 'Email',
      icon: Mail,
      color: 'gray',
      url: `mailto:?subject=${encodeURIComponent(shareTitle)}&body=${encodeURIComponent(shareDescription + '\n\n' + shareUrl)}`,
      bg: '#718096'
    }
  ];

  const handleShare = (url) => {
    window.open(url, '_blank', 'width=600,height=400');
  };

  const getQualityColor = (score) => {
    if (score >= 90) return 'green';
    if (score >= 70) return 'blue';
    if (score >= 50) return 'purple';
    return 'gray';
  };

  const qualityScore = post?.qualityScore || post?.adaptive_score?.overall || 75;

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="lg" isCentered>
      <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(10px)" />
      <ModalContent
        bg="gray.900"
        border="1px solid"
        borderColor="whiteAlpha.200"
        borderRadius="2xl"
      >
        <ModalHeader color="white">
          <HStack>
            <Icon as={Link2} boxSize={5} color="purple.400" />
            <Text>Share Post</Text>
          </HStack>
        </ModalHeader>
        <ModalCloseButton color="gray.400" />
        
        <ModalBody>
          <VStack spacing={4} align="stretch">
            <Box
              p={4}
              bg="black"
              borderRadius="xl"
              border="1px solid"
              borderColor="whiteAlpha.200"
            >
              <VStack align="stretch" spacing={3}>
                <HStack justify="space-between">
                  <HStack spacing={3}>
                    <Box
                      w="40px"
                      h="40px"
                      borderRadius="full"
                      bgGradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                    >
                      <Text fontSize="sm" fontWeight="bold" color="white">
                        {post?.source?.charAt(0) || 'A'}
                      </Text>
                    </Box>
                    <VStack align="start" spacing={0}>
                      <Text fontSize="sm" fontWeight="bold" color="white">
                        {post?.source || 'Anonymous'}
                      </Text>
                      <HStack spacing={2}>
                        <Badge colorScheme={getQualityColor(qualityScore)} fontSize="xs">
                          <HStack spacing={1}>
                            <Icon as={Star} boxSize={3} />
                            <Text>{Math.round(qualityScore)}</Text>
                          </HStack>
                        </Badge>
                        {post?.trending && (
                          <Badge colorScheme="orange" fontSize="xs">
                            <HStack spacing={1}>
                              <Icon as={TrendingUp} boxSize={3} />
                              <Text>Trending</Text>
                            </HStack>
                          </Badge>
                        )}
                      </HStack>
                    </VStack>
                  </HStack>
                </HStack>

                {post?.title && (
                  <Text fontSize="md" fontWeight="bold" color="white" noOfLines={2}>
                    {post.title}
                  </Text>
                )}

                {post?.description && (
                  <Text fontSize="sm" color="gray.400" noOfLines={3}>
                    {post.description}
                  </Text>
                )}

                {post?.category && (
                  <HStack spacing={2}>
                    <Badge colorScheme="cyan" variant="subtle" fontSize="xs">
                      {post.category}
                    </Badge>
                  </HStack>
                )}
              </VStack>
            </Box>

            <Divider borderColor="whiteAlpha.200" />

            <Box>
              <Text fontSize="sm" fontWeight="bold" color="white" mb={3}>
                Share via
              </Text>
              <HStack spacing={3} justify="center">
                {shareOptions.map((option) => (
                  <Tooltip key={option.name} label={option.name} placement="top">
                    <IconButton
                      aria-label={`Share on ${option.name}`}
                      icon={<Icon as={option.icon} boxSize={5} />}
                      onClick={() => handleShare(option.url)}
                      bg={option.bg}
                      color="white"
                      size="lg"
                      borderRadius="full"
                      _hover={{ transform: 'scale(1.1)', opacity: 0.9 }}
                      transition="all 0.2s"
                    />
                  </Tooltip>
                ))}
              </HStack>
            </Box>

            <Divider borderColor="whiteAlpha.200" />

            <Box>
              <Text fontSize="sm" fontWeight="bold" color="white" mb={2}>
                Or copy link
              </Text>
              <HStack>
                <Input
                  value={shareUrl}
                  readOnly
                  bg="whiteAlpha.100"
                  border="1px solid"
                  borderColor="whiteAlpha.300"
                  color="white"
                  fontSize="sm"
                  _focus={{ borderColor: 'purple.500' }}
                />
                <Button
                  leftIcon={<Icon as={copied ? Check : Copy} />}
                  onClick={handleCopyLink}
                  colorScheme={copied ? 'green' : 'purple'}
                  variant={copied ? 'solid' : 'outline'}
                  flexShrink={0}
                >
                  {copied ? 'Copied!' : 'Copy'}
                </Button>
              </HStack>
            </Box>
          </VStack>
        </ModalBody>

        <ModalFooter>
          <Button variant="ghost" onClick={onClose} color="gray.400">
            Close
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default SharePostDialog;
