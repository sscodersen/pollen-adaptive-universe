import { HStack, IconButton, Tooltip, useToast, useClipboard } from '@chakra-ui/react';
import { Copy, Check, Share2, Bookmark, Download } from 'lucide-react';
import { useState } from 'react';

const ResponseActions = ({ content, onFavorite, isFavorited = false }) => {
  const { hasCopied, onCopy } = useClipboard(content);
  const toast = useToast();
  const [isBookmarked, setIsBookmarked] = useState(isFavorited);

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Pollen AI Response',
          text: content,
        });
      } catch (error) {
        if (error.name !== 'AbortError') {
          onCopy();
          toast({
            title: 'Copied to clipboard',
            description: 'Response copied! Share manually.',
            status: 'success',
            duration: 2000,
            isClosable: true,
          });
        }
      }
    } else {
      onCopy();
      toast({
        title: 'Copied to clipboard',
        description: 'Share functionality not supported. Copied instead!',
        status: 'info',
        duration: 2000,
        isClosable: true,
      });
    }
  };

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `pollen-ai-${Date.now()}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast({
      title: 'Downloaded',
      description: 'Response saved as text file',
      status: 'success',
      duration: 2000,
      isClosable: true,
    });
  };

  const handleBookmark = () => {
    setIsBookmarked(!isBookmarked);
    if (onFavorite) {
      onFavorite(!isBookmarked);
    }
    toast({
      title: isBookmarked ? 'Removed from favorites' : 'Added to favorites',
      description: 'Stored locally on your device',
      status: 'success',
      duration: 2000,
      isClosable: true,
    });
  };

  return (
    <HStack spacing={2}>
      <Tooltip label={hasCopied ? 'Copied!' : 'Copy response'}>
        <IconButton
          icon={hasCopied ? <Check size={16} /> : <Copy size={16} />}
          onClick={onCopy}
          size="sm"
          variant="ghost"
          color={hasCopied ? 'green.400' : 'gray.400'}
          _hover={{ bg: 'whiteAlpha.200', color: 'white' }}
        />
      </Tooltip>

      <Tooltip label="Share response">
        <IconButton
          icon={<Share2 size={16} />}
          onClick={handleShare}
          size="sm"
          variant="ghost"
          color="gray.400"
          _hover={{ bg: 'whiteAlpha.200', color: 'white' }}
        />
      </Tooltip>

      <Tooltip label={isBookmarked ? 'Remove from favorites' : 'Add to favorites'}>
        <IconButton
          icon={<Bookmark size={16} fill={isBookmarked ? 'currentColor' : 'none'} />}
          onClick={handleBookmark}
          size="sm"
          variant="ghost"
          color={isBookmarked ? 'pink.400' : 'gray.400'}
          _hover={{ bg: 'whiteAlpha.200', color: 'white' }}
        />
      </Tooltip>

      <Tooltip label="Download as text">
        <IconButton
          icon={<Download size={16} />}
          onClick={handleDownload}
          size="sm"
          variant="ghost"
          color="gray.400"
          _hover={{ bg: 'whiteAlpha.200', color: 'white' }}
        />
      </Tooltip>
    </HStack>
  );
};

export default ResponseActions;
