import React, { useState, useEffect } from 'react';
import { Bell, X, Check, Info, AlertTriangle, CheckCircle, Download, Star } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { storageService } from '@/services/storageService';

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'download' | 'rating' | 'update';
  title: string;
  message: string;
  timestamp: number;
  read: boolean;
  actionUrl?: string;
  actionText?: string;
}

export const NotificationSystem: React.FC = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    loadNotifications();
    
    // Create initial notifications for demo
    if (notifications.length === 0) {
      createInitialNotifications();
    }
  }, []);

  const loadNotifications = () => {
    const stored = storageService.getItem<Notification[]>('notifications', []);
    setNotifications(stored);
  };

  const saveNotifications = (newNotifications: Notification[]) => {
    setNotifications(newNotifications);
    storageService.setItem('notifications', newNotifications);
  };

  const createInitialNotifications = () => {
    const initialNotifications: Notification[] = [
      {
        id: '1',
        type: 'success',
        title: 'Welcome to Pollen Platform!',
        message: 'Your account has been set up successfully. Start exploring AI-powered content.',
        timestamp: Date.now() - 300000, // 5 minutes ago
        read: false
      },
      {
        id: '2',
        type: 'download',
        title: 'New Music Track Available',
        message: 'AI-generated ambient track "Digital Dreams" is ready for download.',
        timestamp: Date.now() - 600000, // 10 minutes ago
        read: false,
        actionUrl: '/music',
        actionText: 'Listen Now'
      },
      {
        id: '3',
        type: 'info',
        title: 'Platform Update',
        message: 'New features added: Enhanced search and improved recommendations.',
        timestamp: Date.now() - 900000, // 15 minutes ago
        read: false
      },
      {
        id: '4',
        type: 'rating',
        title: 'Rate Your Experience',
        message: 'Help us improve by rating your recent downloads.',
        timestamp: Date.now() - 1200000, // 20 minutes ago
        read: true,
        actionUrl: '/ratings',
        actionText: 'Rate Now'
      }
    ];

    saveNotifications(initialNotifications);
  };

  const addNotification = (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
    const newNotification: Notification = {
      ...notification,
      id: Date.now().toString(),
      timestamp: Date.now(),
      read: false
    };

    const updatedNotifications = [newNotification, ...notifications];
    saveNotifications(updatedNotifications);
    
    // Track notification
    storageService.trackEvent('notification_created', {
      type: notification.type,
      title: notification.title
    });
  };

  const markAsRead = (id: string) => {
    const updatedNotifications = notifications.map(notif =>
      notif.id === id ? { ...notif, read: true } : notif
    );
    saveNotifications(updatedNotifications);
    storageService.trackEvent('notification_read', { notificationId: id });
  };

  const markAllAsRead = () => {
    const updatedNotifications = notifications.map(notif => ({ ...notif, read: true }));
    saveNotifications(updatedNotifications);
    storageService.trackEvent('notifications_mark_all_read', { count: unreadCount });
  };

  const deleteNotification = (id: string) => {
    const updatedNotifications = notifications.filter(notif => notif.id !== id);
    saveNotifications(updatedNotifications);
    storageService.trackEvent('notification_deleted', { notificationId: id });
  };

  const clearAll = () => {
    saveNotifications([]);
    storageService.trackEvent('notifications_cleared', { count: notifications.length });
  };

  const getIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'download': return <Download className="w-5 h-5 text-blue-500" />;
      case 'rating': return <Star className="w-5 h-5 text-purple-500" />;
      case 'update': return <Info className="w-5 h-5 text-cyan-500" />;
      default: return <Info className="w-5 h-5 text-muted-foreground" />;
    }
  };

  const formatTime = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return `${Math.floor(diff / 86400000)}d ago`;
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  // Expose addNotification globally for other components to use
  useEffect(() => {
    (window as any).addNotification = addNotification;
    return () => {
      delete (window as any).addNotification;
    };
  }, [notifications]);

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Button 
          variant="ghost" 
          size="sm" 
          className="relative p-2 hover:bg-muted"
          onClick={() => storageService.trackEvent('notifications_opened', {})}
        >
          <Bell className="w-5 h-5 text-muted-foreground" />
          {unreadCount > 0 && (
            <Badge 
              className="absolute -top-1 -right-1 px-1 min-w-[1.2rem] h-5 text-xs bg-primary text-primary-foreground"
            >
              {unreadCount > 99 ? '99+' : unreadCount}
            </Badge>
          )}
        </Button>
      </SheetTrigger>

      <SheetContent className="w-96 bg-background border-border">
        <SheetHeader>
          <SheetTitle className="text-foreground flex items-center justify-between">
            Notifications
            {notifications.length > 0 && (
              <div className="flex space-x-2">
                {unreadCount > 0 && (
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={markAllAsRead}
                    className="text-xs text-muted-foreground hover:text-foreground"
                  >
                    Mark all read
                  </Button>
                )}
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={clearAll}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Clear all
                </Button>
              </div>
            )}
          </SheetTitle>
        </SheetHeader>

        <ScrollArea className="h-[calc(100vh-8rem)] mt-4">
          <div className="space-y-3">
            {notifications.length === 0 ? (
              <Card className="bg-card border-border">
                <CardContent className="p-6 text-center">
                  <Bell className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
                  <p className="text-muted-foreground">No notifications yet</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    You'll see updates and alerts here
                  </p>
                </CardContent>
              </Card>
            ) : (
              notifications.map((notification) => (
                <Card 
                  key={notification.id} 
                  className={`bg-card border-border transition-all hover:bg-card/80 ${
                    !notification.read ? 'ring-1 ring-primary/20' : ''
                  }`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 mt-0.5">
                        {getIcon(notification.type)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between">
                          <h4 className={`text-sm font-medium ${
                            !notification.read ? 'text-foreground' : 'text-muted-foreground'
                          }`}>
                            {notification.title}
                          </h4>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => deleteNotification(notification.id)}
                            className="p-1 h-auto text-muted-foreground hover:text-foreground"
                          >
                            <X className="w-3 h-3" />
                          </Button>
                        </div>
                        
                        <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
                          {notification.message}
                        </p>
                        
                        <div className="flex items-center justify-between mt-2">
                          <span className="text-xs text-muted-foreground">
                            {formatTime(notification.timestamp)}
                          </span>
                          
                          <div className="flex items-center space-x-2">
                            {notification.actionUrl && (
                              <Button 
                                variant="outline" 
                                size="sm"
                                className="text-xs h-7 px-2"
                                onClick={() => {
                                  storageService.trackEvent('notification_action_clicked', {
                                    notificationId: notification.id,
                                    actionUrl: notification.actionUrl
                                  });
                                  markAsRead(notification.id);
                                }}
                              >
                                {notification.actionText || 'View'}
                              </Button>
                            )}
                            
                            {!notification.read && (
                              <Button 
                                variant="ghost" 
                                size="sm"
                                onClick={() => markAsRead(notification.id)}
                                className="text-xs h-7 px-2 text-muted-foreground hover:text-foreground"
                              >
                                <Check className="w-3 h-3" />
                              </Button>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
};