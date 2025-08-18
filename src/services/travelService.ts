// Travel and Hospitality AI Service
import { pollenAdaptiveService } from './pollenAdaptiveService';

export interface TravelItinerary {
  id: string;
  destination: string;
  duration: number; // days
  budget: number;
  activities: Array<{
    day: number;
    time: string;
    activity: string;
    location: string;
    cost: number;
    description: string;
  }>;
  accommodations: Array<{
    name: string;
    type: string;
    pricePerNight: number;
    rating: number;
    amenities: string[];
  }>;
  transportation: Array<{
    type: string;
    from: string;
    to: string;
    cost: number;
    duration: string;
  }>;
  totalCost: number;
  tips: string[];
}

export interface AccommodationRecommendation {
  id: string;
  name: string;
  type: 'hotel' | 'resort' | 'airbnb' | 'hostel' | 'apartment';
  rating: number;
  priceRange: string;
  location: string;
  amenities: string[];
  highlights: string[];
  suitableFor: string[];
  bookingTips: string[];
}

export interface LocalGuide {
  destination: string;
  bestTimeToVisit: string;
  weather: {season: string; temperature: string; description: string}[];
  localCulture: {
    language: string;
    currency: string;
    customs: string[];
    etiquette: string[];
  };
  mustSee: Array<{name: string; description: string; cost: string}>;
  hiddenGems: Array<{name: string; description: string; tip: string}>;
  foodRecommendations: Array<{dish: string; where: string; price: string}>;
  transportation: {
    options: string[];
    costs: string[];
    tips: string[];
  };
  safety: {
    rating: 'low' | 'medium' | 'high';
    tips: string[];
    emergencyNumbers: string[];
  };
  budgetTips: string[];
}

export class TravelService {
  async generateItinerary(preferences: {
    destination: string;
    duration: number;
    budget: number;
    interests: string[];
    travelStyle: string;
  }): Promise<TravelItinerary> {
    const proposal = await pollenAdaptiveService.proposeTask(
      `Create ${preferences.duration}-day itinerary for ${preferences.destination} with budget $${preferences.budget}`
    );

    const activities = this.generateActivities(preferences);
    const accommodations = this.generateAccommodations(preferences);
    const transportation = this.generateTransportation(preferences);
    
    const totalCost = this.calculateTotalCost(activities, accommodations, transportation);

    return {
      id: `itinerary_${Date.now()}`,
      destination: preferences.destination,
      duration: preferences.duration,
      budget: preferences.budget,
      activities,
      accommodations,
      transportation,
      totalCost,
      tips: this.generateTravelTips(preferences.destination)
    };
  }

  async recommendAccommodations(criteria: {
    destination: string;
    checkIn: string;
    checkOut: string;
    guests: number;
    budget: number;
    preferences: string[];
  }): Promise<AccommodationRecommendation[]> {
    const recommendations: AccommodationRecommendation[] = [];
    const types: Array<'hotel' | 'resort' | 'airbnb' | 'hostel' | 'apartment'> = ['hotel', 'resort', 'airbnb', 'hostel', 'apartment'];

    for (let i = 0; i < 5; i++) {
      const type = types[i % types.length];
      recommendations.push({
        id: `accommodation_${Date.now()}_${i}`,
        name: this.generateAccommodationName(type, criteria.destination),
        type,
        rating: Number((3.5 + Math.random() * 1.5).toFixed(1)),
        priceRange: this.generatePriceRange(type, criteria.budget),
        location: this.generateLocationDescription(criteria.destination),
        amenities: this.generateAmenities(type),
        highlights: this.generateHighlights(type),
        suitableFor: this.generateSuitability(type),
        bookingTips: this.generateBookingTips(type)
      });
    }

    return recommendations;
  }

  async createLocalGuide(destination: string): Promise<LocalGuide> {
    const trendAnalysis = await pollenAdaptiveService.analyzeTrends(`travel ${destination}`);
    
    return {
      destination,
      bestTimeToVisit: this.determineBestTime(destination),
      weather: this.generateWeatherInfo(destination),
      localCulture: this.generateCulturalInfo(destination),
      mustSee: this.generateMustSeeAttractions(destination),
      hiddenGems: this.generateHiddenGems(destination),
      foodRecommendations: this.generateFoodRecommendations(destination),
      transportation: this.generateTransportationInfo(destination),
      safety: this.generateSafetyInfo(destination),
      budgetTips: this.generateBudgetTips(destination)
    };
  }

  async getWeatherForecast(destination: string, dates: string[]): Promise<Array<{
    date: string;
    temperature: {high: number; low: number};
    condition: string;
    precipitation: number;
    recommendation: string;
  }>> {
    return dates.map(date => ({
      date,
      temperature: {
        high: 65 + Math.random() * 25,
        low: 45 + Math.random() * 20
      },
      condition: this.generateWeatherCondition(),
      precipitation: Math.random() * 30,
      recommendation: this.generateWeatherRecommendation()
    }));
  }

  private generateActivities(preferences: any): Array<{day: number; time: string; activity: string; location: string; cost: number; description: string}> {
    const activities = [];
    const baseActivities = [
      'City walking tour',
      'Museum visit',
      'Local market exploration',
      'Restaurant dining',
      'Historical site tour',
      'Cultural performance',
      'Shopping district visit',
      'Scenic viewpoint'
    ];

    for (let day = 1; day <= preferences.duration; day++) {
      const dailyActivities = Math.min(3, baseActivities.length);
      for (let i = 0; i < dailyActivities; i++) {
        const activity = baseActivities[Math.floor(Math.random() * baseActivities.length)];
        activities.push({
          day,
          time: i === 0 ? '9:00 AM' : i === 1 ? '1:00 PM' : '6:00 PM',
          activity,
          location: `${preferences.destination} - ${this.generateLocationName()}`,
          cost: Math.floor(Math.random() * 100) + 25,
          description: `Enjoy ${activity.toLowerCase()} in the heart of ${preferences.destination}`
        });
      }
    }

    return activities;
  }

  private generateAccommodations(preferences: any): Array<{name: string; type: string; pricePerNight: number; rating: number; amenities: string[]}> {
    const budgetPerNight = preferences.budget / preferences.duration * 0.4; // 40% of daily budget for accommodation
    
    return [{
      name: `${preferences.destination} Grand Hotel`,
      type: budgetPerNight > 200 ? 'Luxury Hotel' : budgetPerNight > 100 ? 'Mid-range Hotel' : 'Budget Hotel',
      pricePerNight: Math.min(budgetPerNight, 150 + Math.random() * 100),
      rating: 4.2 + Math.random() * 0.6,
      amenities: ['WiFi', 'Breakfast', 'Gym', 'Pool', '24/7 Front Desk']
    }];
  }

  private generateTransportation(preferences: any): Array<{type: string; from: string; to: string; cost: number; duration: string}> {
    return [
      {
        type: 'Flight',
        from: 'Home Airport',
        to: `${preferences.destination} Airport`,
        cost: Math.floor(preferences.budget * 0.3),
        duration: '4h 30m'
      },
      {
        type: 'Local Transport',
        from: 'Airport',
        to: 'Hotel',
        cost: 25,
        duration: '45m'
      }
    ];
  }

  private calculateTotalCost(activities: any[], accommodations: any[], transportation: any[]): number {
    const activityCost = activities.reduce((sum, activity) => sum + activity.cost, 0);
    const accommodationCost = accommodations.reduce((sum, acc) => sum + (acc.pricePerNight * 7), 0); // Assume 7 nights
    const transportationCost = transportation.reduce((sum, trans) => sum + trans.cost, 0);
    
    return activityCost + accommodationCost + transportationCost;
  }

  private generateTravelTips(destination: string): string[] {
    return [
      `Learn basic local phrases for ${destination}`,
      'Pack according to local weather conditions',
      'Keep copies of important documents',
      'Research local customs and etiquette',
      'Download offline maps and translation apps'
    ];
  }

  private generateAccommodationName(type: string, destination: string): string {
    const prefixes = {
      hotel: ['Grand', 'Royal', 'Plaza', 'Boutique'],
      resort: ['Paradise', 'Tropical', 'Ocean View', 'Sunset'],
      airbnb: ['Cozy', 'Modern', 'Charming', 'Stylish'],
      hostel: ['Backpacker', 'Traveler', 'Social', 'Urban'],
      apartment: ['Downtown', 'City Center', 'Local', 'Executive']
    };

    const prefix = prefixes[type as keyof typeof prefixes][Math.floor(Math.random() * 4)];
    return `${prefix} ${destination} ${type.charAt(0).toUpperCase() + type.slice(1)}`;
  }

  private generatePriceRange(type: string, budget: number): string {
    const ranges = {
      hotel: '$100-250/night',
      resort: '$200-500/night',
      airbnb: '$60-150/night',
      hostel: '$25-60/night',
      apartment: '$80-200/night'
    };
    return ranges[type as keyof typeof ranges];
  }

  private generateLocationDescription(destination: string): string {
    const descriptions = [
      `Heart of ${destination}`,
      `${destination} City Center`,
      `Historic District, ${destination}`,
      `Waterfront Area, ${destination}`,
      `Cultural Quarter, ${destination}`
    ];
    return descriptions[Math.floor(Math.random() * descriptions.length)];
  }

  private generateAmenities(type: string): string[] {
    const commonAmenities = ['WiFi', 'Air Conditioning', '24/7 Support'];
    const typeSpecific = {
      hotel: ['Room Service', 'Concierge', 'Business Center'],
      resort: ['Pool', 'Spa', 'Beach Access', 'Restaurant'],
      airbnb: ['Kitchen', 'Living Room', 'Local Host'],
      hostel: ['Shared Kitchen', 'Common Room', 'Laundry'],
      apartment: ['Full Kitchen', 'Washing Machine', 'Workspace']
    };

    return [...commonAmenities, ...typeSpecific[type as keyof typeof typeSpecific]];
  }

  private generateHighlights(type: string): string[] {
    return [
      'Excellent location near attractions',
      'Recently renovated facilities',
      'Outstanding customer service',
      'Great value for money'
    ];
  }

  private generateSuitability(type: string): string[] {
    const suitability = {
      hotel: ['Business travelers', 'Couples', 'Luxury seekers'],
      resort: ['Families', 'Honeymooners', 'Relaxation seekers'],
      airbnb: ['Families', 'Long-term stays', 'Local experience seekers'],
      hostel: ['Budget travelers', 'Solo travelers', 'Young adventurers'],
      apartment: ['Families', 'Business travelers', 'Extended stays']
    };
    return suitability[type as keyof typeof suitability];
  }

  private generateBookingTips(type: string): string[] {
    return [
      'Book in advance for better rates',
      'Check cancellation policies',
      'Read recent guest reviews',
      'Compare prices across platforms'
    ];
  }

  private determineBestTime(destination: string): string {
    const seasons = ['Spring (March-May)', 'Summer (June-August)', 'Fall (September-November)', 'Winter (December-February)'];
    return seasons[Math.floor(Math.random() * seasons.length)];
  }

  private generateWeatherInfo(destination: string): Array<{season: string; temperature: string; description: string}> {
    return [
      {season: 'Spring', temperature: '15-25°C', description: 'Mild and pleasant weather'},
      {season: 'Summer', temperature: '25-35°C', description: 'Warm with occasional rain'},
      {season: 'Fall', temperature: '10-20°C', description: 'Cool and comfortable'},
      {season: 'Winter', temperature: '0-10°C', description: 'Cold with possible snow'}
    ];
  }

  private generateCulturalInfo(destination: string) {
    return {
      language: 'Local Language',
      currency: 'Local Currency (LC)',
      customs: [
        'Respect local traditions',
        'Dress appropriately for religious sites',
        'Remove shoes when entering homes'
      ],
      etiquette: [
        'Greet with local customs',
        'Tip according to local standards',
        'Be punctual for appointments'
      ]
    };
  }

  private generateMustSeeAttractions(destination: string): Array<{name: string; description: string; cost: string}> {
    return [
      {name: `${destination} Cathedral`, description: 'Historic religious architecture', cost: '$10'},
      {name: 'Central Market', description: 'Local crafts and food', cost: 'Free'},
      {name: 'Art Museum', description: 'Local and international art', cost: '$15'},
      {name: 'Scenic Overlook', description: 'Panoramic city views', cost: '$5'}
    ];
  }

  private generateHiddenGems(destination: string): Array<{name: string; description: string; tip: string}> {
    return [
      {name: 'Local Café District', description: 'Authentic coffee culture', tip: 'Visit early morning for best selection'},
      {name: 'Artisan Quarter', description: 'Local craftspeople at work', tip: 'Weekends have more artisans present'},
      {name: 'Sunset Point', description: 'Secret viewpoint locals love', tip: 'Ask locals for exact directions'}
    ];
  }

  private generateFoodRecommendations(destination: string): Array<{dish: string; where: string; price: string}> {
    return [
      {dish: 'Traditional Local Dish', where: 'Family-run restaurant', price: '$12-18'},
      {dish: 'Street Food Special', where: 'Market stalls', price: '$3-6'},
      {dish: 'Regional Dessert', where: 'Local bakery', price: '$4-8'},
      {dish: 'Seafood Specialty', where: 'Waterfront restaurant', price: '$20-35'}
    ];
  }

  private generateTransportationInfo(destination: string) {
    return {
      options: ['Metro/Subway', 'Bus', 'Taxi', 'Bike Rental', 'Walking'],
      costs: ['$2-5', '$1-3', '$10-20', '$15/day', 'Free'],
      tips: [
        'Buy daily/weekly transport passes for savings',
        'Download local transport apps',
        'Walking is often the best way to explore'
      ]
    };
  }

  private generateSafetyInfo(destination: string): {rating: 'low' | 'medium' | 'high'; tips: string[]; emergencyNumbers: string[]} {
    return {
      rating: 'medium',
      tips: [
        'Keep valuables secure',
        'Stay aware of surroundings',
        'Use reputable transport services',
        'Keep emergency contacts handy'
      ],
      emergencyNumbers: ['Police: 911', 'Medical: 112', 'Tourist Hotline: 1-800-HELP']
    };
  }

  private generateBudgetTips(destination: string): string[] {
    return [
      'Eat at local markets for authentic, affordable meals',
      'Use public transportation instead of taxis',
      'Look for free walking tours and museum days',
      'Stay in neighborhoods slightly outside city center',
      'Book activities in advance for discounts'
    ];
  }

  private generateLocationName(): string {
    const locations = ['Downtown', 'Old Town', 'Riverside', 'Historic District', 'Cultural Center'];
    return locations[Math.floor(Math.random() * locations.length)];
  }

  private generateWeatherCondition(): string {
    const conditions = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Light Rain', 'Clear'];
    return conditions[Math.floor(Math.random() * conditions.length)];
  }

  private generateWeatherRecommendation(): string {
    const recommendations = [
      'Perfect day for outdoor activities',
      'Great weather for sightseeing',
      'Consider indoor activities',
      'Pack an umbrella',
      'Ideal conditions for walking tours'
    ];
    return recommendations[Math.floor(Math.random() * recommendations.length)];
  }
}

export const travelService = new TravelService();