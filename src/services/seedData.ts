import { storageService } from './storageService';

export interface SeedDataConfig {
  communities: number;
  feedItems: number;
  healthInsights: number;
  forumTopics: number;
  opportunities: number;
}

class SeedDataService {
  private static instance: SeedDataService;
  private isSeeded = false;

  static getInstance(): SeedDataService {
    if (!SeedDataService.instance) {
      SeedDataService.instance = new SeedDataService();
    }
    return SeedDataService.instance;
  }

  async seedAllData(config?: Partial<SeedDataConfig>): Promise<void> {
    const existingSeed = localStorage.getItem('demo_data_seeded');
    if (existingSeed && !config) {
      console.log('✅ Demo data already seeded');
      this.isSeeded = true;
      return;
    }

    console.log('🌱 Seeding demo data...');

    const defaultConfig: SeedDataConfig = {
      communities: 8,
      feedItems: 20,
      healthInsights: 10,
      forumTopics: 12,
      opportunities: 15,
      ...config
    };

    await this.seedCommunities(defaultConfig.communities);
    await this.seedFeedContent(defaultConfig.feedItems);
    await this.seedHealthResearch(defaultConfig.healthInsights);
    await this.seedForumTopics(defaultConfig.forumTopics);
    await this.seedOpportunities(defaultConfig.opportunities);

    localStorage.setItem('demo_data_seeded', new Date().toISOString());
    this.isSeeded = true;
    console.log('✅ Demo data seeding complete!');
  }

  private async seedCommunities(count: number): Promise<void> {
    const communities = [
      {
        id: 'comm_wellness_001',
        name: 'Mental Wellness Warriors',
        description: 'A supportive space for mental health journey sharing and peer support',
        type: 'support_group',
        category: 'wellness',
        memberCount: 1247,
        posts: [
          { content: 'Started meditation today. Small steps matter! 🧘‍♀️', likes: 34, replies: 8 },
          { content: 'Anyone have tips for managing anxiety? Looking for natural remedies.', likes: 28, replies: 15 },
          { content: 'Celebrating 30 days of daily journaling! It really helps process emotions.', likes: 56, replies: 12 }
        ]
      },
      {
        id: 'comm_farming_002',
        name: 'Smart Farming Innovators',
        description: 'Modern agriculture techniques, sustainable farming, and crop optimization',
        type: 'interest_group',
        category: 'agriculture',
        memberCount: 892,
        posts: [
          { content: 'Tried vertical farming in my greenhouse. 40% increase in yield! 🌱', likes: 67, replies: 21 },
          { content: 'Looking for advice on drip irrigation systems for small-scale farms', likes: 45, replies: 18 },
          { content: 'AI-powered pest detection saved my crop this season. Technology is amazing!', likes: 89, replies: 14 }
        ]
      },
      {
        id: 'comm_social_003',
        name: 'Community Change Makers',
        description: 'Grassroots initiatives, social impact projects, and community development',
        type: 'interest_group',
        category: 'social_impact',
        memberCount: 2156,
        posts: [
          { content: 'Our local food bank project just hit 10,000 meals served! 🎉', likes: 234, replies: 45 },
          { content: 'Starting a coding bootcamp for underserved youth. Any mentors interested?', likes: 156, replies: 67 },
          { content: 'Clean water initiative update: 5 wells completed, 3 more in progress', likes: 198, replies: 34 }
        ]
      },
      {
        id: 'comm_fitness_004',
        name: 'Fitness & Movement Hub',
        description: 'Share workout routines, fitness goals, and celebrate progress together',
        type: 'support_group',
        category: 'wellness',
        memberCount: 3421,
        posts: [
          { content: 'Hit my first 5K milestone today! Never thought I could do it 🏃‍♂️', likes: 143, replies: 29 },
          { content: 'What are your favorite home workout routines? Gym is too expensive', likes: 87, replies: 42 },
          { content: 'Progress pic: Down 30 lbs in 6 months with consistent effort!', likes: 456, replies: 78 }
        ]
      },
      {
        id: 'comm_careers_005',
        name: 'Career Growth & Opportunities',
        description: 'Job hunting tips, career advice, skill development, and networking',
        type: 'topic_based',
        category: 'opportunities',
        memberCount: 5678,
        posts: [
          { content: 'Just landed my first remote job! Persistence pays off 💼', likes: 234, replies: 56 },
          { content: 'Free certification courses that actually helped me get hired (thread)', likes: 567, replies: 123 },
          { content: 'How to negotiate salary when you have no experience?', likes: 189, replies: 87 }
        ]
      },
      {
        id: 'comm_nutrition_006',
        name: 'Mindful Nutrition Circle',
        description: 'Healthy eating, meal planning, and nutritional science discussions',
        type: 'support_group',
        category: 'wellness',
        memberCount: 1834,
        posts: [
          { content: 'Meal prep Sunday! Sharing my budget-friendly healthy recipes 🥗', likes: 298, replies: 67 },
          { content: 'Anyone else struggling with late-night snacking? Tips needed!', likes: 156, replies: 45 },
          { content: 'Plant-based diet month 3: Energy levels through the roof!', likes: 234, replies: 78 }
        ]
      },
      {
        id: 'comm_climate_007',
        name: 'Climate Action Network',
        description: 'Environmental activism, sustainable living, and climate solutions',
        type: 'interest_group',
        category: 'social_impact',
        memberCount: 4123,
        posts: [
          { content: 'Our community just planted 1,000 trees! 🌳 Next goal: 5,000', likes: 789, replies: 145 },
          { content: 'Zero-waste living tips that actually work (and save money)', likes: 456, replies: 98 },
          { content: 'Started a local composting initiative. 20 families joined already!', likes: 234, replies: 56 }
        ]
      },
      {
        id: 'comm_learning_008',
        name: 'Lifelong Learners',
        description: 'Self-education, online courses, skill-sharing, and knowledge exchange',
        type: 'topic_based',
        category: 'opportunities',
        memberCount: 6234,
        posts: [
          { content: 'Completed my 50th online course this year! Learning is addictive 📚', likes: 345, replies: 67 },
          { content: 'Best free resources for learning data science? Share your favorites!', likes: 567, replies: 189 },
          { content: 'Started teaching Python to my neighbors. Community education matters!', likes: 234, replies: 45 }
        ]
      }
    ];

    const existingCommunities = storageService.getItem('demo_communities', []);
    if (existingCommunities.length === 0) {
      storageService.setItem('demo_communities', communities.slice(0, count));
      console.log(`✅ Seeded ${Math.min(count, communities.length)} communities`);
    }
  }

  private async seedFeedContent(count: number): Promise<void> {
    const feedItems = [
      {
        id: 'feed_wellness_001',
        type: 'wellness_tip',
        title: '5-Minute Breathing Exercise',
        content: 'Try box breathing: Inhale for 4 counts, hold for 4, exhale for 4, hold for 4. Repeat 5 times to reduce stress instantly.',
        category: 'mental_health',
        likes: 892,
        views: 5643
      },
      {
        id: 'feed_social_001',
        type: 'social_impact',
        title: 'Local Food Bank Needs Volunteers',
        content: 'Help serve 500+ families this weekend. Every hour counts! Sign up link in bio.',
        category: 'volunteer',
        likes: 456,
        views: 3421,
        aiScore: 9.2
      },
      {
        id: 'feed_opportunity_001',
        type: 'opportunity',
        title: 'Remote Junior Developer Position',
        content: 'Tech startup hiring entry-level developers. No degree required, just passion and portfolio. $60k-75k/year.',
        category: 'career',
        likes: 1234,
        views: 8976,
        verified: true
      },
      {
        id: 'feed_wellness_002',
        type: 'wellness_tip',
        title: 'Sleep Better Tonight',
        content: 'Blue light exposure before bed disrupts melatonin. Use night mode 2 hours before sleep for better rest.',
        category: 'sleep',
        likes: 567,
        views: 4123
      },
      {
        id: 'feed_farming_001',
        type: 'agriculture',
        title: 'Smart Irrigation Saves 40% Water',
        content: 'New AI-powered drip system monitors soil moisture in real-time. Perfect for small farms.',
        category: 'technology',
        likes: 234,
        views: 1876
      },
      {
        id: 'feed_opportunity_002',
        type: 'opportunity',
        title: 'Free Coding Bootcamp Scholarships',
        content: '50 full scholarships available for underrepresented groups. Applications open until next month.',
        category: 'education',
        likes: 2341,
        views: 12456,
        verified: true
      },
      {
        id: 'feed_social_002',
        type: 'social_impact',
        title: 'Community Solar Project Launched',
        content: 'Bring clean energy to 100 homes. Crowdfunding goal: $50k. Currently at $23k. Join the movement!',
        category: 'environment',
        likes: 789,
        views: 5432,
        aiScore: 8.7,
        fundingProgress: 46
      },
      {
        id: 'feed_wellness_003',
        type: 'wellness_tip',
        title: 'Desk Stretches for Remote Workers',
        content: 'Every hour: neck rolls (10x), shoulder shrugs (15x), wrist circles (20x). Prevent pain, boost energy!',
        category: 'fitness',
        likes: 445,
        views: 3210
      },
      {
        id: 'feed_opportunity_003',
        type: 'opportunity',
        title: 'Micro-Grant for Local Entrepreneurs',
        content: '$5,000 grants for community-focused businesses. No credit check, just a solid business plan.',
        category: 'business',
        likes: 892,
        views: 6543,
        verified: true
      },
      {
        id: 'feed_wellness_004',
        type: 'wellness_tip',
        title: 'Hydration Reminder',
        content: 'Drink water before you feel thirsty. Aim for half your body weight in ounces daily. Track it!',
        category: 'nutrition',
        likes: 321,
        views: 2145
      },
      {
        id: 'feed_social_003',
        type: 'social_impact',
        title: 'Youth Mentorship Program Expanding',
        content: '200 high school students need career mentors. 1 hour/week can change a life. Virtual options available.',
        category: 'education',
        likes: 567,
        views: 4321,
        aiScore: 9.0
      },
      {
        id: 'feed_farming_002',
        type: 'agriculture',
        title: 'Companion Planting Guide',
        content: 'Tomatoes + Basil = Better growth & pest control. Carrots + Onions = Enhanced flavor. Nature knows best!',
        category: 'sustainable',
        likes: 178,
        views: 1432
      },
      {
        id: 'feed_opportunity_004',
        type: 'opportunity',
        title: 'UX Design Apprenticeship',
        content: 'Learn by doing. 6-month paid apprenticeship at design agency. Portfolio review starts next week.',
        category: 'career',
        likes: 1567,
        views: 9876,
        verified: true
      },
      {
        id: 'feed_wellness_005',
        type: 'wellness_tip',
        title: 'Gratitude Journaling Works',
        content: 'Write 3 things you\'re grateful for each morning. Studies show it reduces depression by 35%.',
        category: 'mental_health',
        likes: 892,
        views: 5234
      },
      {
        id: 'feed_social_004',
        type: 'social_impact',
        title: 'Clean Water Initiative Update',
        content: '8 wells completed serving 5,000 people. Goal: 20 wells by year-end. Every donation counts!',
        category: 'health',
        likes: 1234,
        views: 7654,
        aiScore: 9.5,
        fundingProgress: 40
      }
    ];

    const existingFeed = storageService.getItem('demo_feed_items', []);
    if (existingFeed.length === 0) {
      storageService.setItem('demo_feed_items', feedItems.slice(0, count));
      console.log(`✅ Seeded ${Math.min(count, feedItems.length)} feed items`);
    }
  }

  private async seedHealthResearch(count: number): Promise<void> {
    const healthInsights = [
      {
        id: 'health_insight_001',
        type: 'trend',
        title: 'Morning Exercise Linked to Better Sleep Quality',
        description: 'Analysis of 5,000+ data points shows 30 minutes of morning exercise improves sleep duration by average 47 minutes',
        dataPoints: 5234,
        confidence: 0.89,
        category: 'fitness',
        visualData: { avgImprovement: 47, participants: 5234 }
      },
      {
        id: 'health_insight_002',
        type: 'correlation',
        title: 'Mediterranean Diet Reduces Inflammation Markers',
        description: 'Users following Mediterranean diet showed 34% reduction in C-reactive protein levels over 12 weeks',
        dataPoints: 3456,
        confidence: 0.92,
        category: 'nutrition',
        visualData: { reduction: 34, duration: 12 }
      },
      {
        id: 'health_insight_003',
        type: 'recommendation',
        title: 'Optimal Sleep Window Identified',
        description: 'Data suggests sleeping between 10 PM - 6 AM correlates with 28% better cognitive performance',
        dataPoints: 8765,
        confidence: 0.85,
        category: 'sleep',
        visualData: { improvement: 28, optimalStart: 22, optimalEnd: 6 }
      },
      {
        id: 'health_insight_004',
        type: 'breakthrough',
        title: 'Mindfulness Meditation Shows Anxiety Reduction',
        description: 'Just 10 minutes daily of mindfulness practice reduced reported anxiety scores by average 42%',
        dataPoints: 4321,
        confidence: 0.88,
        category: 'mental_health',
        visualData: { reduction: 42, minDuration: 10 }
      },
      {
        id: 'health_insight_005',
        type: 'trend',
        title: 'Plant-Based Protein Gains Popularity',
        description: 'Users switching to plant proteins report 25% increase in energy levels and improved digestion',
        dataPoints: 2987,
        confidence: 0.81,
        category: 'nutrition',
        visualData: { energyIncrease: 25, satisfactionRate: 87 }
      },
      {
        id: 'health_insight_006',
        type: 'correlation',
        title: 'Hydration Impact on Cognitive Function',
        description: 'Proper hydration (8+ glasses/day) associated with 19% better focus and memory retention',
        dataPoints: 6543,
        confidence: 0.86,
        category: 'wellness',
        visualData: { improvement: 19, minGlasses: 8 }
      },
      {
        id: 'health_insight_007',
        type: 'recommendation',
        title: 'Strength Training Frequency Optimization',
        description: 'Data shows 3x/week strength training optimal for muscle growth without overtraining',
        dataPoints: 5678,
        confidence: 0.90,
        category: 'fitness',
        visualData: { optimalFrequency: 3, muscleGain: 12 }
      },
      {
        id: 'health_insight_008',
        type: 'trend',
        title: 'Digital Detox Benefits Confirmed',
        description: '1-hour screen-free before bed improves sleep onset time by average 23 minutes',
        dataPoints: 7890,
        confidence: 0.87,
        category: 'sleep',
        visualData: { improvement: 23, recommendation: 1 }
      },
      {
        id: 'health_insight_009',
        type: 'breakthrough',
        title: 'Social Connection Boosts Mental Health',
        description: 'Regular social interaction (3+ times/week) reduces depression symptoms by 38%',
        dataPoints: 4567,
        confidence: 0.91,
        category: 'mental_health',
        visualData: { reduction: 38, minInteractions: 3 }
      },
      {
        id: 'health_insight_010',
        type: 'correlation',
        title: 'Outdoor Time Enhances Mood',
        description: '20+ minutes of outdoor activity daily linked to 31% improvement in mood scores',
        dataPoints: 6789,
        confidence: 0.84,
        category: 'wellness',
        visualData: { improvement: 31, minMinutes: 20 }
      }
    ];

    const existingInsights = storageService.getItem('demo_health_insights', []);
    if (existingInsights.length === 0) {
      storageService.setItem('demo_health_insights', healthInsights.slice(0, count));
      console.log(`✅ Seeded ${Math.min(count, healthInsights.length)} health insights`);
    }
  }

  private async seedForumTopics(count: number): Promise<void> {
    const forumTopics = [
      {
        id: 'forum_topic_001',
        title: 'How do we ensure AI systems are truly unbiased?',
        category: 'ai_bias',
        description: 'Discussion on techniques and frameworks for detecting and mitigating bias in AI decision-making',
        posts: 45,
        views: 1234,
        upvotes: 89,
        replies: [
          { content: 'Regular audits with diverse datasets are crucial. We need representation in training data.', upvotes: 34 },
          { content: 'Explainable AI helps identify where bias creeps in. Transparency is key.', upvotes: 28 }
        ]
      },
      {
        id: 'forum_topic_002',
        title: 'Privacy vs Personalization: Finding the Balance',
        category: 'privacy',
        description: 'How can we offer personalized experiences while respecting user privacy?',
        posts: 67,
        views: 2345,
        upvotes: 123,
        replies: [
          { content: 'Federated learning allows personalization without centralizing data. Win-win!', upvotes: 56 },
          { content: 'Give users granular control. Let them choose their privacy-convenience tradeoff.', upvotes: 42 }
        ]
      },
      {
        id: 'forum_topic_003',
        title: 'AI Decision Transparency: What Should Be Disclosed?',
        category: 'transparency',
        description: 'Defining standards for AI decision explanations and user communication',
        posts: 38,
        views: 987,
        upvotes: 67,
        replies: [
          { content: 'Users deserve to know: What data was used? How was the decision made? What were alternatives?', upvotes: 45 },
          { content: 'Transparency reports should be standardized across the industry.', upvotes: 31 }
        ]
      },
      {
        id: 'forum_topic_004',
        title: 'Algorithmic Fairness in Hiring AI',
        category: 'fairness',
        description: 'Best practices for ensuring hiring algorithms don\'t perpetuate discrimination',
        posts: 56,
        views: 1876,
        upvotes: 98,
        replies: [
          { content: 'Blind recruitment with anonymized data helps, but AI can still find proxy variables.', upvotes: 52 },
          { content: 'Regular fairness audits across protected classes are non-negotiable.', upvotes: 38 }
        ]
      },
      {
        id: 'forum_topic_005',
        title: 'Who is Accountable When AI Makes Mistakes?',
        category: 'accountability',
        description: 'Exploring liability frameworks and responsibility chains for AI errors',
        posts: 72,
        views: 2567,
        upvotes: 134,
        replies: [
          { content: 'The deploying organization is ultimately responsible, not just the developers.', upvotes: 67 },
          { content: 'Need clear audit trails: who trained it, who approved it, who deployed it.', upvotes: 54 }
        ]
      },
      {
        id: 'forum_topic_006',
        title: 'AI Safety: Preventing Unintended Consequences',
        category: 'safety',
        description: 'Discussing fail-safes, testing protocols, and safety-by-design principles',
        posts: 43,
        views: 1432,
        upvotes: 78,
        replies: [
          { content: 'Red team testing should be mandatory before any AI system goes live.', upvotes: 46 },
          { content: 'Human-in-the-loop for critical decisions is a must.', upvotes: 35 }
        ]
      },
      {
        id: 'forum_topic_007',
        title: 'Ethical Guidelines for AI-Generated Content',
        category: 'transparency',
        description: 'Should all AI-generated content be labeled? How and when?',
        posts: 89,
        views: 3456,
        upvotes: 156,
        replies: [
          { content: 'Watermarking AI content is technically feasible and ethically necessary.', upvotes: 78 },
          { content: 'Users have a right to know when they\'re interacting with AI vs humans.', upvotes: 65 }
        ]
      },
      {
        id: 'forum_topic_008',
        title: 'Bias in Healthcare AI: Life or Death Consequences',
        category: 'ai_bias',
        description: 'Critical examination of bias in medical diagnosis and treatment recommendation AI',
        posts: 61,
        views: 2234,
        upvotes: 112,
        replies: [
          { content: 'Underrepresented groups in medical datasets lead to diagnostic failures. This is urgent.', upvotes: 89 },
          { content: 'FDA needs stricter AI validation requirements for medical devices.', upvotes: 71 }
        ]
      },
      {
        id: 'forum_topic_009',
        title: 'Data Ownership in the AI Era',
        category: 'privacy',
        description: 'Who owns the data AI learns from? What are user rights?',
        posts: 54,
        views: 1987,
        upvotes: 94,
        replies: [
          { content: 'GDPR right-to-deletion should extend to AI training data.', upvotes: 62 },
          { content: 'Users should be compensated when their data trains commercial AI.', upvotes: 48 }
        ]
      },
      {
        id: 'forum_topic_010',
        title: 'Environmental Impact of AI Training',
        category: 'accountability',
        description: 'Carbon footprint of large language models and sustainable AI practices',
        posts: 47,
        views: 1654,
        upvotes: 87,
        replies: [
          { content: 'Energy-efficient architectures and carbon-neutral data centers are achievable goals.', upvotes: 54 },
          { content: 'Publish carbon costs alongside model performance metrics.', upvotes: 41 }
        ]
      },
      {
        id: 'forum_topic_011',
        title: 'AI in Criminal Justice: Ethical Boundaries',
        category: 'fairness',
        description: 'Risk assessment tools, sentencing algorithms, and justice system AI',
        posts: 78,
        views: 2876,
        upvotes: 145,
        replies: [
          { content: 'Historical bias in arrest data makes predictive policing inherently unfair.', upvotes: 98 },
          { content: 'Human judges must retain final authority. AI should inform, not decide.', upvotes: 76 }
        ]
      },
      {
        id: 'forum_topic_012',
        title: 'Building Trust in AI Systems',
        category: 'transparency',
        description: 'Strategies for increasing public confidence in AI technology',
        posts: 52,
        views: 1765,
        upvotes: 102,
        replies: [
          { content: 'Open source models allow community scrutiny. Transparency builds trust.', upvotes: 67 },
          { content: 'Independent AI ethics boards should certify high-stakes systems.', upvotes: 53 }
        ]
      }
    ];

    const existingTopics = storageService.getItem('demo_forum_topics', []);
    if (existingTopics.length === 0) {
      storageService.setItem('demo_forum_topics', forumTopics.slice(0, count));
      console.log(`✅ Seeded ${Math.min(count, forumTopics.length)} forum topics`);
    }
  }

  private async seedOpportunities(count: number): Promise<void> {
    const opportunities = [
      {
        id: 'opp_001',
        title: 'Remote Software Engineer - Entry Level',
        company: 'TechForward Inc',
        type: 'job',
        location: 'Remote',
        salary: '$65k - $85k',
        description: 'Join a fast-growing startup building AI tools for education. No degree required.',
        tags: ['remote', 'entry-level', 'tech', 'startup'],
        verified: true,
        aiScore: 9.1
      },
      {
        id: 'opp_002',
        title: 'Free Full-Stack Development Bootcamp',
        company: 'CodePath',
        type: 'education',
        location: 'Online',
        salary: 'Free',
        description: '12-week intensive program. 95% job placement rate. Scholarships for underrepresented groups.',
        tags: ['free', 'bootcamp', 'coding', 'scholarship'],
        verified: true,
        aiScore: 9.5
      },
      {
        id: 'opp_003',
        title: 'Social Impact Micro-Grant',
        company: 'Community Foundation',
        type: 'grant',
        location: 'Local',
        salary: '$5,000',
        description: 'Funding for community projects. Focus on education, health, or environment.',
        tags: ['grant', 'social-impact', 'community', 'non-profit'],
        verified: true,
        aiScore: 8.8
      },
      {
        id: 'opp_004',
        title: 'UX Design Apprenticeship',
        company: 'Design Studio Co',
        type: 'apprenticeship',
        location: 'Hybrid',
        salary: '$45k + mentorship',
        description: 'Learn from senior designers. Build your portfolio. 6-month paid program.',
        tags: ['design', 'apprenticeship', 'paid', 'mentorship'],
        verified: true,
        aiScore: 8.7
      },
      {
        id: 'opp_005',
        title: 'Climate Tech Startup Equity Position',
        company: 'GreenFuture Labs',
        type: 'job',
        location: 'Remote',
        salary: '$70k + equity',
        description: 'Early-stage climate tech company. Help build carbon tracking software.',
        tags: ['climate', 'startup', 'equity', 'remote'],
        verified: true,
        aiScore: 8.9
      },
      {
        id: 'opp_006',
        title: 'Agricultural Innovation Grant',
        company: 'Farm Foundation',
        type: 'grant',
        location: 'Rural',
        salary: '$25,000',
        description: 'Support for sustainable farming innovations. Equipment and training included.',
        tags: ['agriculture', 'grant', 'innovation', 'sustainable'],
        verified: true,
        aiScore: 9.0
      },
      {
        id: 'opp_007',
        title: 'Data Science Certificate Program',
        company: 'Online University',
        type: 'education',
        location: 'Online',
        salary: '$299/month',
        description: 'Industry-recognized certification. Financial aid available. Job guarantee program.',
        tags: ['data-science', 'certificate', 'online', 'financial-aid'],
        verified: true,
        aiScore: 8.6
      },
      {
        id: 'opp_008',
        title: 'Community Health Worker Position',
        company: 'Local Health Dept',
        type: 'job',
        location: 'Local',
        salary: '$42k - $48k',
        description: 'Make a difference in public health. Training provided. Bilingual preferred.',
        tags: ['healthcare', 'community', 'public-service', 'local'],
        verified: true,
        aiScore: 8.5
      },
      {
        id: 'opp_009',
        title: 'Entrepreneur Fellowship Program',
        company: 'Innovation Hub',
        type: 'fellowship',
        location: 'Hybrid',
        salary: '$30k stipend + resources',
        description: '1-year fellowship for social entrepreneurs. Mentorship, funding, workspace included.',
        tags: ['entrepreneurship', 'fellowship', 'social-impact', 'funded'],
        verified: true,
        aiScore: 9.3
      },
      {
        id: 'opp_010',
        title: 'Renewable Energy Technician Training',
        company: 'Green Energy Institute',
        type: 'education',
        location: 'On-site',
        salary: 'Free + Job Placement',
        description: 'Solar & wind energy certification. 8-week program. 90% employment rate.',
        tags: ['renewable-energy', 'training', 'certification', 'job-placement'],
        verified: true,
        aiScore: 9.1
      },
      {
        id: 'opp_011',
        title: 'Content Creator Partnership',
        company: 'Media Collective',
        type: 'partnership',
        location: 'Remote',
        salary: 'Revenue share',
        description: 'Join creator network. Support with production, marketing, monetization.',
        tags: ['content', 'creative', 'partnership', 'remote'],
        verified: false,
        aiScore: 7.8
      },
      {
        id: 'opp_012',
        title: 'Affordable Housing Development Grant',
        company: 'Housing Authority',
        type: 'grant',
        location: 'Urban',
        salary: '$100,000',
        description: 'Funding for affordable housing projects. Community-focused development.',
        tags: ['housing', 'grant', 'development', 'urban'],
        verified: true,
        aiScore: 9.2
      },
      {
        id: 'opp_013',
        title: 'Cybersecurity Analyst - Junior',
        company: 'SecureNet Corp',
        type: 'job',
        location: 'Remote',
        salary: '$58k - $72k',
        description: 'Entry-level cybersecurity role. Training provided. Security+ certification bonus.',
        tags: ['cybersecurity', 'remote', 'entry-level', 'tech'],
        verified: true,
        aiScore: 8.8
      },
      {
        id: 'opp_014',
        title: 'Teaching Assistant Scholarship',
        company: 'Education Foundation',
        type: 'education',
        location: 'University',
        salary: 'Tuition + $18k/year',
        description: 'Graduate degree + teaching experience. Full tuition coverage plus stipend.',
        tags: ['education', 'scholarship', 'graduate', 'teaching'],
        verified: true,
        aiScore: 9.0
      },
      {
        id: 'opp_015',
        title: 'Mobile App Development Contract',
        company: 'Startup Incubator',
        type: 'contract',
        location: 'Remote',
        salary: '$8k - $15k/project',
        description: 'Build apps for early-stage startups. Flexible hours. Portfolio building.',
        tags: ['mobile', 'contract', 'development', 'remote'],
        verified: true,
        aiScore: 8.4
      }
    ];

    const existingOpps = storageService.getItem('demo_opportunities', []);
    if (existingOpps.length === 0) {
      storageService.setItem('demo_opportunities', opportunities.slice(0, count));
      console.log(`✅ Seeded ${Math.min(count, opportunities.length)} opportunities`);
    }
  }

  async clearAllData(): Promise<void> {
    localStorage.removeItem('demo_data_seeded');
    storageService.removeItem('demo_communities');
    storageService.removeItem('demo_feed_items');
    storageService.removeItem('demo_health_insights');
    storageService.removeItem('demo_forum_topics');
    storageService.removeItem('demo_opportunities');
    this.isSeeded = false;
    console.log('🗑️ Demo data cleared');
  }

  isDataSeeded(): boolean {
    return this.isSeeded || !!localStorage.getItem('demo_data_seeded');
  }
}

export const seedDataService = SeedDataService.getInstance();
