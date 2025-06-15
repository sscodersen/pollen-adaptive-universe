
export interface Product {
  id: string;
  name: string;
  description: string;
  price: string;
  originalPrice?: string;
  rating: number;
  reviews: number;
  category: string;
  tags: string[];
  significance: number;
  trending: boolean;
  link: string;
  seller: string;
  discount?: number;
  features: string[];
  inStock: boolean;
}
