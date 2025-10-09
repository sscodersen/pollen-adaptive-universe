interface WeatherData {
  temperature: number;
  condition: string;
  icon: string;
  location: string;
}

// Using Open-Meteo API (free, no API key required)
export const realWeatherService = {
  async getCurrentWeather(latitude?: number, longitude?: number): Promise<WeatherData> {
    try {
      // Default to a location if geolocation not provided
      const lat = latitude || 40.7128; // New York
      const lon = longitude || -74.0060;

      const response = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true&temperature_unit=celsius`
      );

      if (!response.ok) {
        throw new Error('Weather API request failed');
      }

      const data = await response.json();
      const currentWeather = data.current_weather;

      // Map weather codes to conditions
      const weatherConditions: Record<number, { condition: string; icon: string }> = {
        0: { condition: 'Clear', icon: '☀️' },
        1: { condition: 'Mainly Clear', icon: '🌤️' },
        2: { condition: 'Partly Cloudy', icon: '⛅' },
        3: { condition: 'Overcast', icon: '☁️' },
        45: { condition: 'Foggy', icon: '🌫️' },
        48: { condition: 'Foggy', icon: '🌫️' },
        51: { condition: 'Light Drizzle', icon: '🌦️' },
        53: { condition: 'Moderate Drizzle', icon: '🌧️' },
        55: { condition: 'Dense Drizzle', icon: '🌧️' },
        61: { condition: 'Light Rain', icon: '🌧️' },
        63: { condition: 'Moderate Rain', icon: '🌧️' },
        65: { condition: 'Heavy Rain', icon: '🌧️' },
        71: { condition: 'Light Snow', icon: '🌨️' },
        73: { condition: 'Moderate Snow', icon: '🌨️' },
        75: { condition: 'Heavy Snow', icon: '❄️' },
        95: { condition: 'Thunderstorm', icon: '⛈️' },
      };

      const weatherInfo = weatherConditions[currentWeather.weathercode as number] || {
        condition: 'Unknown',
        icon: '🌡️'
      };

      return {
        temperature: Math.round(currentWeather.temperature),
        condition: weatherInfo.condition,
        icon: weatherInfo.icon,
        location: 'Current Location'
      };
    } catch (error) {
      console.error('Failed to fetch weather:', error);
      // Return fallback data
      return {
        temperature: 11,
        condition: 'Clear',
        icon: '☀️',
        location: 'Unknown'
      };
    }
  },

  async getWeatherByCity(city: string): Promise<WeatherData> {
    try {
      // Geocoding API to get coordinates from city name
      const geocodeResponse = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`
      );
      
      const geocodeData = await geocodeResponse.json();
      
      if (!geocodeData.results || geocodeData.results.length === 0) {
        throw new Error('City not found');
      }

      const { latitude, longitude, name } = geocodeData.results[0];
      const weather = await this.getCurrentWeather(latitude, longitude);
      
      return {
        ...weather,
        location: name
      };
    } catch (error) {
      console.error('Failed to fetch weather for city:', error);
      return this.getCurrentWeather();
    }
  }
};
