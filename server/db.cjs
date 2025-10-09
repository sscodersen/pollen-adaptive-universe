const { Pool, neonConfig } = require('@neondatabase/serverless');
const { drizzle } = require('drizzle-orm/neon-serverless');
const ws = require("ws");

neonConfig.webSocketConstructor = ws;

if (!process.env.DATABASE_URL) {
  console.warn("DATABASE_URL not set - database functionality will be limited");
  module.exports = { 
    pool: null, 
    db: null,
    getSchema: async () => ({})
  };
} else {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL });
  const db = drizzle({ client: pool });

  module.exports = { 
    pool, 
    db,
    getSchema: async () => {
      // Import schema dynamically - this will be compiled by the build process
      try {
        const schema = await import('../shared/schema.js');
        return schema;
      } catch (error) {
        console.warn("Could not load schema:", error.message);
        return {};
      }
    }
  };
}
