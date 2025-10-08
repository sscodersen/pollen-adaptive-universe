const { Pool, neonConfig } = require('@neondatabase/serverless');
const { drizzle } = require('drizzle-orm/neon-serverless');
const ws = require("ws");

neonConfig.webSocketConstructor = ws;

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL must be set. Did you forget to provision a database?",
  );
}

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

let schema;
(async () => {
  schema = await import('../shared/schema.js');
})();

const db = drizzle({ client: pool, schema: {} });

module.exports = { pool, db, getSchema: async () => {
  if (!schema) {
    schema = await import('../shared/schema.js');
  }
  return schema;
}};
