CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS items;

-- bucket is uniform over 0..999, so `WHERE bucket < N` selects exactly N/1000
-- of the table. One column, every selectivity level, no reload.
CREATE TABLE items (
  id        int PRIMARY KEY,
  bucket    int NOT NULL,
  embedding vector(128) NOT NULL
);
