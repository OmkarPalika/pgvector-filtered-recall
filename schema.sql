CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS items;

-- bucket is uniform over 0..999, so `WHERE bucket < N` selects exactly N/1000
-- of the table. One column, every selectivity level, no reload.
-- bucket_corr ranks rows by distance to a fixed anchor vector, so `bucket_corr < N`
-- selects the N/1000 rows nearest that anchor: a compact region of embedding space
-- rather than a scatter. Same exact selectivity as `bucket`, opposite spatial layout.
CREATE TABLE items (
  id          int PRIMARY KEY,
  bucket      int NOT NULL,
  bucket_corr int NOT NULL,
  embedding   vector(128) NOT NULL
);
