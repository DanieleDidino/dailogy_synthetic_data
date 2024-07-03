-- First version: embedding as "TEXT"
-- CREATE TABLE examples (
--     id SERIAL PRIMARY KEY,
--     dysfunctional TEXT NOT NULL,
--     embedding TEXT NOT NULL,
--     functional TEXT NOT NULL
-- );

-- Embedding as "BLOB" (Binary Large Object)
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dysfunctional TEXT,
    embedding BLOB,
    functional TEXT
)
