-- One-time setup. Run as a Postgres superuser (e.g. the "postgres" user).
-- Change both passwords first.

-- ---- STEP 1: run while connected to your default database ------------------
CREATE DATABASE ai_data_analyst;

CREATE ROLE app_ingest LOGIN PASSWORD 'Onionsmak3youCrY1$1' CREATEROLE;
CREATE ROLE app_query  LOGIN PASSWORD 'Onionsmak3youCrY1$1'  NOINHERIT;

-- ---- STEP 2: reconnect to ai_data_analyst (psql: \c ai_data_analyst), then run

-- The ingest role may create schemas (one per user) and per-user roles.
GRANT CREATE  ON DATABASE ai_data_analyst TO app_ingest;
GRANT CONNECT ON DATABASE ai_data_analyst TO app_query;

-- Nobody gets anything by default.
REVOKE ALL ON SCHEMA public FROM PUBLIC;

-- Safety net: the query login can never write, even if app code has a bug.
ALTER ROLE app_query SET default_transaction_read_only = on;

-- How isolation works:
--   * app_query has NO privileges of its own (NOINHERIT).
--   * For each user, the ingest code creates a NOLOGIN role "user_<id>_ro" that
--     can only SELECT from schema "user_<id>", and grants it to app_query.
--   * At query time the app runs "SET LOCAL ROLE user_<id>_ro", so Postgres
--     itself blocks access to every other user's data.
