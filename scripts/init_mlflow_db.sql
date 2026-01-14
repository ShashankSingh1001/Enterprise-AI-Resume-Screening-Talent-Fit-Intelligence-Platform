-- ========================================
-- MLflow Database Initialization Script
-- Automatically creates mlflow database
-- Runs on PostgreSQL container first startup
-- ========================================

-- Create mlflow database if it doesn't exist
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;