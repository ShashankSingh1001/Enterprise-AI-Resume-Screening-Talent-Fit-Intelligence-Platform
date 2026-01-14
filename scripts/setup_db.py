"""
Database Setup Script
Creates database schema and initial tables
"""

import os
import sys
from pathlib import Path
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv


def get_db_connection():
    """Get database connection from environment variables"""
    
    load_dotenv()
    
    db_config = {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': os.getenv('DATABASE_PORT', '5432'),
        'database': os.getenv('DATABASE_NAME', 'resume_ai'),
        'user': os.getenv('DATABASE_USER', 'postgres'),
        'password': os.getenv('DATABASE_PASSWORD', 'postgres')
    }
    
    return db_config


def create_database():
    """Create the main database if it doesn't exist"""
    
    print("\n" + "="*60)
    print("Creating Database")
    print("="*60)
    
    db_config = get_db_connection()
    
    try:
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database='postgres',
            user=db_config['user'],
            password=db_config['password']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_config['database'],)
        )
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(db_config['database'])
                )
            )
            print(f"Created database: {db_config['database']}")
        else:
            print(f"Database already exists: {db_config['database']}")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"Error creating database: {e}")
        return False


def create_tables():
    """Create database tables"""
    
    print("\n" + "="*60)
    print("Creating Database Tables")
    print("="*60)
    
    db_config = get_db_connection()
    
    schema = """
    CREATE TABLE IF NOT EXISTS resumes (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255),
        resume_text TEXT NOT NULL,
        parsed_data JSONB,
        file_path VARCHAR(512),
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS job_descriptions (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255),
        jd_text TEXT NOT NULL,
        parsed_data JSONB,
        requirements JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        resume_id INTEGER REFERENCES resumes(id) ON DELETE CASCADE,
        jd_id INTEGER REFERENCES job_descriptions(id) ON DELETE CASCADE,
        similarity_score FLOAT,
        fit_probability FLOAT,
        predicted_label BOOLEAN,
        model_version VARCHAR(50),
        features JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS explanations (
        id SERIAL PRIMARY KEY,
        prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
        method VARCHAR(50),
        explanation_data JSONB,
        feature_importance JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS bias_audit_logs (
        id SERIAL PRIMARY KEY,
        audit_type VARCHAR(100),
        sensitive_feature VARCHAR(100),
        metrics JSONB,
        dataset_size INTEGER,
        passed BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        hashed_password VARCHAR(255) NOT NULL,
        role VARCHAR(50) DEFAULT 'user',
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS model_registry (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(255) NOT NULL,
        model_version VARCHAR(50) NOT NULL,
        model_path VARCHAR(512),
        mlflow_run_id VARCHAR(255),
        metrics JSONB,
        parameters JSONB,
        status VARCHAR(50) DEFAULT 'staging',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(model_name, model_version)
    );
    
    CREATE INDEX IF NOT EXISTS idx_resumes_uploaded_at ON resumes(uploaded_at);
    CREATE INDEX IF NOT EXISTS idx_predictions_resume_id ON predictions(resume_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_jd_id ON predictions(jd_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
    CREATE INDEX IF NOT EXISTS idx_bias_audit_created_at ON bias_audit_logs(created_at);
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(schema)
        conn.commit()
        
        print("All tables created successfully")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"Error creating tables: {e}")
        return False


def verify_database():
    """Verify database connection and tables"""
    
    print("\n" + "="*60)
    print("Verifying Database Setup")
    print("="*60)
    
    db_config = get_db_connection()
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print("Connected to PostgreSQL")
        print(f"Version: {version[0][:50]}...")
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        table_count = cursor.fetchone()[0]
        print(f"Found {table_count} tables")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"Database verification failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("Resume AI - Database Setup")
    print("="*60)
    
    if not create_database():
        return False
    
    if not create_tables():
        return False
    
    if not verify_database():
        return False
    
    print("\nDatabase Setup Complete")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
