"""
Setup Phase 5 Feature Store Database
Creates database, tables, indexes, views
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
def setup_phase5_database():
    print("="*60)
    print("PHASE 5: DATABASE SETUP")
    print("="*60)
    
    # Step 1: Connect to default 'postgres' database
    print("\n1️⃣ Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres",
            password=os.getenv("DB_PASSWORD")  # Change if different
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("   ✅ Connected")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("\n💡 Make sure PostgreSQL is running:")
        print("   Windows: sc query postgresql-x64-14")
        return False
    
    # Step 2: Check if database exists
    print("\n2️⃣ Checking if 'resume_screening' database exists...")
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname='resume_screening'")
    exists = cur.fetchone()
    
    if exists:
        print("   ℹ️ Database already exists")
    else:
        print("   Creating database...")
        try:
            cur.execute("CREATE DATABASE resume_screening")
            print("   ✅ Database created")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    cur.close()
    conn.close()
    
    # Step 3: Create tables using SQL file
    print("\n3️⃣ Creating tables...")
    sql_file = Path(__file__).parent.parent / "sql" / "create_feature_store.sql"
    
    if not sql_file.exists():
        print(f"   ❌ SQL file not found: {sql_file}")
        return False
    
    try:
        # Connect to resume_screening database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="resume_screening",
            user="postgres",
            password=os.getenv("DB_PASSWORD")
        )
        cur = conn.cursor()
        
        # Execute SQL file
        with open(sql_file, 'r') as f:
            sql = f.read()
            cur.execute(sql)
        
        conn.commit()
        print("   ✅ Tables created")
        
        # Verify tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        print(f"\n   Created {len(tables)} tables:")
        for table in tables:
            print(f"     - {table[0]}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ PHASE 5 DATABASE SETUP COMPLETE!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = setup_phase5_database()
    sys.exit(0 if success else 1)