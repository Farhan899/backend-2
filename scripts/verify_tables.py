import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def verify_tables():
    # Parse the DATABASE_URL
    db_url = os.getenv('DATABASE_URL', '')
    # Convert asyncpg URL to standard connection string
    db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
    db_url = db_url.replace('?ssl=require', '')

    conn = await asyncpg.connect(db_url, ssl='require')

    try:
        # List all tables
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)

        print("✓ Database Tables:")
        for table in tables:
            print(f"  - {table['table_name']}")

        # Check tasks table columns
        if any(t['table_name'] == 'tasks' for t in tables):
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'tasks'
                ORDER BY ordinal_position;
            """)

            print("\n✓ Tasks Table Columns:")
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  - {col['column_name']}: {col['data_type']} ({nullable})")

        # Check foreign keys
        fks = await conn.fetch("""
            SELECT
                tc.constraint_name,
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_name='tasks';
        """)

        if fks:
            print("\n✓ Foreign Keys:")
            for fk in fks:
                print(f"  - {fk['constraint_name']}: {fk['table_name']}.{fk['column_name']} -> {fk['foreign_table_name']}.{fk['foreign_column_name']}")

        print("\n✓ Database setup verification complete!")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(verify_tables())
