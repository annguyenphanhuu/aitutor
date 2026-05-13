import asyncio
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings
from app.db.database import engine, Base, async_session
from app.db.models import User

async def test_postgres():
    settings = get_settings()
    print(f"Testing connection to: {settings.DATABASE_URL}")
    
    try:
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Tables created successfully.")
        
        # Test insert and query
        async with async_session() as session:
            # Create test user
            new_user = User(username="pg_test_user", display_name="PostgreSQL Test User")
            session.add(new_user)
            await session.commit()
            print("✅ Data inserted successfully.")
            
            # Query it back
            from sqlalchemy import select
            result = await session.execute(select(User).where(User.username == "pg_test_user"))
            user = result.scalar_one_or_none()
            if user:
                print(f"✅ Data retrieved successfully: {user.username} - {user.display_name}")
            else:
                print("❌ Failed to retrieve data.")
                
            # Cleanup
            await session.delete(user)
            await session.commit()
            print("✅ Test data cleaned up.")
            
    except Exception as e:
        print(f"❌ Connection or operation failed: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_postgres())
