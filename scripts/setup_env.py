"""
Environment Setup Script
Helps create a properly configured .env file
"""

import secrets
import shutil
from pathlib import Path


def generate_secret_key():
    """Generate a secure secret key"""
    return secrets.token_urlsafe(32)


def setup_env():
    """Set up .env file from .env.example"""
    
    print("\n" + "="*50)
    print("🔐 Environment Setup")
    print("="*50 + "\n")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    # Check if .env.example exists
    if not env_example.exists():
        print("❌ .env.example not found!")
        return False
    
    # Check if .env already exists
    if env_file.exists():
        response = input("⚠️  .env already exists. Overwrite? (y/N): ").lower()
        if response != 'y':
            print("ℹ️  Keeping existing .env file")
            return True
    
    # Copy .env.example to .env
    shutil.copy(env_example, env_file)
    print("✅ Created .env from .env.example")
    
    # Read .env content
    content = env_file.read_text()
    
    # Generate new secret key
    new_secret = generate_secret_key()
    
    # Replace placeholders
    replacements = {
        'your-secret-key-change-this-in-production': new_secret,
        'your_password_here': 'postgres123'  # Default password
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Write updated content
    env_file.write_text(content)
    
    print("\n✅ Updated .env with secure values:")
    print(f"   SECRET_KEY: {new_secret[:20]}...")
    print(f"   DATABASE_PASSWORD: postgres123")
    
    print("\n" + "="*50)
    print("✅ Environment Setup Complete!")
    print("="*50)
    
    print("\n⚠️  IMPORTANT:")
    print("   1. Review .env file and update if needed")
    print("   2. Never commit .env to Git")
    print("   3. Change DATABASE_PASSWORD for production")
    
    print("\n💡 Your .env is ready to use!")
    
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if setup_env() else 1)