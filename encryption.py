"""
Encryption utilities for secure data handling
"""

import base64
import hashlib
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

class DataEncryptor:
    """Encrypt and decrypt sensitive data"""
    
    def __init__(self, config):
        self.config = config
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)
    
    def _load_or_generate_key(self):
        """Load encryption key from config or generate new one"""
        key_str = self.config.get('security', {}).get('encryption_key', '')
        
        if key_str and len(key_str) >= 32:
            # Use existing key
            return base64.urlsafe_b64encode(key_str.encode()[:32].ljust(32, b'0'))
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Save to config
            if 'security' not in self.config:
                self.config['security'] = {}
            self.config['security']['encryption_key'] = base64.urlsafe_b64decode(key).decode()
            
            return key
    
    def encrypt_data(self, data):
        """Encrypt data"""
        try:
            if isinstance(data, dict):
                data_str = json.dumps(data)
            elif isinstance(data, str):
                data_str = data
            else:
                data_str = str(data)
            
            encrypted = self.cipher.encrypt(data_str.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
            
        except Exception as e:
            print(f"Encryption error: {e}")
            return None
    
    def decrypt_data(self, encrypted_data):
        """Decrypt data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return json.loads(decrypted.decode())
            
        except Exception as e:
            print(f"Decryption error: {e}")
            return None
    
    def create_hash(self, data, salt=None):
        """Create secure hash of data"""
        if salt is None:
            salt = self.config.get('security', {}).get('hash_salt', 'default_salt')
        
        data_str = str(data) + salt
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def anonymize_data(self, data, user_id):
        """Anonymize data by replacing identifiers"""
        if isinstance(data, dict):
            anonymized = data.copy()
            if 'user_id' in anonymized:
                anonymized['user_id'] = self.create_hash(user_id)
            if 'name' in anonymized:
                anonymized['name'] = f"User_{self.create_hash(user_id)[:8]}"
            return anonymized
        return data

class SecureStorage:
    """Handle secure data storage"""
    
    def __init__(self, encryptor):
        self.encryptor = encryptor
    
    def save_secure(self, data, filepath):
        """Save encrypted data to file"""
        try:
            encrypted = self.encryptor.encrypt_data(data)
            if encrypted:
                with open(filepath, 'w') as f:
                    f.write(encrypted)
                return True
            return False
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def load_secure(self, filepath):
        """Load and decrypt data from file"""
        try:
            with open(filepath, 'r') as f:
                encrypted = f.read()
            return self.encryptor.decrypt_data(encrypted)
        except Exception as e:
            print(f"Load error: {e}")
            return None
    
    def save_session(self, session_data, session_id):
        """Save session data securely"""
        # Create session directory
        session_dir = f"./sessions/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        # Separate sensitive and non-sensitive data
        sensitive = {
            'user_id': session_data.get('user_id'),
            'raw_data': session_data.get('data', [])
        }
        
        non_sensitive = {
            'session_id': session_data.get('session_id'),
            'type': session_data.get('type'),
            'start_time': session_data.get('start_time'),
            'features': session_data.get('features', {}),
            'results': session_data.get('results', {})
        }
        
        # Save encrypted sensitive data
        self.save_secure(sensitive, f"{session_dir}/sensitive.enc")
        
        # Save non-sensitive data as JSON
        with open(f"{session_dir}/session.json", 'w') as f:
            json.dump(non_sensitive, f, indent=2, default=str)
        
        # Create hash for verification
        session_hash = self.encryptor.create_hash(str(non_sensitive))
        with open(f"{session_dir}/hash.txt", 'w') as f:
            f.write(session_hash)
        
        return session_dir