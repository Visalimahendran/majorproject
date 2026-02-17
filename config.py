"""
Configuration Management
"""

import yaml
import os
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Create default config
            create_default_config(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set environment variables
        os.environ['NEUROMOTOR_DEBUG'] = str(config.get('system', {}).get('debug', False))
        
        return config
        
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()

def get_default_config():
    """Get default configuration"""
    return {
        'system': {
            'name': 'NeuroMotor Health System',
            'version': '1.0.0',
            'debug': True
        },
        'data_acquisition': {
            'webcam': {'device_id': 0, 'width': 1280, 'height': 720},
            'tablet': {'vendor_id': 0x056a, 'product_id': 0x030e},
            'digital': {'canvas_width': 800, 'canvas_height': 600}
        }
    }

def create_default_config(config_path):
    """Create default configuration file"""
    default_config = get_default_config()
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Created default config at {config_path}")

def save_config(config, config_path="config.yaml"):
    """Save configuration to file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_section(config, section_path):
    """Get specific configuration section"""
    sections = section_path.split('.')
    result = config
    
    for section in sections:
        if section in result:
            result = result[section]
        else:
            return None
    
    return result