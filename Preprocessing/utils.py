import yaml

CONFIG_FILE = 'Preprocessing/config.yaml'

def read_token():
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('HUGGINGFACE_TOKEN')

def write_token(new_token):
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
    
    config['HUGGINGFACE_TOKEN'] = new_token
    
    with open(CONFIG_FILE, 'w') as file:
        yaml.safe_dump(config, file)
