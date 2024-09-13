import yaml

CONFIG_FILE = 'config.yaml'

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

# Example usage:
if __name__ == "__main__":
    print("Current Token:", read_token())
    write_token("new_token_value")
    print("Updated Token:", read_token())