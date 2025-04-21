import yaml

class ConfigNamespace:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            elif isinstance(value, list):
                # 处理列表中的字典元素
                setattr(self, key, [ConfigNamespace(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name):
        if name not in self.__dict__:
            raise AttributeError(f"'ConfigNamespace' object has no attribute '{name}'")
        return self.__dict__[name]
    
    def __repr__(self):
        return str(self.__dict__)
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                # 处理列表中的ConfigNamespace对象
                result[key] = [item.to_dict() if isinstance(item, ConfigNamespace) else item for item in value]
            else:
                result[key] = value
        return result

def load_config(config_path):
    """加载YAML配置文件并转换为ConfigNamespace对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigNamespace: 配置对象
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        return ConfigNamespace(config_dict) 
    

if __name__ == "__main__":
    config = load_config("../configs/env_config_1.yaml")
    print(config)
    print(config.map.size)
    print(config.map.obstacles.polygons[0].center)
    print(config.map.obstacles.circles[0].pos)
    print(config.robot.laser.fov)
    print(config.robot.goal.pos)