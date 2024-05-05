from eq_data_loader import get_eq_data
import json

data_config = json.load(open('/workspaces/runtime_test/data_config.json', 'r'))

data = get_eq_data(data_config['correlation_thresh'])

print(data.head())