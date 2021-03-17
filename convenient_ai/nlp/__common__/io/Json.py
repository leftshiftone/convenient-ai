import json
import os
from typing import Dict


class Json:
    @staticmethod
    def read(path: str, name: str) -> Dict:
        with open(os.path.join(path, name + ".json"), encoding='utf-8') as open_file:
            return json.load(open_file)

    @staticmethod
    def write(path: str, name: str, content: Dict):
        with open(os.path.join(path, name + ".json"), 'w', encoding='utf-8') as file:
            json.dump(content, file, ensure_ascii=False)
