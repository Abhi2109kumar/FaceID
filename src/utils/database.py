import json
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="data/users.json"):
        self.db_path = db_path
        self._ensure_db_exists()
        self.users = self._load()

    def _ensure_db_exists(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({}, f)

    def _load(self):
        with open(self.db_path, "r") as f:
            return json.load(f)

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.users, f, indent=4)

    def register_user(self, name, signature):
        """
        Saves a user's name and their face signature (landmark list).
        """
        self.users[name] = {
            "signature": signature,
            "created_at": datetime.now().isoformat()
        }
        self._save()
        return True

    def get_all_users(self):
        return self.users

    def find_user(self, name):
        return self.users.get(name)
