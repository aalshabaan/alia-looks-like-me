from data import *
import sqlite3
import sqlite_vec

# Enable sqlite_vec if not enabled
db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

