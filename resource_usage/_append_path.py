import sys
import os
_dir = os.getcwd()
print(_dir)
sys.path.append(_dir)
sys.path.append(os.path.dirname(_dir))
