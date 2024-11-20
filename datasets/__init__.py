import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

utilities_dir = os.path.join(current_dir, '..', 'utilities')

if utilities_dir not in sys.path:
    sys.path.append(utilities_dir)
