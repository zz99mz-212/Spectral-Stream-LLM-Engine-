"""
Common imports for the format package.

Provides a single point of import for widely used modules across the
format package. This reduces duplication and ensures consistent
import patterns.

Usage:
    from spectralstream.format._imports import (
        np, Path, os, json, struct, re, logging,
        Any, Dict, List, Optional, Tuple, Iterator,
    )
"""

from __future__ import annotations


import gzip
import json
import logging
import os
import re
import struct
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Tuple

import numpy as np
