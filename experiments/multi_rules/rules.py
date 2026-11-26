"""兼容入口：规则实现已移到 app/multi_rules.py（服务端由 HC_MULTI_RULES 开关控制），这里只做转发。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.multi_rules import *  # noqa: E402,F401,F403
from app.multi_rules import postprocess  # noqa: E402,F401
