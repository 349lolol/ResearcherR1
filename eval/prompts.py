"""Centralized prompts loaded from .env"""
import os
from dotenv import load_dotenv
load_dotenv()

ROUTER_SYSTEM_PROMPT = os.environ["ROUTER_SYSTEM_PROMPT"]
EFR_DEDUCE_SYSTEM_PROMPT = os.environ["EFR_DEDUCE_SYSTEM_PROMPT"]
BASELINE_DEDUCE_SYSTEM_PROMPT = os.environ["BASELINE_DEDUCE_SYSTEM_PROMPT"]
VERIFY_SYSTEM_PROMPT = os.environ["VERIFY_SYSTEM_PROMPT"]
REPAIR_SYSTEM_PROMPT = os.environ["REPAIR_SYSTEM_PROMPT"]
