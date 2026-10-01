"""Public NLPBridge application integration API; legacy nlbridge APIs remain available."""
from .selection import select_operation
from nlbridge.common import BridgeError, ModelError, PlanError
from nlbridge.providers import ChatModel

__version__ = "0.2.0"
__all__ = ["select_operation", "ChatModel", "BridgeError", "ModelError", "PlanError"]
