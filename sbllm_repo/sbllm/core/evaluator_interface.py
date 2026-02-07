from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

@dataclass
class EvalMetric:
    """Standardized evaluation metrics."""
    time: float = 99999.0
    size: int = 0
    correct: bool = False
    output: str = ""
    error: str = ""
    compile_success: bool = False
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def build_env(self):
        """Setup build environment."""
        pass

    @abstractmethod
    def evaluate(self, code: str, metadata: Dict[str, Any], **kwargs) -> EvalMetric:
        """
        Evaluate a single code snippet.
        
        Args:
            code: The source code to evaluate.
            metadata: Context like input data, benchmark name, etc.
            
        Returns:
            EvalMetric object containing results.
        """
        pass
