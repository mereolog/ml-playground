from abc import abstractmethod
from typing import Generic, TypeVar, Optional, Dict, Any, Callable, Awaitable
import asyncio

from algorithms.base.algorithm import Algorithm
from schemas.configs.algorithm_configs import SupervisedAlgorithmsParams

SP = TypeVar("SP", bound=SupervisedAlgorithmsParams)


class StreamingMixin:
    """Mixin class to add streaming capabilities to any algorithm."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        self.streaming_enabled = False
        self.current_epoch = 0
    
    def enable_streaming(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Enable streaming with a callback function for sending updates.
        
        Args:
            callback: Async function that takes update data and sends it via WebSocket
        """
        self.streaming_callback = callback
        self.streaming_enabled = True
    
    def disable_streaming(self):
        """Disable streaming updates."""
        self.streaming_callback = None
        self.streaming_enabled = False
    
    async def _send_streaming_update(self, update_data: Dict[str, Any]):
        """Send a streaming update if streaming is enabled.
        
        Args:
            update_data: Dictionary containing update information
        """
        if self.streaming_enabled and self.streaming_callback:
            try:
                await self.streaming_callback(update_data)
            except Exception as e:
                # Log error but don't stop training
                import logging
                logging.error(f"Failed to send streaming update: {e}")
    
    async def _send_training_started(self, total_epochs: int, data_shape: Dict[str, int]):
        """Send training started notification."""
        await self._send_streaming_update({
            "task": "training_started",
            "total_epochs": total_epochs,
            "data_shape": data_shape
        })
    
    async def _send_epoch_update(self, epoch: int, total_epochs: int, metrics: Dict[str, Any]):
        """Send epoch progress update."""
        await self._send_streaming_update({
            "task": "training_update", 
            "epoch": epoch,
            "total_epochs": total_epochs,
            "progress": (epoch / total_epochs) * 100,
            **metrics
        })
    
    async def _send_training_completed(self, final_metrics: Dict[str, Any]):
        """Send training completed notification."""
        await self._send_streaming_update({
            "task": "training_completed",
            **final_metrics
        })
    
    async def _send_training_error(self, error: str):
        """Send training error notification."""
        await self._send_streaming_update({
            "task": "training_error",
            "error": error
        })


class SupervisedAlgorithm(Algorithm[SP], Generic[SP]):
    """
    Base class for supervised algorithms
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def params(self) -> SP:
        """Supervised algorithm params"""
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fit the algorithm to training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def score(self, X, y):
        """Evaluate the algorithm performance.
        
        Args:
            X: Test features
            y: True targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass


class StreamingSupervisedAlgorithm(SupervisedAlgorithm[SP], StreamingMixin, Generic[SP]):
    """
    Base class for supervised algorithms with streaming capabilities.
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    async def fit_streaming(self, X, y):
        """Fit the algorithm with real-time streaming updates.
        
        This method should be overridden by subclasses to provide streaming training.
        Default implementation falls back to regular fit() method.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        try:
            # Send training started notification
            n_samples, n_features = X.shape
            await self._send_training_started(
                total_epochs=getattr(self.params, 'epochs', 1),
                data_shape={"samples": n_samples, "features": n_features}
            )
            
            # Default: use regular fit method
            result = self.fit(X, y)
            
            # Send completion notification
            final_metrics = self.score(X, y) if hasattr(self, 'score') else {}
            await self._send_training_completed({"final_metrics": final_metrics})
            
            return result
            
        except Exception as e:
            await self._send_training_error(str(e))
            raise
