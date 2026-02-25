"""
ATEN Configuration Management
Centralized configuration with environment variables and defaults.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class FirebaseConfig:
    """Firebase configuration for state management and real-time collaboration"""
    credential_path: str = "config/firebase_credentials.json"
    project_id: str = "aten-trading"
    database_url: str = "https://aten-trading-default-rtdb.firebaseio.com/"
    firestore_collection: str = "trading_agents"
    
@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    default_exchange: str = "binance"
    supported_exchanges: list = field(default_factory=lambda: ["binance", "coinbase", "kraken"])
    max_position_size_usd: float = 10000.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    min_confidence_threshold: float = 0.65
    
@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    training_window_days: int = 180
    prediction_horizon_hours: int = 24
    model_save_path: str = "models/"
    retrain_interval_hours: int = 168  # Weekly retraining
    
@dataclass
class ATENConfig:
    """Main configuration container"""
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log_level: str = "INFO"
    data_cache_path: str = "data/cache/"
    
    @classmethod
    def from_env(cls) -> 'ATENConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        config.log_level = os.getenv("ATEN_LOG_LEVEL", config.log_level)
        config.trading.default_exchange = os.getenv("ATEN_DEFAULT_EXCHANGE", config.trading.default_exchange)
        
        # Firebase config from env
        firebase_creds = os.getenv("FIREBASE_CREDENTIALS_PATH")
        if firebase_creds and Path(firebase_creds).exists():
            config.firebase.credential_path = firebase_creds
            
        return config

def setup_logging(config: ATENConfig) -> logging.Logger:
    """Configure application logging"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=getattr(logging, config.log_level), format=log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.log_level))
    console_handler.setFormatter(logging.Formatter(log_format))
    
    logger = logging.getLogger("aten")
    logger.addHandler(console_handler)
    
    return logger