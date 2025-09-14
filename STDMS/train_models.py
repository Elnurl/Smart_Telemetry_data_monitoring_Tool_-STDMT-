#!/usr/bin/env python3
"""
Train ML Models for STDMS
Simple script to train the anomaly detection models
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from telemetry_monitor.anomaly import get_anomaly_detector
from telemetry_monitor.storage import get_database
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_models():
    """Train ML models with existing telemetry data"""
    try:
        print("🚀 Starting ML Model Training...")
        
        # Get detector and database
        detector = get_anomaly_detector()
        db = get_database()
        
        # Get training data using the correct database interface
        logger.info("Fetching training data from database...")
        
        # Use the database query method
        recent_data = db.get_recent_telemetry(limit=500)
        
        if len(recent_data) < 50:
            print(f"⚠️ Need more telemetry data for training. Found only {len(recent_data)} records.")
            print("💡 Let the simulator run for a few minutes to collect more data.")
            return False
        
        print(f"✅ Found {len(recent_data)} data points for training")
        
        # Extract features for training (solar_power, attitude_x, orbit_altitude)
        training_data = []
        for record in recent_data:
            # Assuming record has these fields based on telemetry schema
            training_data.append([
                record.get('solar_power', 0),
                record.get('attitude_x', 0), 
                record.get('orbit_altitude', 0)
            ])
        
        print("🤖 Training ML models...")
        detector.train_models(training_data)
        
        print("🎉 ML Models trained successfully!")
        print("✅ Smart anomaly detection is now active!")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        
        # Try with simulated data if real data fails
        print("🔄 Trying with simulated training data...")
        try:
            import numpy as np
            # Generate some realistic training data
            simulated_data = []
            for i in range(100):
                simulated_data.append([
                    np.random.normal(50, 10),    # solar_power
                    np.random.normal(0, 5),      # attitude_x
                    np.random.normal(400, 50)    # orbit_altitude
                ])
            
            detector.train_models(simulated_data)
            print("✅ Models trained with simulated data!")
            return True
            
        except Exception as e2:
            logger.error(f"Simulated training also failed: {e2}")
            print(f"❌ Both training attempts failed: {e2}")
            return False

if __name__ == "__main__":
    success = train_models()
    if success:
        print("\n🎯 Next steps:")
        print("1. Run the STDMS application: python -m telemetry_monitor.main")
        print("2. Start the simulator from the Dashboard") 
        print("3. Watch for smart anomaly detection in action!")
    else:
        print("\n💡 Try running the STDMS GUI and let the simulator collect more data first.")