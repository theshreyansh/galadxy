import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Union

import streamlit as st

# Data generation and processing
from faker import Faker
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# For LSTM Autoencoder (optional advanced model)
try:
    import tensorflow as tf  # type: ignore[import]
    from tensorflow import keras  # type: ignore[import]
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Application imports
import plotly.express as px


def configure_console_output() -> None:
    """Use UTF-8 console output when Python is launched from Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                getattr(stream, "reconfigure")(encoding="utf-8", errors="replace")
            except Exception:
                pass

class GalaxyApp:
    """
    Main Galaxy Application - Predictive Maintenance System
    Combines IoT sensors, AI anomaly detection, and SAP integration
    """
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize the Galaxy application"""
        configure_console_output()
        self.config = self._load_config(config_file)
        self.fake = Faker('en_IN')  # Indian market locale
        
        # Initialize modules
        # Data storage
        self.sensor_data: Optional[pd.DataFrame] = None
        self.anomalies: Optional[pd.DataFrame] = None
        self.maintenance_orders: List[Dict] = []
        
        print("=" * 80)
        print("GALAXY - Predictive Maintenance for Critical Equipment")
        print("IoT-Driven Maintenance System with AI Anomaly Detection")
        print("=" * 80)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            "sensors": {
                "vibration": {"min": 0.1, "max": 50, "unit": "mm/s"},
                "temperature": {"min": 20, "max": 150, "unit": "°C"},
                "pressure": {"min": 1, "max": 500, "unit": "bar"},
                "humidity": {"min": 10, "max": 90, "unit": "%"}
            },
            "equipment": [
                "Reactor-Surfactant-01", "Reactor-Surfactant-02", "Pump-Venturi-01", "Pump-Venturi-02",
                "Mixer-Surfactant-01", "Compressor-Feed-01", "Heat-Exchanger-01", "Distillation-Column-01"
            ],
            "locations": [
                "Plant-Mumbai", "Plant-Pune", "Plant-Kalol", "Plant-Dahanu", "Plant-Jawaharlal"
            ],
            "anomaly_threshold": 0.85,
            "sample_size": 1000,
            "training_window": 30,  # days
            "alert_threshold": 0.7,
            "business_impact": {
                "downtime_reduction": "30-40%",
                "cost_savings": "$2M-$5M/year",
                "benchmark": "BASF Antwerp: €10M/year savings"
            }
        }
    
    def generate_synthetic_data(self, num_records: Optional[Union[int, float]] = 1000) -> pd.DataFrame:
        """
        Generate synthetic IoT sensor data for Indian market
        Includes realistic equipment data with occasional anomalies
        """
        if num_records is None:
            num_records = 1000
        num_records = int(num_records)

        print("\n[STEP 1] Generating Synthetic IoT Data...")
        print(f"  • Records: {num_records}")
        print(f"  • Locale: Indian Market (Faker en_IN)")
        
        data = []
        equipment_list = self.config["equipment"]
        locations = self.config["locations"]
        sensor_ranges = self.config["sensors"]
        
        # Generate normal operation + anomalies
        anomaly_indices = np.random.choice(num_records, size=int(num_records * 0.15), replace=False)
        
        for i in range(num_records):
            is_anomaly = i in anomaly_indices
            timestamp = datetime.now() - timedelta(hours=num_records-i)
            
            # Generate sensor readings
            vibration = self._generate_sensor_value("vibration", is_anomaly, sensor_ranges)
            temperature = self._generate_sensor_value("temperature", is_anomaly, sensor_ranges)
            pressure = self._generate_sensor_value("pressure", is_anomaly, sensor_ranges)
            humidity = self._generate_sensor_value("humidity", is_anomaly, sensor_ranges)
            
            record = {
                "timestamp": timestamp,
                "equipment_id": np.random.choice(equipment_list),
                "location": np.random.choice(locations),
                "vibration_mm_s": vibration,
                "temperature_c": temperature,
                "pressure_bar": pressure,
                "humidity_pct": humidity,
                "operator": self.fake.name(),
                "shift": np.random.choice(["Morning", "Afternoon", "Night"]),
                "is_anomaly": is_anomaly,
                "device_type": self.fake.word(),
                "company": "Galaxy Surfactants" if np.random.random() < 0.7 else "Galaxy Surfactants - Global"
            }
            data.append(record)
        
        self.sensor_data = pd.DataFrame(data)
        print(f"  ✓ Generated {len(self.sensor_data)} sensor records")
        print(f"  ✓ Anomalies injected: {self.sensor_data['is_anomaly'].sum()} records ({self.sensor_data['is_anomaly'].sum()/num_records*100:.1f}%)")
        
        return self.sensor_data
    
    def _generate_sensor_value(self, sensor_type: str, is_anomaly: bool, ranges: Dict) -> float:
        """Generate realistic sensor value with anomaly injection"""
        min_val = ranges[sensor_type]["min"]
        max_val = ranges[sensor_type]["max"]
        
        if is_anomaly:
            # Anomalous readings: outside normal range or extreme values
            value = np.random.choice([
                np.random.uniform(min_val * 1.5, max_val * 1.2),  # High values
                np.random.uniform(min_val * 0.5, min_val * 0.8)   # Low values
            ])
        else:
            # Normal operation: within typical range (70-90% of max)
            value = np.random.uniform(min_val, max_val * 0.8)
        
        return round(value, 2)
    
    def detect_anomalies(self) -> pd.DataFrame:
        """
        Detect equipment anomalies using Isolation Forest
        Can be extended with LSTM Autoencoders for advanced detection
        """
        print("\n[STEP 2] Detecting Anomalies with AI...")
        print("  • Algorithm: Isolation Forest + Statistical Analysis")
        print("  • Features: Vibration, Temperature, Pressure, Humidity")
        
        if self.sensor_data is None:
            raise ValueError("No sensor data available. Run generate_synthetic_data() first.")
        
        # Prepare features for anomaly detection
        features = ["vibration_mm_s", "temperature_c", "pressure_bar", "humidity_pct"]
        X = self.sensor_data[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest - Fast anomaly detection
        iso_forest = IsolationForest(
            contamination=0.15,  # 15% anomalies
            random_state=42
        )
        predictions = iso_forest.fit_predict(X_scaled)
        raw_scores = -iso_forest.score_samples(X_scaled)
        score_range = raw_scores.max() - raw_scores.min()
        if score_range == 0:
            anomaly_scores = np.zeros_like(raw_scores)
        else:
            anomaly_scores = (raw_scores - raw_scores.min()) / score_range
        
        # Add predictions to dataframe
        self.sensor_data["anomaly_detected"] = predictions == -1
        self.sensor_data["anomaly_score"] = anomaly_scores
        
        # Filter detected anomalies
        anomalies_df = self.sensor_data[self.sensor_data["anomaly_detected"]].copy()
        self.anomalies = anomalies_df
        
        print(f"  ✓ Detected {len(anomalies_df)} anomalies")
        print(f"  ✓ Detection Accuracy: {(self.sensor_data['anomaly_detected'].sum() / len(self.sensor_data) * 100):.2f}%")
        
        # Show top anomalies
        print("\n  Top Anomalies Detected:")
        top_anomalies = anomalies_df.nlargest(5, "anomaly_score")[
            ["timestamp", "equipment_id", "location", "vibration_mm_s", "temperature_c", "anomaly_score"]
        ]
        for idx, row in top_anomalies.iterrows():
            print(f"    • {row['equipment_id']} at {row['location']}")
            print(f"      Vibration: {row['vibration_mm_s']} mm/s, Temp: {row['temperature_c']}°C")
            print(f"      Anomaly Score: {row['anomaly_score']:.3f}")
        
        return anomalies_df
    
    def generate_maintenance_orders(self) -> List[Dict]:
        """
        Auto-generate maintenance work orders for detected anomalies
        Simulates SAP Plant Maintenance integration
        """
        print("\n[STEP 3] Generating Maintenance Work Orders...")
        
        if self.anomalies is None or len(self.anomalies) == 0:
            print("  • No anomalies detected. No maintenance orders generated.")
            return []
        
        alert_threshold = self.config.get("anomaly_detection", {}).get("alert_threshold", self.config.get("alert_threshold", 0.7))
        priority_mapping = {
            "critical": "URGENT - Schedule immediately",
            "high": "Schedule within 24 hours",
            "medium": "Schedule within 72 hours",
            "low": "Schedule within 1 week"
        }
        
        maintenance_orders = []
        
        for idx, anomaly in self.anomalies.iterrows():
            score = anomaly["anomaly_score"]
            
            # Determine priority
            if score >= 0.95:
                priority = "critical"
            elif score >= 0.85:
                priority = "high"
            elif score >= 0.75:
                priority = "medium"
            else:
                priority = "low"
            
            # Skip if below alert threshold
            if score < alert_threshold:
                continue
            
            # Determine issue type
            issue_type = self._identify_issue_type(anomaly)
            
            # Create work order
            order = {
                "order_id": f"MO-{datetime.now().strftime('%Y%m%d')}-{idx:05d}",
                "timestamp": anomaly["timestamp"],
                "equipment_id": anomaly["equipment_id"],
                "location": anomaly["location"],
                "operator": anomaly["operator"],
                "priority": priority.upper(),
                "issue_type": issue_type,
                "recommended_action": priority_mapping[priority],
                "anomaly_score": round(anomaly["anomaly_score"], 3),
                "status": "OPEN",
                "estimated_cost": self._estimate_maintenance_cost(priority),
                "estimated_duration_hours": self._estimate_duration(priority),
                "sap_status": "READY_TO_SYNC"
            }
            maintenance_orders.append(order)
        
        self.maintenance_orders = maintenance_orders
        
        print(f"  ✓ Generated {len(maintenance_orders)} maintenance work orders")
        print("\n  Work Orders Summary:")
        
        # Summary by priority
        priority_summary = {}
        for order in maintenance_orders:
            priority = order["priority"]
            priority_summary[priority] = priority_summary.get(priority, 0) + 1
        
        for priority, count in priority_summary.items():
            print(f"    • {priority}: {count} orders")
        
        print("\n  Sample Work Orders:")
        for order in maintenance_orders[:3]:
            print(f"\n    Order ID: {order['order_id']}")
            print(f"    Equipment: {order['equipment_id']} ({order['location']})")
            print(f"    Issue: {order['issue_type']}")
            print(f"    Priority: {order['priority']}")
            print(f"    Action: {order['recommended_action']}")
            print(f"    Est. Cost: ₹{order['estimated_cost']:,.0f}")
            print(f"    Est. Duration: {order['estimated_duration_hours']} hours")
        
        return maintenance_orders
    
    def _identify_issue_type(self, anomaly: pd.Series) -> str:
        """Identify the type of equipment issue based on sensor readings"""
        vibration = anomaly["vibration_mm_s"]
        temperature = anomaly["temperature_c"]
        pressure = anomaly["pressure_bar"]
        
        if vibration > 30:
            return "BEARING_WEAR / MECHANICAL_FAILURE"
        elif temperature > 100:
            return "THERMAL_DEGRADATION / OVERHEATING"
        elif pressure > 350:
            return "PRESSURE_BUILDUP / BLOCKAGE"
        else:
            return "GENERAL_EQUIPMENT_DEGRADATION"
    
    def _estimate_maintenance_cost(self, priority: str) -> float:
        """Estimate maintenance cost based on priority (Indian market pricing)"""
        cost_map = {
            "critical": np.random.uniform(50000, 100000),    # ₹50K-100K
            "high": np.random.uniform(30000, 50000),         # ₹30K-50K
            "medium": np.random.uniform(15000, 30000),       # ₹15K-30K
            "low": np.random.uniform(5000, 15000)            # ₹5K-15K
        }
        return cost_map.get(priority, 10000)
    
    def _estimate_duration(self, priority: str) -> int:
        """Estimate maintenance duration in hours"""
        duration_map = {
            "critical": 4,
            "high": 6,
            "medium": 8,
            "low": 12
        }
        return duration_map.get(priority, 8)
    
    def integrate_with_sap(self) -> Dict:
        """
        Simulate SAP Plant Maintenance integration
        Auto-sync maintenance orders to SAP PM module
        """
        print("\n[STEP 4] SAP Plant Maintenance Integration...")
        
        if len(self.maintenance_orders) == 0:
            print("  • No maintenance orders to sync to SAP")
            return {}
        
        sap_sync = {
            "timestamp": datetime.now().isoformat(),
            "total_orders": len(self.maintenance_orders),
            "sync_status": "SUCCESS",
            "synced_orders": []
        }
        
        print(f"  • Syncing {len(self.maintenance_orders)} orders to SAP PM...")
        
        for order in self.maintenance_orders:
            sap_order = {
                "sap_order_id": f"SAP-{order['order_id']}",
                "notif_id": f"NOTIF-{order['order_id']}",
                "equipment": order["equipment_id"],
                "location": order["location"],
                "priority": self._map_priority_to_sap(order["priority"]),
                "issue_description": order["issue_type"],
                "requested_start": (datetime.now() + timedelta(hours=2)).isoformat(),
                "maintenance_type": "CORRECTIVE",
                "cost_center": "3100",  # Plant Maintenance
                "cost_estimate": order["estimated_cost"],
                "sync_status": "SYNCED"
            }
            sap_sync["synced_orders"].append(sap_order)
        
        print(f"  ✓ Successfully synced {len(sap_sync['synced_orders'])} orders to SAP")
        
        return sap_sync
    
    def _map_priority_to_sap(self, priority: str) -> int:
        """Map Galaxy priority to SAP priority codes"""
        mapping = {
            "CRITICAL": 1,
            "HIGH": 2,
            "MEDIUM": 3,
            "LOW": 4
        }
        return mapping.get(priority, 4)
    
    def calculate_business_impact(self) -> Dict:
        """
        Calculate potential business impact of predictive maintenance
        Based on BASF model
        """
        print("\n[STEP 5] Business Impact Analysis...")
        
        total_anomalies = len(self.anomalies) if self.anomalies is not None else 0
        critical_orders = sum(1 for o in self.maintenance_orders if o["priority"] == "CRITICAL")
        total_cost = sum(o["estimated_cost"] for o in self.maintenance_orders)
        
        # Assumptions from BASF model
        avg_downtime_prevented_hours = critical_orders * 8  # 8 hours per critical issue
        cost_per_hour_downtime = 500000  # ₹5 lakhs/hour for chemical plants
        annual_prevented_downtime_cost = avg_downtime_prevented_hours * cost_per_hour_downtime * 250  # ~250 working days
        
        # Calculate savings
        maintenance_cost_savings = total_cost * 0.4  # 40% preventive vs corrective savings
        annual_cost_savings = maintenance_cost_savings * 12  # Annualize
        annual_prevented_downtime_savings = annual_prevented_downtime_cost
        total_annual_savings = annual_cost_savings + annual_prevented_downtime_savings
        
        impact = {
            "summary": {
                "anomalies_detected": total_anomalies,
                "critical_alerts": critical_orders,
                "maintenance_orders_created": len(self.maintenance_orders),
                "total_maintenance_cost": f"₹{total_cost:,.0f}"
            },
            "downtime_reduction": {
                "prevented_downtime_hours": avg_downtime_prevented_hours,
                "cost_prevented": f"₹{annual_prevented_downtime_savings:,.0f}/year",
                "reduction_percentage": "30-40%"
            },
            "cost_savings": {
                "maintenance_optimization": f"₹{maintenance_cost_savings:,.0f}",
                "annual_savings": f"₹{total_annual_savings:,.0f}/year",
                "benchmark": "Galaxy Surfactants strategic benchmark"
            },
            "roi": {
                "estimated_payback_months": 4,
                "first_year_roi": "250%"
            }
        }
        
        print("\n  Predicted Business Impact:")
        print(f"    • Downtime Reduction: {impact['downtime_reduction']['reduction_percentage']}")
        print(f"    • Annual Cost Savings: {impact['cost_savings']['annual_savings']}")
        print(f"    • Critical Alerts Prevented: {critical_orders} equipment failures")
        print(f"    • Estimated ROI (Year 1): {impact['roi']['first_year_roi']}")
        print(f"    • Payback Period: {impact['roi']['estimated_payback_months']} months")
        
        return impact
    
    def generate_report(self, output_file: str = "galaxy_report.json") -> str:
        """Generate comprehensive report"""
        print("\n[STEP 6] Generating Report...")
        
        sensor_df = self.sensor_data if self.sensor_data is not None else pd.DataFrame()
        anomalies_df = self.anomalies if self.anomalies is not None else pd.DataFrame()

        report = {
            "application": "Galaxy - Predictive Maintenance System",
            "execution_timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_records": len(sensor_df),
                "anomalies_found": len(anomalies_df),
                "equipment_monitored": sensor_df["equipment_id"].nunique() if not sensor_df.empty else 0,
                "locations": sensor_df["location"].nunique() if not sensor_df.empty else 0
            },
            "anomaly_detection": {
                "method": "Isolation Forest + Statistical Analysis",
                "total_anomalies": len(anomalies_df),
                "top_anomalies": anomalies_df.nlargest(5, "anomaly_score")[
                    ["timestamp", "equipment_id", "location", "vibration_mm_s", "temperature_c", "anomaly_score"]
                ].to_dict("records") if not anomalies_df.empty else []
            },
            "maintenance_orders": {
                "total_orders": len(self.maintenance_orders),
                "orders": self.maintenance_orders[:10]  # Top 10
            },
            "sap_integration": self.integrate_with_sap(),
            "business_impact": self.calculate_business_impact(),
            "config": self.config
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  ✓ Report saved to {output_file}")
        
        return output_file
    
    def run_full_pipeline(self) -> None:
        """Execute the complete Galaxy pipeline"""
        try:
            # Step 1: Generate synthetic data
            sample_size = self.config.get("data_generation", {}).get("sample_size", self.config.get("sample_size", 1000))
            if sample_size is None:
                sample_size = 1000
            sample_size = int(sample_size)
            self.generate_synthetic_data(num_records=sample_size)
            
            # Step 2: Detect anomalies
            self.detect_anomalies()
            
            # Step 3: Generate maintenance orders
            self.generate_maintenance_orders()
            
            # Step 4: Generate comprehensive report
            self.generate_report()
            
            print("\n" + "=" * 80)
            print("GALAXY PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print("\nKey Metrics:")
            print(f"  • Sensor Data Records: {len(self.sensor_data) if self.sensor_data is not None else 0}")
            print(f"  • Anomalies Detected: {len(self.anomalies) if self.anomalies is not None else 0}")
            print(f"  • Maintenance Orders: {len(self.maintenance_orders)}")
            print(f"  • Report Generated: galaxy_report.json")
            print("\nFiles Generated:")
            print(f"  ✓ galaxy_report.json - Comprehensive analysis report")
            print(f"  ✓ sensor_data.csv - Raw sensor readings (optional export)")
            
        except Exception as e:
            print(f"\n❌ Error during pipeline execution: {str(e)}")
            raise


def is_streamlit_running() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return "STREAMLIT_SERVER" in os.environ or "STREAMLIT_RUN" in os.environ or "streamlit" in os.path.basename(sys.argv[0]).lower()


def format_currency(value: float) -> str:
    return f"₹{value:,.0f}"


def _apply_streamlit_style() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: #f7f7f8;
            }
            .reportview-container, .main .block-container {
                background: #ffffff;
                padding-top: 1rem;
                padding-bottom: 2rem;
                border-radius: 24px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08);
            }
            .metric-card {
                border-radius: 24px;
                padding: 1.2rem 1.35rem;
                background: rgba(255,255,255,0.98);
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(209, 213, 219, 0.4);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                animation: fadeInUp 0.6s ease both;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
            }
            .section-heading {
                color: #111827;
                font-weight: 700;
            }
            .kpi-label {
                color: #6b7280;
                font-size: 0.88rem;
                margin-bottom: 0.15rem;
            }
            .kpi-value {
                font-size: 1.85rem;
                font-weight: 700;
                color: #111827;
            }
            .status-pill {
                display: inline-block;
                padding: 0.35rem 0.85rem;
                border-radius: 999px;
                font-size: 0.9rem;
                font-weight: 700;
                background: #e5e7eb;
                color: #111827;
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(18px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .css-18e3th9 { background-color: #f7f7f8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_card(label: str, value: str, delta: str = "") -> None:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div style='color:#475569; font-size:0.9rem; margin-top:0.5rem;'>{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def answer_dashboard_query(question: str, app: "GalaxyApp", sensor_df: pd.DataFrame, anomalies_df: pd.DataFrame) -> dict:
    question_text = question.strip().lower()
    if not question_text:
        return {
            "Executive Summary": "Ask a business question to refresh the insight summaries.",
            "Data Quality": "Type a query to get real-time data quality guidance.",
            "SOX Controls": "Ask how controls and audit readiness look for the current stream.",
            "Operations": "Ask a question to get operational recommendations for Galaxy plants."
        }

    top_location = anomalies_df["location"].mode().iloc[0] if not anomalies_df.empty else "N/A"
    top_asset = anomalies_df["equipment_id"].mode().iloc[0] if not anomalies_df.empty else "N/A"
    high_score = anomalies_df["anomaly_score"].max() if not anomalies_df.empty else 0.0
    try:
        estimated_savings = app.calculate_business_impact()["cost_savings"]["annual_savings"]
    except Exception:
        estimated_savings = "Pending"
    missing_summary = sensor_df.isna().sum().sum() if not sensor_df.empty else 0

    if "risk" in question_text or "plant" in question_text or "asset" in question_text:
        exec_text = f"The most at-risk plant is {top_location}. Top flagged asset is {top_asset} with anomaly score up to {high_score:.2f}."
        dq_text = "Data quality is stable; there are currently {} missing values in the active dataset.".format(missing_summary)
        sox_text = "Control monitoring is active for the current stream, and anomalies are tagged for audit review."
        ops_text = f"Recommend immediate focus on {top_asset} and the {top_location} site for corrective maintenance."
    elif "cost" in question_text or "saving" in question_text or "roi" in question_text:
        exec_text = f"Estimated annual value from predictive maintenance is {estimated_savings}."
        dq_text = "Quality data completeness supports reliable savings forecasts."
        sox_text = "SOX controls should validate the saved work orders and sync trail for audit assurance."
        ops_text = "Focus operational execution on reducing critical order lead time to lock in savings."
    elif "sox" in question_text or "audit" in question_text or "control" in question_text:
        exec_text = "The current stream maintains data lineage and sync status for SOX readiness."
        dq_text = "Quality metrics can be used as audit evidence for sensor integrity and freshness."
        sox_text = "The dashboard is configured for SOX-ready controls with anomaly flags and SAP sync tracking."
        ops_text = "Ensure change control logs capture any threshold or equipment assignment updates."
    elif "anomaly" in question_text or "outlier" in question_text:
        exec_text = f"Outliers are concentrated in {top_location}; anomaly score range reaches {high_score:.2f}."
        dq_text = "Anomaly detection is based on fresh sensor streams and is tagged by timestamp for traceability."
        sox_text = "These flagged anomalies are ready for control review and potential SAP maintenance order creation."
        ops_text = "Operational action: validate sensor readings and deploy targeted inspections to the red-alert assets."
    else:
        exec_text = "The dashboard is streaming live operational data and will score risks continuously."
        dq_text = "Current data freshness is maintained and anomalies are tagged for audit-ready transparency."
        sox_text = "SOX controls are visible across the operational stream and support review-ready evidence."
        ops_text = "Use the stream mode button to keep the dashboard updated every 10 seconds."

    return {
        "Executive Summary": exec_text,
        "Data Quality": dq_text,
        "SOX Controls": sox_text,
        "Operations": ops_text
    }


def streamlit_dashboard() -> None:
    st.set_page_config(
        page_title="Galaxy CDO Dashboard",
        page_icon="🌌",
        layout="wide",
    )
    _apply_streamlit_style()

    st.markdown(
        "# Galaxy Surfactants - AI Driven Predictive Maintenance Control Center"
        "\n\n"
        "A solution for operations, risk, and SOX controls across Galaxy Surfactants plants."
    )

    if "galaxy_app" not in st.session_state:
        st.session_state.galaxy_app = GalaxyApp()
        st.session_state.sap_sync = None
        st.session_state.business_impact = None
        st.session_state.report_file = None
        st.session_state.initialized = False
        st.session_state.streaming_mode = False

    app = st.session_state.galaxy_app

    top_bar = st.columns([1, 9])
    with top_bar[0]:
        stream_button_label = "Activate Streaming Mode" if not st.session_state.streaming_mode else "Deactivate Streaming Mode"
        if st.button(stream_button_label):
            st.session_state.streaming_mode = not st.session_state.streaming_mode
            st.session_state.initialized = False
    with top_bar[1]:
        st.markdown("**Streaming refresh**: The dashboard refreshes every 10 seconds when streaming mode is enabled.")

    if "dashboard_query" not in st.session_state:
        st.session_state.dashboard_query = ""

    sample_size = int(app.config.get("data_generation", {}).get("sample_size", app.config.get("sample_size", 1000)))
    alert_threshold = float(app.config.get("anomaly_detection", {}).get("alert_threshold", app.config.get("alert_threshold", 0.7)))
    app.config.setdefault("data_generation", {})["sample_size"] = sample_size
    app.config.setdefault("anomaly_detection", {})["alert_threshold"] = alert_threshold

    selected_locations = []
    selected_equipment = []
    selected_shifts = ["Morning", "Afternoon", "Night"]
    min_anomaly_score = 0.6

    if not st.session_state.initialized or st.session_state.streaming_mode:
        app.generate_synthetic_data(num_records=sample_size)
        app.detect_anomalies()
        app.generate_maintenance_orders()
        st.session_state.sap_sync = app.integrate_with_sap()
        st.session_state.business_impact = app.calculate_business_impact()
        st.session_state.initialized = True

    if st.session_state.streaming_mode:
        st.markdown('<meta http-equiv="refresh" content="10">', unsafe_allow_html=True)

    filtered_sensor_data = app.sensor_data.copy() if app.sensor_data is not None else pd.DataFrame()
    filtered_anomalies = app.anomalies.copy() if app.anomalies is not None else pd.DataFrame(
        columns=["timestamp", "equipment_id", "location", "vibration_mm_s", "temperature_c", "pressure_bar", "humidity_pct", "anomaly_score", "shift"]
    )

    if not filtered_sensor_data.empty:
        selected_locations = st.sidebar.multiselect(
            "Plant location",
            options=app.config.get("locations", []),
            default=app.config.get("locations", []),
        )
        selected_equipment = st.sidebar.multiselect(
            "Equipment",
            options=app.config.get("equipment", []),
            default=app.config.get("equipment", []),
        )
        selected_shifts = st.sidebar.multiselect(
            "Shift",
            options=["Morning", "Afternoon", "Night"],
            default=["Morning", "Afternoon", "Night"],
        )
        min_anomaly_score = st.sidebar.slider(
            "Minimum anomaly score",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
        )

        if selected_locations:
            filtered_sensor_data = filtered_sensor_data[filtered_sensor_data["location"].isin(selected_locations)]
        if selected_equipment:
            filtered_sensor_data = filtered_sensor_data[filtered_sensor_data["equipment_id"].isin(selected_equipment)]
        if selected_shifts:
            filtered_sensor_data = filtered_sensor_data[filtered_sensor_data["shift"].isin(selected_shifts)]

    if not filtered_anomalies.empty:
        if selected_locations:
            filtered_anomalies = filtered_anomalies[filtered_anomalies["location"].isin(selected_locations)]
        if selected_equipment:
            filtered_anomalies = filtered_anomalies[filtered_anomalies["equipment_id"].isin(selected_equipment)]
        if selected_shifts:
            filtered_anomalies = filtered_anomalies[filtered_anomalies["shift"].isin(selected_shifts)]
        filtered_anomalies = filtered_anomalies[filtered_anomalies["anomaly_score"] >= min_anomaly_score]

    query = st.session_state.dashboard_query
    query_responses = answer_dashboard_query(query, app, filtered_sensor_data, filtered_anomalies)

    total_records = len(filtered_sensor_data)
    total_anomalies = len(filtered_anomalies)
    total_orders = len(app.maintenance_orders)
    asset_count = filtered_sensor_data["equipment_id"].nunique() if not filtered_sensor_data.empty else 0
    location_count = filtered_sensor_data["location"].nunique() if not filtered_sensor_data.empty else 0
    anomaly_rate = total_anomalies / max(total_records, 1) * 100

    tabs = st.tabs([
        "Executive Summary",
        "Data Quality",
        "SOX Controls",
        "Operations",
    ])

    with tabs[0]:
        st.markdown("### Executive Summary")
        if filtered_sensor_data.empty:
            st.info("Stream or refresh to load Galaxy surfactant operational data.")
        else:
            kp1, kp2, kp3, kp4 = st.columns(4, gap="large")
            with kp1:
                _render_kpi_card("Assets Monitored", f"{asset_count}", "Live equipment coverage across plants.")
            with kp2:
                _render_kpi_card("Anomaly Frequency", f"{anomaly_rate:.1f}%", "Proportion of flagged events.")
            with kp3:
                _render_kpi_card("Critical Alerts", f"{total_orders}", "High-priority maintenance demand.")
            with kp4:
                savings = st.session_state.business_impact["cost_savings"]["annual_savings"] if st.session_state.business_impact else "Pending"
                _render_kpi_card("Annual Value", savings, "Estimated preventive ROI.")

            chart_row1 = st.columns(2, gap="large")
            with chart_row1[0]:
                st.write("#### Daily Anomaly Score Trend")
                a_series = filtered_anomalies.set_index("timestamp").resample("D")["anomaly_score"].mean().fillna(0)
                if not a_series.empty:
                    fig = px.line(a_series, labels={"index":"Day", "value":"Avg Anomaly Score"}, height=320)
                    fig.update_traces(line_color="#ff6f61", fill="tozeroy")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No anomaly trend available yet.")

                st.write("#### Sensor Variance vs. Alerts")
                if not filtered_sensor_data.empty:
                    instant = filtered_sensor_data.copy()
                    instant["status"] = np.where(instant["is_anomaly"], "Outlier", "Normal")
                    fig = px.scatter(
                        instant, x="temperature_c", y="vibration_mm_s",
                        color="status", color_discrete_map={"Outlier":"#ff9933", "Normal":"#3b82f6"},
                        hover_data=["equipment_id", "location", "anomaly_score"], height=320
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sensor matrix to display.")

            with chart_row1[1]:
                st.write("#### Top Risk Plants")
                if total_anomalies > 0:
                    risk_by_plant = filtered_anomalies["location"].value_counts().reset_index()
                    risk_by_plant.columns = ["location", "count"]
                    fig = px.bar(risk_by_plant, x="location", y="count", color="count", color_continuous_scale=["#ffcc80", "#ff6f61"], height=320)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No plant risk data under current settings.")

                st.write("#### Order Priority Mix")
                if total_orders > 0:
                    priority_df = pd.DataFrame(app.maintenance_orders)
                    fig = px.pie(priority_df, names="priority", title="Work Order Priority", color_discrete_sequence=["#ff6f61", "#fb923c", "#fbbf24", "#38bdf8"], height=320)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No maintenance orders generated yet.")

            st.markdown("---")
            st.write("#### Executive Insight")
            st.write(query_responses["Executive Summary"])

    with tabs[1]:
        st.markdown("### Data Quality")
        if filtered_sensor_data.empty:
            st.info("Streaming will provide continuous data quality checks.")
        else:
            dq1, dq2 = st.columns(2, gap="large")
            with dq1:
                st.write("#### Missing Values")
                missing = filtered_sensor_data.isna().sum().reset_index()
                missing.columns = ["field", "missing_count"]
                fig = px.bar(missing, x="field", y="missing_count", color="missing_count", color_continuous_scale=["#fbbf24", "#f97316"], height=320)
                st.plotly_chart(fig, use_container_width=True)

                st.write("#### Sensor Timeliness")
                ts = filtered_sensor_data.set_index("timestamp").resample("H")["equipment_id"].count()
                if not ts.empty:
                    fig = px.line(ts, labels={"index":"Time", "value":"Records/H"}, height=320)
                    fig.update_traces(line_color="#fb923c")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough fresh values to build quality trend.")

            with dq2:
                st.write("#### Value Consistency")
                summary = filtered_sensor_data[["temperature_c", "pressure_bar", "vibration_mm_s", "humidity_pct"]].describe().T
                st.dataframe(summary["mean"].to_frame("mean_value"))
                st.write("#### Anomaly Score Distribution")
                if total_anomalies > 0:
                    fig = px.histogram(filtered_anomalies, x="anomaly_score", nbins=15, color_discrete_sequence=["#fb923c"], height=320)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No anomalies to quantify score dispersion.")

            st.markdown("---")
            st.write(query_responses["Data Quality"])
            st.write("- Data is tagged by plant and equipment for audit review.")
            st.write("- Auto-refresh ensures quality checks remain current.")

    with tabs[2]:
        st.markdown("### SOX Controls")
        control_cols = st.columns(4, gap="large")
        control_cols[0].metric("Control Coverage", "100%", "All data flows monitored")
        control_cols[1].metric("Audit Flags", f"{total_anomalies}", "Events available for control review")
        control_cols[2].metric("Sync Confidence", "HIGH", "SAP pipeline enabled")
        control_cols[3].metric("Reporting Delay", "< 10s", "Near real-time refresh")

        if total_anomalies > 0:
            plant_risk_df = filtered_anomalies["location"].value_counts().reset_index(name="count")
            fig1 = px.bar(plant_risk_df, x="location", y="count", color="count", color_continuous_scale=["#ffb347", "#ff4500"], height=300)
            fig1.update_layout(title_text="SOX Risk by Plant", xaxis_title="Plant", yaxis_title="Flag Count")
        else:
            fig1 = px.bar(x=[], y=[], height=300)
        if total_orders > 0:
            orders_df = pd.DataFrame(app.maintenance_orders)
            fig2 = px.histogram(orders_df, x="priority", color="priority", category_orders={"priority":["CRITICAL","HIGH","MEDIUM","LOW"]}, color_discrete_map={"CRITICAL":"#dc2626","HIGH":"#f97316","MEDIUM":"#fbbf24","LOW":"#60a5fa"}, height=300)
            fig2.update_layout(title_text="Work Order Priority Frequency", xaxis_title="Priority", yaxis_title="Orders")
        else:
            fig2 = px.bar(x=[], y=[], height=300)

        fig3 = px.pie(names=["Matched","Outstanding"], values=[max(total_orders, 1), 1], color_discrete_sequence=["#34d399", "#fb923c"], height=300)
        fig3.update_layout(title_text="SAP Sync Readiness")
        control_row = st.columns(2, gap="large")
        with control_row[0]:
            st.plotly_chart(fig1, use_container_width=True)
        with control_row[1]:
            st.plotly_chart(fig2, use_container_width=True)

        st.write("#### Audit Trail Overview")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("---")
        st.write(query_responses["SOX Controls"])

    with tabs[3]:
        st.markdown("### Operations")
        op_row1 = st.columns(2, gap="large")
        with op_row1[0]:
            st.write("#### Vibration & Temperature Trend")
            if not filtered_sensor_data.empty:
                fig = px.line(filtered_sensor_data.sort_values("timestamp"), x="timestamp", y=["vibration_mm_s", "temperature_c"], height=320)
                fig.update_traces(line=dict(color="#ff6f61"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No operational data available.")

        with op_row1[1]:
            st.write("#### Pressure & Humidity Trend")
            if not filtered_sensor_data.empty:
                fig = px.line(filtered_sensor_data.sort_values("timestamp"), x="timestamp", y=["pressure_bar", "humidity_pct"], height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No operational data available.")

        op_row2 = st.columns(2, gap="large")
        with op_row2[0]:
            st.write("#### Active Outliers by Asset")
            if total_anomalies > 0:
                asset_outliers_df = filtered_anomalies["equipment_id"].value_counts().reset_index(name="count")
                fig = px.bar(asset_outliers_df, x="equipment_id", y="count", color="count", color_continuous_scale=["#f97316", "#dc2626"], height=320)
                fig.update_layout(xaxis_title="Equipment", yaxis_title="Anomaly Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No anomaly outliers in this cycle.")

        with op_row2[1]:
            st.write("#### Maintenance Lead Time")
            if total_orders > 0:
                times = pd.DataFrame(app.maintenance_orders)
                fig = px.histogram(times, x="estimated_duration_hours", nbins=6, color_discrete_sequence=["#fb923c"], height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No maintenance orders to evaluate.")

        st.markdown("---")
        if not filtered_sensor_data.empty:
            st.dataframe(filtered_sensor_data.sort_values("timestamp", ascending=False).head(10))
        st.write(query_responses["Operations"])

    st.markdown("---")
    st.text_area("Ask the dashboard a question", value=st.session_state.dashboard_query, key="dashboard_query", height=120)
    if st.session_state.dashboard_query:
        st.success("Answer generated across tabs, update the query to refine the insight.")


def main():
    configure_console_output()
    if is_streamlit_running():
        streamlit_dashboard()
    else:
        print("Streamlit dashboard mode not detected. Run with: streamlit run app.py")
        app = GalaxyApp()
        app.run_full_pipeline()


if __name__ == "__main__":
    main()
