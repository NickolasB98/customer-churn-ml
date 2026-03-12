"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the E-Commerce Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="E-Commerce Customer Churn Prediction API",
    description="ML API for predicting customer churn in e-commerce",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer data schema for e-commerce churn prediction.

    This schema defines the exact 19 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    # Demographics
    Gender: str                          # "Male" or "Female"
    MaritalStatus: str                   # "Single", "Married", or "Divorced"

    # Engagement & Activity
    Tenure: float                        # Months as customer
    HourSpendOnApp: float                # Hours spent on app (0-5)
    NumberOfDeviceRegistered: int        # Number of devices registered
    NumberOfAddress: int                 # Number of addresses on file

    # Location & Logistics
    CityTier: int                        # City tier (1, 2, or 3)
    WarehouseToHome: float               # Distance from warehouse (km)

    # Purchase Behavior
    PreferedOrderCat: str                # Preferred product category
    OrderCount: float                    # Total number of orders
    OrderAmountHikeFromlastYear: float   # % increase in order amount
    CouponUsed: float                    # Number of coupons used
    DaySinceLastOrder: float             # Days since last order
    CashbackAmount: float                # Total cashback received

    # Service Preferences
    PreferredLoginDevice: str            # "Mobile Phone", "Phone", or "Computer"
    PreferredPaymentMode: str            # "Debit Card", "UPI", "CC", "Cash on Delivery", "E wallet"

    # Satisfaction & Feedback
    SatisfactionScore: int               # Satisfaction score (1-5)
    Complain: int                        # Whether customer complained (0 or 1)


# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.

    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format

    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


# =================================================== #


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    Gender, MaritalStatus, Tenure, HourSpendOnApp, NumberOfDeviceRegistered,
    NumberOfAddress, CityTier, WarehouseToHome, PreferedOrderCat, OrderCount,
    OrderAmountHikeFromlastYear, CouponUsed, DaySinceLastOrder, CashbackAmount,
    PreferredLoginDevice, PreferredPaymentMode, SatisfactionScore, Complain
):
    """
    Gradio interface function that processes form inputs and returns prediction.

    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string

    """
    # Construct data dictionary matching CustomerData schema
    data = {
        "Gender": Gender,
        "MaritalStatus": MaritalStatus,
        "Tenure": float(Tenure),
        "HourSpendOnApp": float(HourSpendOnApp),
        "NumberOfDeviceRegistered": int(NumberOfDeviceRegistered),
        "NumberOfAddress": int(NumberOfAddress),
        "CityTier": int(CityTier),
        "WarehouseToHome": float(WarehouseToHome),
        "PreferedOrderCat": PreferedOrderCat,
        "OrderCount": float(OrderCount),
        "OrderAmountHikeFromlastYear": float(OrderAmountHikeFromlastYear),
        "CouponUsed": float(CouponUsed),
        "DaySinceLastOrder": float(DaySinceLastOrder),
        "CashbackAmount": float(CashbackAmount),
        "PreferredLoginDevice": PreferredLoginDevice,
        "PreferredPaymentMode": PreferredPaymentMode,
        "SatisfactionScore": int(SatisfactionScore),
        "Complain": int(Complain),
    }

    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display

# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface with all customer features
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Demographics section
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Single", "Married", "Divorced"], label="Marital Status", value="Single"),

        # Engagement & Activity section
        gr.Number(label="Tenure (months)", value=10, minimum=0, maximum=61),
        gr.Number(label="Hour Spend On App", value=2.5, minimum=0, maximum=5),
        gr.Number(label="Number of Device Registered", value=2, minimum=1, maximum=6),
        gr.Number(label="Number of Address", value=3, minimum=1, maximum=22),

        # Location & Logistics section
        gr.Dropdown(["1", "2", "3"], label="City Tier", value="2"),
        gr.Number(label="Warehouse to Home (km)", value=30, minimum=5, maximum=127),

        # Purchase Behavior section
        gr.Dropdown([
            "Laptop & Accessory", "Mobile", "Mobile Phone",
            "Others", "Fashion", "Grocery"
        ], label="Preferred Order Category", value="Laptop & Accessory"),
        gr.Number(label="Order Count", value=5, minimum=1, maximum=16),
        gr.Number(label="Order Amount Hike From Last Year (%)", value=15, minimum=11, maximum=26),
        gr.Number(label="Coupon Used", value=2, minimum=0, maximum=16),
        gr.Number(label="Days Since Last Order", value=10, minimum=0, maximum=46),
        gr.Number(label="Cashback Amount ($)", value=100, minimum=0, maximum=324.99),

        # Service Preferences section
        gr.Dropdown(["Mobile Phone", "Phone", "Computer"], label="Preferred Login Device", value="Mobile Phone"),
        gr.Dropdown([
            "Debit Card", "UPI", "CC", "Cash on Delivery", "E wallet"
        ], label="Preferred Payment Mode", value="Debit Card"),

        # Satisfaction & Feedback section
        gr.Dropdown(["1", "2", "3", "4", "5"], label="Satisfaction Score (1-5)", value="4"),
        gr.Dropdown(["0", "1"], label="Customer Complained (0=No, 1=Yes)", value="0"),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🛒 E-Commerce Customer Churn Predictor",
    description="""
    **Predict customer churn probability using machine learning**

    Fill in the customer details below to get a churn prediction. The model uses XGBoost trained on
    historical e-commerce customer data to identify customers at risk of churning.

    💡 **Tip**: Customers with low satisfaction scores, long gaps since last order, and few devices
    registered tend to have higher churn rates.
    """,
    examples=[
        # High churn risk example
        ["Female", "Single", 5, 1.0, 1, 3, 3, 50, "Mobile", 2, 12, 0, 30, 50,
         "Mobile Phone", "Debit Card", 2, 1],
        # Low churn risk example
        ["Male", "Married", 40, 4.0, 4, 8, 1, 20, "Laptop & Accessory", 12, 20, 5, 5, 200,
         "Computer", "CC", 5, 0]
    ],
    theme=gr.themes.Soft()  # Professional appearance
)

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)
