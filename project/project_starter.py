import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

import os
from openai import OpenAI
from dotenv import load_dotenv

# Set up and load your env parameters and instantiate your model.
load_dotenv()

client = OpenAI(
    api_key=os.getenv("UDACITY_OPENAI_API_KEY", ""),
    base_url="https://openai.vocareum.com/v1"
)

MODEL_NAME = "gpt-4o-mini"

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

def call_llm(system_prompt, user_message, temperature=0.2):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role":"user", "content": user_message}
        ],
        temperature=temperature,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()

def calculate_bulk_discount(total_units):
    if total_units < 500:
        return 0.0
    elif total_units < 1000:
        return 0.05
    elif total_units < 5000:
        return 0.10
    else:
        return 0.15

def find_best_item_match(requested_name, inventory):
    requested_lower = requested_name.lower()
    for catalogue_name in inventory.keys():
        if (requested_lower in catalogue_name.lower() or catalogue_name.lower() in requested_lower):
            return catalogue_name
    return requested_name

# Tools for inventory agent

def inventory_agent(request, request_date):
    inventory_snapshot = get_all_inventory(request_date)
    inventory_list = "\n".join(
        f"  - {name}: {qty} units in stock"
        for name, qty in inventory_snapshot.items()
    ) or "  (no items currently in stock)"

    system_prompt = """You are an inventory specialist for Beaver's Choice Paper Company.
Analyse customer requests and check against available inventory.
Respond in this exact format:

ITEMS_REQUESTED:
- <item_name>: <quantity> units

ANALYSIS:
<brief analysis>"""

    user_message = f"""Customer Request: {request}

Current Inventory as of {request_date}:
{inventory_list}

List every item and quantity the customer is requesting."""

    llm_response = call_llm(system_prompt, user_message)

    available_items = []
    unavailable_items = []

    for line in llm_response.split("\n"):
        if line.strip().startswith("- ") and ":" in line:
            try:
                parts = line.strip("- ").split(":")
                item_raw = parts[0].strip()
                qty_str = parts[1].strip().split()[0].replace(",", "")
                qty_requested = int(qty_str)
                matched_name = find_best_item_match(item_raw, inventory_snapshot)
                stock_in_hand = inventory_snapshot.get(matched_name, 0)
                inv_df = pd.read_sql(
                    "SELECT unit_price FROM inventory WHERE item_name = :name",
                    db_engine, params={"name": matched_name},
                )
                unit_price = float(inv_df["unit_price"].iloc[0]) if not inv_df.empty else 0.0
                item_info = {
                    "item_name": matched_name,
                    "requested_qty": qty_requested,
                    "stock_in_hand": stock_in_hand,
                    "unit_price": unit_price,
                    "can_fulfill": stock_in_hand >= qty_requested,
                }
                if item_info["can_fulfill"]:
                    available_items.append(item_info)
                else:
                    item_info["supplier_delivery_date"] = get_supplier_delivery_date(
                        request_date, qty_requested
                    )
                    unavailable_items.append(item_info)
            except (IndexError, ValueError):
                continue

    return {
        "available_items": available_items,
        "unavailable_items": unavailable_items,
        "inventory_snapshot": inventory_snapshot,
        "summary": (f"{len(available_items)} item(s) available, "
                    f"{len(unavailable_items)} item(s) unavailable."),
    }

# Tools for quoting agent

def quoting_agent(request, request_date, inventory_result):
    available_items = inventory_result.get("available_items", [])
    unavailable_items = inventory_result.get("unavailable_items", [])

    if not available_items:
        return {
            "total_amount": 0.0, "discount_rate": 0.0,
            "discount_amount": 0.0, "final_amount": 0.0,
            "line_items": [], "quote_explanation": "No items available.",
            "can_fulfill": False,
        }

    search_terms = [item["item_name"].split()[0] for item in available_items[:3]]
    history = search_quote_history(search_terms, limit=3)
    history_text = "\n".join(
        f"  - {h.get('original_request','')[:80]}: ${h.get('total_amount',0):.2f}"
        for h in history
    ) or "  (no historical quotes found)"

    total_units = sum(i["requested_qty"] for i in available_items)
    discount_rate = calculate_bulk_discount(total_units)
    line_items = []
    subtotal = 0.0

    for item in available_items:
        line_price = item["requested_qty"] * item["unit_price"]
        subtotal += line_price
        line_items.append({
            "item_name": item["item_name"],
            "quantity": item["requested_qty"],
            "unit_price": item["unit_price"],
            "line_total": line_price,
        })

    discount_amount = subtotal * discount_rate
    final_amount = subtotal - discount_amount

    unavailable_note = ""
    if unavailable_items:
        unavailable_note = "UNAVAILABLE ITEMS:\n" + "\n".join(
            f"  - {i['item_name']}: only {i['stock_in_hand']} in stock"
            for i in unavailable_items
        )

    system_prompt = """You are a sales quotation specialist for Beaver's Choice Paper Company.
Write clear, friendly, transparent quote summaries.
Never reveal internal cost margins or system error messages.
Always justify discounts applied."""

    user_message = f"""Generate a quote explanation:

Request: {request}
Subtotal: ${subtotal:.2f}
Bulk Discount ({discount_rate*100:.0f}%): -${discount_amount:.2f}
Final Amount: ${final_amount:.2f}

Line Items:
{chr(10).join(f"  - {li['item_name']}: {li['quantity']} x ${li['unit_price']:.2f} = ${li['line_total']:.2f}" for li in line_items)}

{unavailable_note}

Historical Pricing:
{history_text}

Write a professional 2-3 sentence quote explanation for the customer."""

    quote_explanation = call_llm(system_prompt, user_message)

    return {
        "total_amount": subtotal, "discount_rate": discount_rate,
        "discount_amount": discount_amount, "final_amount": final_amount,
        "line_items": line_items, "quote_explanation": quote_explanation,
        "can_fulfill": True,
    }

# Tools for ordering agent
def sales_agent(request, request_date, quote_result, inventory_result):
    if not quote_result.get("can_fulfill", False):
        unavailable = inventory_result.get("unavailable_items", [])
        rejection = call_llm(
            "You are a helpful customer service agent for Beaver's Choice Paper Company. "
            "Write a polite message explaining why the order cannot be fulfilled. "
            "Do not reveal internal system details.",
            f"Request: {request}\n\nItems out of stock:\n" + "\n".join(
                f"  - {i['item_name']}: {i['stock_in_hand']} available, "
                f"{i['requested_qty']} requested"
                for i in unavailable
            ),
        )
        return {
            "fulfilled_items": [], "unfulfilled_items": unavailable,
            "total_charged": 0.0, "transaction_ids": [],
            "customer_message": rejection, "success": False,
        }

    fulfilled_items = []
    transaction_ids = []
    discount_factor = 1.0 - quote_result.get("discount_rate", 0.0)

    for item in quote_result.get("line_items", []):
        discounted_price = item["line_total"] * discount_factor
        txn_id = create_transaction(
            item_name=item["item_name"],
            transaction_type="sales",
            quantity=item["quantity"],
            price=discounted_price,
            date=request_date,
        )
        transaction_ids.append(txn_id)
        fulfilled_items.append({
            "item_name": item["item_name"], "quantity": item["quantity"],
            "unit_price": item["unit_price"], "charged": discounted_price,
            "txn_id": txn_id,
        })

    max_qty = max((i["quantity"] for i in quote_result.get("line_items", [])), default=1)
    delivery_date = get_supplier_delivery_date(request_date, max_qty)
    unfulfilled = inventory_result.get("unavailable_items", [])

    system_prompt = ("You are an order confirmation specialist for Beaver's Choice "
                     "Paper Company. Write clear, friendly confirmations. "
                     "Do not reveal profit margins or internal information.")
    user_message = f"""Write an order confirmation:

Request: {request}
Fulfilled: {chr(10).join(f"  - {i['item_name']}: {i['quantity']} units" for i in fulfilled_items)}
Not fulfilled: {chr(10).join(f"  - {i['item_name']}: insufficient stock" for i in unfulfilled) or "None"}
Total charged: ${quote_result['final_amount']:.2f}
Discount applied: {quote_result['discount_rate']*100:.0f}%
Estimated delivery: {delivery_date}

Write a warm 2-3 sentence confirmation."""

    customer_message = call_llm(system_prompt, user_message)

    return {
        "fulfilled_items": fulfilled_items, "unfulfilled_items": unfulfilled,
        "total_charged": quote_result["final_amount"],
        "transaction_ids": transaction_ids,
        "customer_message": customer_message, "success": True,
    }

# Set up your agents and create an orchestration agent that will manage them.
def orchestrator_agent(request_with_date):
    request_date = datetime.now().strftime("%Y-%m-%d")
    clean_request = request_with_date

    if "(Date of request:" in request_with_date:
        parts = request_with_date.split("(Date of request:")
        clean_request = parts[0].strip()
        try:
            request_date = parts[1].strip().rstrip(")").strip()
        except IndexError:
            pass

    print(f"\n[Orchestrator] Processing request dated {request_date}...")
    print("[Orchestrator] → Inventory Agent...")
    inventory_result = inventory_agent(clean_request, request_date)
    print(f"[Orchestrator]   {inventory_result['summary']}")

    print("[Orchestrator] → Quoting Agent...")
    quote_result = quoting_agent(clean_request, request_date, inventory_result)
    print(f"[Orchestrator]   Quote: ${quote_result['final_amount']:.2f}")

    print("[Orchestrator] → Sales Agent...")
    sales_result = sales_agent(clean_request, request_date, quote_result, inventory_result)
    print(f"[Orchestrator]   {'SUCCESS' if sales_result['success'] else 'REJECTED'}")

    fulfilled = sales_result.get("fulfilled_items", [])
    unfulfilled = sales_result.get("unfulfilled_items", [])

    sections = [sales_result["customer_message"]]

    if fulfilled:
        sections.append(
            f"\nORDER SUMMARY\n" +
            "\n".join(f"  - {i['item_name']}: {i['quantity']} units — ${i['charged']:.2f}"
                      for i in fulfilled) +
            f"\n  Total: ${sales_result['total_charged']:.2f}" +
            (f" ({quote_result['discount_rate']*100:.0f}% bulk discount applied)"
             if quote_result["discount_rate"] > 0 else "")
        )

    if unfulfilled:
        sections.append(
            "\nITEMS NOT AVAILABLE\n" +
            "\n".join(f"  - {i['item_name']}: only {i['stock_in_hand']} in stock "
                      f"(requested {i['requested_qty']})"
                      for i in unfulfilled)
        )

    if quote_result.get("quote_explanation"):
        sections.append(f"\n{quote_result['quote_explanation']}")

    return "\n".join(sections)

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = call_your_multi_agent_system(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

def call_your_multi_agent_system(request_with_date):
    return orchestrator_agent(request_with_date)

if __name__ == "__main__":
    results = run_test_scenarios()
