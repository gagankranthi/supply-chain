import sqlite3
import pandas as pd

EXCEL_PATH = "Supply chain logistics problem.xlsx"
DB_PATH = "supply_chain.db"

def create_tables(conn):
    cur = conn.cursor()
    # Example schema - adjust based on actual columns
    cur.execute('''
    CREATE TABLE IF NOT EXISTS shipments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT,
        origin TEXT,
        destination TEXT,
        product TEXT,
        quantity REAL,
        ship_date TEXT
    )
    ''')
    conn.commit()

def populate_from_excel(conn, df):
    # Try to map likely columns to shipments table
    cols = df.columns.str.lower()
    df2 = pd.DataFrame()
    def first_column(df, candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return None

    df2['order_id'] = first_column(df, ['Order ID', 'order id', 'OrderID', 'Order'])
    df2['origin'] = first_column(df, ['Origin', 'origin', 'From'])
    df2['destination'] = first_column(df, ['Destination', 'destination', 'To'])
    # take first string-like column as product
    product_col = None
    for c in df.columns:
        if df[c].dtype == 'object' and c.lower() not in ['origin','destination','order id','order']:
            product_col = c
            break
    if product_col:
        df2['product'] = df[product_col]
    else:
        df2['product'] = None
    # quantity
    qty = None
    for c in df.columns:
        if 'qty' in c.lower() or 'quantity' in c.lower() or 'weight' in c.lower():
            qty = c
            break
    if qty:
        df2['quantity'] = pd.to_numeric(df[qty], errors='coerce')
    else:
        df2['quantity'] = None
    # ship_date
    date_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'ship' in c.lower():
            date_col = c
            break
    if date_col:
        df2['ship_date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df2['ship_date'] = None

    df2.to_sql('shipments', conn, if_exists='append', index=False)

if __name__ == '__main__':
    df = pd.read_excel(EXCEL_PATH)
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    populate_from_excel(conn, df)
    print('DB created at', DB_PATH)
