import pandas as pd

def load_and_clean_data(path):
    # ✅ FIX: correct separator (your file uses comma)
    df = pd.read_csv(path, sep=',', low_memory=False)

    # Clean column names
    df.columns = df.columns.str.strip()

    print("Columns found:", df.columns.tolist())

    # Handle missing values
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Create DateTime column
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )

    # Set index
    df.set_index('DateTime', inplace=True)

    # Convert numeric columns
    df['Global_active_power'] = df['Global_active_power'].astype(float)

    return df