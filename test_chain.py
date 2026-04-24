from moomoo import *
import pandas as pd

# Configuration
OPEND_ADDR = '127.0.0.1'
OPEND_PORT = 11111

def test_xsp_chain():
    # Initialize Context
    quote_ctx = OpenQuoteContext(host=OPEND_ADDR, port=OPEND_PORT)
    
    print(f"📡 Querying official Moomoo Option Chain for US.XSP...")
    
    # We query for any PUTs in the next few days to see the naming convention
    # Adjust dates if today is a weekend
    ret, data = quote_ctx.get_option_chain(
        code='US..XSP', 
        start='2026-04-21', 
        end='2026-04-22', 
        option_type=OptionType.PUT
    )

    if ret == RET_OK:
        print(f"✅ Success! Found {len(data)} contracts.")
        print("\n--- Top 10 Symbols from Official Chain ---")
        # We look at the 'code' and 'strike_price' columns
        preview = data[['code', 'strike_price', 'strike_time']].head(10)
        print(preview.to_string(index=False))
        
        # Identify the Strike multiplier
        sample_code = data['code'].iloc[0]
        sample_strike = data['strike_price'].iloc[0]
        print(f"\nAnalysis:")
        print(f"Symbol: {sample_code}")
        print(f"Actual Strike: {sample_strike}")
    else:
        print(f"❌ Failed to get chain: {data}")

    quote_ctx.close()

if __name__ == "__main__":
    test_xsp_chain()
