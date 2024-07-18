import yfinance as yf
from answerbot.chat import Chat

def get_nasdaq_price(ticker: str):
    """Get the current price of a NASDAQ ticker"""
    ticker = yf.Ticker(ticker)
    current_price = ticker.history(period="1d")['Close'].iloc[0]
    return current_price

# Create Chat instance
chat = Chat(
    model="gpt-3.5-turbo-1106",
    system_prompt="You are a helpful assistant that can look up stock prices.",
)

# Add user question
chat("What is the price of AAPL?", tools=[get_nasdaq_price])

# Process the response
results = chat.process()

# Print the results
for result in results:
    print(f"The price of AAPL is {result}")
