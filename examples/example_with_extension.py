import yfinance as yf
from dataclasses import dataclass
from answerbot.chat import Chat, Prompt, SystemPrompt
from jinja2 import Environment, DictLoader

@dataclass(frozen=True)
class UserPrompt(Prompt):
    tickers: list[str]

@dataclass(frozen=True)
class TickerPrices(Prompt):
    ticker_prices: dict[str, float]

def join_with_and(value):
    if not value:
        return ""
    elif len(value) == 1:
        return value[0]
    else:
        return ', '.join(value[:-1]) + ' and ' + value[-1]

ticker_tempalte = """
# Nasdaq Ticker Prices

| Ticker | Price  |
|--------|--------|
{%- for symbol, price in ticker_prices.items() %}
| {{ symbol|center(6) }} |${{ "%.2f"|format(price) }} |
{%- endfor %}
"""

# Define templates
templates = {
    "SystemPrompt": "You are a helpful assistant that can look up stock prices.",
    "UserPrompt": "What are the prices of {{tickers|join_with_and}}?",
    "TickerPrices": ticker_tempalte,
}

# Create Jinja2 Environment with HumanizeExtension and templates
env = Environment(
    loader=DictLoader(templates))
env.filters['join_with_and'] = join_with_and


def get_nasdaq_price(ticker: str):
    """Get the current price of a NASDAQ ticker"""
    ticker = yf.Ticker(ticker)
    current_price = ticker.history(period="1d")['Close'].iloc[0]
    return current_price

def get_ticker_prices(tickers: list[str]):
    return TickerPrices({ticker: get_nasdaq_price(ticker) for ticker in tickers})

# Create Chat instance
chat = Chat(
    model="gpt-3.5-turbo-1106",
    system_prompt=SystemPrompt(),
    template_env=env
)

# Add user question
chat(UserPrompt(tickers=["AAPL", "GOOGL", "MSFT"]), tools=[get_ticker_prices])

# Process the response
results = chat.process()

# Print the results
for result in results:
    print(result)


# OUTPUT

#
## Nasdaq Ticker Prices
#
#| Ticker | Price  |
#|--------|--------|
#|  AAPL  |$228.88 |
#| GOOGL  |$181.02 |
#|  MSFT  |$443.52 |