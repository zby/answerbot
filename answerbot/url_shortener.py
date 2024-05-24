class UrlShortener:
    def __init__(self, min_length=50):
        self.url_map = {}
        self.short_url_map = {}
        self.base_url = "http://short.url/"
        self.counter = 0  # Initialize the counter
        self.min_length = min_length  # Minimum length to trigger URL shortening

    def _generate_short_url(self):
        """Generate a short URL using a hexadecimal encoded counter."""
        short_url = format(self.counter, 'x')  # Convert counter to hexadecimal
        self.counter += 1  # Increment the counter
        return short_url

    def shorten(self, original_url):
        """Shorten the original URL if it exceeds the minimum length."""
        if len(original_url) <= self.min_length:
            return original_url  # Return the original URL if it's already short enough

        if original_url in self.url_map:
            return self.base_url + self.url_map[original_url]
        else:
            short_url = self._generate_short_url()
            self.url_map[original_url] = short_url
            self.short_url_map[short_url] = original_url
            return self.base_url + short_url

    def retrieve(self, short_url):
        """Retrieve the original URL from the short URL. Return the original URL if not found."""
        short_key = short_url.replace(self.base_url, "")
        if short_key in self.short_url_map:
            return self.short_url_map[short_key]
        else:
            return short_url  # Return the input short_url as it is the original URL

# Example usage:
if __name__ == "__main__":
    url_shortener = UrlShortener(min_length=30)

    original_url = "https://www.example.com/some/very/long/url"
    short_url = url_shortener.shorten(original_url)
    print(f"Original URL: {original_url}")
    print(f"Shortened URL: {short_url}")

    retrieved_url = url_shortener.retrieve(short_url)
    print(f"Retrieved URL: {retrieved_url}")

    # Example with a short URL
    short_original_url = "http://ex.com/q"
    short_url = url_shortener.shorten(short_original_url)
    print(f"Short Original URL: {short_original_url}")
    print(f"Shortened URL: {short_url}")
    print(f"Retrieved URL: {url_shortener.retrieve('http://not.registered/url')}")
