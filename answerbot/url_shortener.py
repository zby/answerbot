import random
import string


class URLShortener:
    def __init__(self):
        self.url_map = {}
        self.short_url_map = {}
        self.base_url = "http://short.url/"
        self.short_url_length = 6

    def _generate_short_url(self):
        """Generate a random short URL."""
        chars = string.ascii_letters + string.digits
        short_url = ''.join(random.choice(chars) for _ in range(self.short_url_length))
        return short_url

    def shorten_url(self, original_url):
        """Shorten the original URL."""
        if original_url in self.url_map:
            short_url = self.url_map[original_url]
        else:
            short_url = self._generate_short_url()
            while short_url in self.short_url_map:
                short_url = self._generate_short_url()
            self.url_map[original_url] = short_url
            self.short_url_map[short_url] = original_url

        return self.base_url + short_url

    def retrieve_url(self, short_url):
        """Retrieve the original URL from the short URL."""
        short_url = short_url.replace(self.base_url, "")
        return self.short_url_map.get(short_url)


# Example usage:
if __name__ == "__main__":
    url_shortener = URLShortener()

    original_url = "https://www.example.com/some/very/long/url"
    short_url = url_shortener.shorten_url(original_url)

    print(f"Original URL: {original_url}")
    print(f"Shortened URL: {short_url}")

    retrieved_url = url_shortener.retrieve_url(short_url)
    print(f"Retrieved URL: {retrieved_url}")
