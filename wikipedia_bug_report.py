import wikipedia

try:
    # Try to get the 'Machine learning' page
    wikipedia.set_lang('en')
    page = wikipedia.page('Machine learning')
    print("Wikipedia Page Summary:")
    print(page.summary)
except Exception as e:
    print(f"Error: {str(e)}")

# Perform a search for 'Machine learning'
search_results = wikipedia.search('Machine learning')
if search_results:
    print("The first search result:")
    print(search_results[0])
    print("Trying again with that title")
    try:
        # Try to get the 'Machine learning' page
        wikipedia.set_lang('en')
        title = search_results[0]
        title = title.replace(" ", "_")
        page = wikipedia.page(title)
        print("Wikipedia Page Summary:")
        print(page.summary)
    except Exception as e:
        print(f"Error: {str(e)}")
else:
    print("No search results found for 'Machine learning'.")