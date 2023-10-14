import wikipedia
import sys
import re

from document import WikipediaDocument

query = "Ukrainian War"

wikipages = wikipedia.search(query)
if len(wikipages) == 0:
    print('No relevant wikipedia articles found')
    sys.exit(0)

print(f'Pages found:', ", ".join(wikipages))
print(f'Getting the contents of "{wikipages[0]}"')

page = None

for result in wikipages:
    try:
        page = wikipedia.page(result).content
        break
    except wikipedia.DisambiguationError as de:
        # Handle disambiguation pages by attempting to get the first option
        try:
            page = wikipedia.page(de.options[0]).content
            break
        except:
            continue
    except wikipedia.exceptions.PageError:
        # If the page doesn't exist, move to the next result
        continue

#print(page)
document = WikipediaDocument(page)

print(document.first_chunk())
#section_list = "\n".join(list(map(lambda section: f' - {section}', sections)))

