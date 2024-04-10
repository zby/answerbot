## URL or links
We can give to the LLM urls and let it retrieve it if it thinks it is useful, or give it links. When
presented with Markdown formatted links the LLM usually tries to follow the link text - this 
creates some problems because you can have multiple links with the same text - but links
are more universal because we can add links in places where there are no links in the original document.
We need to decide what is better. How difficult it will be to add this as another named option to the config
dictionary?

## Hierarchical documents
I want to have documents with structure - so that the LLM could move around it.
When retrieving a node we would let him know all the sub-nodes (chapters).
The listing should be adaptable - depending on how much space we have and how many sub-nodes there are
it should do the inteligent thing. If the chunk could not contain the whole list we should let the LLM
page through it. If there is enough space it should go as many levels deep as possible.
The LLM should be able to jump into any node by its title (maybe we should use links here - that would be 
pretty universal).
When searching the LLM could constrain the search to only sub-nodes of the current node.
read_chunk should jump to next section - if the current one is finished

We could also try to fit directories of documents into the same structure.

## PDF Parsing
We probably don't want to use PDFs in queries directly - we want batch scripts to parse them into Markdown documents.
But we might experiment with pages.

## Static chunking
Everyone chunks the documents in a static way - that is up-front before even the user query is received.
We currently try to find the best chunk that contains a position
in the document (for example after searching for a word) - maybe we need to go back to static chunking and use
more generic tools?

