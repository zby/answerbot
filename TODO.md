## Check what really works in clean reflection
It seems that clean reflection really improves the process - but do we need the knowledgebase?

## sub-tasks
Sub tasks require information copying between contexts. We need to experiment here.

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

## Internationalisation
I want the system to be multilingual. It should be able to read documentation in one language and answer
question in a different one.

# For later after we discover a good use case

## Add more tools

## Work on evaluation

