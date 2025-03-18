Large Model-Based Document Retrieval and Question Answering System
Task:
The project aims to develop a question-answering system centered around a large model to respond to user inquiries about automobiles. The system must locate relevant information in documents based on the question and generate an appropriate answer using the large model. The questions primarily focus on automobile usage, maintenance, and repairs, as illustrated in the examples below:

Question 1: How do I turn on the hazard warning lights?
Answer 1: The hazard warning light switch is located below the steering wheel. Press the switch to turn on the hazard lights.

Question 2: How should I maintain my vehicle?
Answer 2: To keep your vehicle in optimal condition, regularly monitor its status, including routine maintenance, car washes, interior and exterior cleaning, tire maintenance, and low-voltage battery care.

Question 3: What should I do if my seatback feels too hot?
Answer 3: If your seatback is too hot, you can try turning off the seat heating function. On the multimedia display screen, tap the air conditioning activation button → Seats → Heating, then disable seat heating from this interface.

raining Data:
The project uses Lynk & Co car user manuals as the training dataset.

Test Set Questions:
A set of automobile-related questions will be used to evaluate the system’s performance.

3. Solution Approach
3.1 PDF Parsing
3.1.1 Block-Based Parsing
To preserve textual integrity, the PDF is parsed in blocks, ensuring that each section includes a subheading and its corresponding content.

3.1.2 Sliding Window Parsing
A sliding window method ensures that content spanning across pages remains continuous. This technique processes the PDF as a single string, splits it by punctuation marks, and applies a sliding window.

For instance, given the text:
["aa", "bb", "cc", "dd"]
Using a window size of 2, the parsed segments would be:

aabb
bbcc
ccdd
This method ensures optimal text retrieval, preventing key information from being fragmented or overlooked.

3.1.3 Final Parsing Strategy
The project employs a hybrid approach combining three parsing methods:

Block-Based Parsing: Extracting sections based on subheadings, maintaining content length of 512 and 1024 tokens.
Sliding Window Parsing: Splitting text into sentences, applying a sliding window with lengths of 256 and 512 tokens.
Fixed-Size Segmentation: Splitting text based on predefined chunk sizes of 256 and 512 tokens.
After processing, redundant blocks are removed before feeding them into the retrieval module.

3.2 Retrieval
The retrieval system employs LangChain’s retrievers for text retrieval, combining vector search (FAISS) and BM25.

Vector retrieval captures deep semantic meaning, ensuring broad generalization.
BM25 retrieval focuses on keyword matching, improving lexical accuracy.
3.2.1 Vector Retrieval
FAISS (Facebook AI Similarity Search) is used for indexing and searching embeddings generated by M3E-large.

M3E (Moka Massive Mixed Embedding) is a large-scale embedding model that:

Is trained on 22M+ Chinese sentence pairs.
Supports bilingual (Chinese-English) text similarity and retrieval.
Comes in three sizes: small, base, large.
The project utilizes M3E-large for enhanced accuracy.
3.2.2 BM25 Retrieval
BM25 is a ranking function widely used in search engines for text relevance scoring. It calculates the importance of query terms in a document and assigns a weighted score. BM25 retrieval is implemented using LangChain's BM25 retriever.

3.3 Re-ranking
A re-ranker fine-tunes search results by improving the order of retrieved documents, ensuring that the most relevant ones are prioritized. The bge-reranker-large model is used, which follows a cross-encoder architecture.

Why Use Re-ranking?
Basic vector retrieval (bi-encoder) focuses on approximate matches.
Cross-encoder-based re-ranking significantly improves relevance by directly comparing the query and retrieved documents.


3.3.1 Cross-Encoder Re-ranking
The project adopts bge-reranker-large, an open-source re-ranking model that rivals commercial models like Cohere Reranker in accuracy.

Workflow:

Retrieve Top-K results from vector and sparse retrieval (BM25).
Re-rank the results using bge-reranker-large to obtain the most relevant documents.
Feed the top-ranked documents into the language model for final answer generation.
Challenges and Trade-offs
Re-ranking improves accuracy but increases computation costs and latency.
Requires a balance between search quality, response time, and cost efficiency.
