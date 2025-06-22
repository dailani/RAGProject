from langchain_core.prompts import PromptTemplate


REPHRASE_QUERY_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""
You are an assistant that processes user queries for a technical product database.
Your task is to:
1. Rephrase the query for vector search retrieval—keep it concise and focused.
2. Extract the product ID if it is mentioned in the user query.
3. Assign the query to ONE of the following headers, based on the property the user is asking about:

    - Areas of application
    - Product features and benefits
    - Technical Data
    - Photometric Data
    - Electrical Data
    - Physical Attributes
    - Operating Conditions
    - Lifetime Data
    - Environmental & Regulatory Information
    - Safety advice
    - Logistical Data

**Return your answer as a valid JSON object** with the following keys:
- "rephrased_query": [the simplified search query]
- "product_id": [the extracted product ID or null if not present]
- "header": [the best matching header from the list above]

User query: {input}
""",
)