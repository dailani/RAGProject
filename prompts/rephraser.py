from langchain_core.prompts import PromptTemplate

MULTI_HEADER_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""
You are an assistant for a technical product database.

Your task:
- Analyze the user's query.
- Identify ALL relevant headers (not just one) from the list below, based on the properties or attributes mentioned in the query. Use the property-to-header mapping below.

Headers and Example Properties:
- Areas of application: stage, studio, film, TV, photography, club, disco
- Product features and benefits: robust construction, color consistency, instant on, dimmable, portfolio breadth
- Technical Data: product number, product name, lamp type, ANSI code, global order reference
- Electrical Data: nominal wattage, nominal voltage, type of current
- Photometric Data: luminous flux, luminous efficacy, illuminated field, color temperature, CCT, chromaticity, color rendering index, light center length (LCL)
- Physical Attributes: lamp base, diameter, length, product weight
- Operating Conditions: burning position, dimmable, cooling
- Lifetime Data: nominal lifetime
- Environmental & Regulatory Information: REACh, article identifier, energy efficiency, declaration in SCIP, candidate substances
- Safety advice: operation warnings, heat warning, UV protection, protections
- Logistical Data: product code, packaging, dimensions, volume, weight, shipping/carton info

### Examples
User query: color temperature of SIRIUS HRI 330W 2/CS 1/SKU  
Response:  
[
    {{"header": "Photometric Data"}}
]

User query: Give me all light bulbs with at least 1000 watts and a lifespan of more than 400 hours.  
Response:  
[
    {{"header": "Electrical Data"}},
    {{"header": "Lifetime Data"}}
]

**Return ONLY a valid JSON array** of objects in this format (each object is one header):  
[
    {{"header": "Header Name"}},
    {{"header": "Header Name"}},
    ...
]


- Include every header relevant to the user's query.
- If only one header is relevant, return an array with just one element.

User query: {input}
""",
)


VECTOR_REPHRASER_QUERY_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""You are an assistant tasked with taking a natural languge query from a user
    and converting it into a query for a vectorstore. In the process, strip out all 
    information that is not relevant for the retrieval task and return a new, simplified
    question for vectorstore retrieval.
    Here is the user query: {input} """,
)