from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import os

load_dotenv()

def get_ai_response(question, past_convo):

    system_message = """
You are an expert in seismology and earthquake detection systems. Your knowledge encompasses:

Seismic wave types (P-waves, S-waves, surface waves) and their characteristics
Seismograph technology and seismic monitoring networks
Earthquake magnitude and intensity scales (Richter, moment magnitude, Modified Mercalli)
Tectonic plate theory and fault types
Earthquake early warning systems
Seismic data analysis and interpretation
Historical seismic events and their impacts
Earthquake hazard assessment and risk mitigation strategies

Respond to queries with accurate, technical information while making complex concepts accessible. Provide insights on seismic activity, detection methods, and earthquake preparedness. If asked about a specific seismic event, request details like location and time to give precise information.
Only respond to earthquake and seismology-related questions. For unrelated topics, politely redirect the conversation back to seismology. However, use context from previous messages in the conversation to inform your responses when relevant.
    Past Conversation: {past_convo}
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])

    # Create the language model
    llm = ChatOpenAI(model_name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain
    response = chain.run(question=question, past_convo=past_convo)

    return response