import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
import chainlit as cl

# Load env
load_dotenv()

# Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Create client & model
client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)

# Define agents
arabic_translator = Agent(
    name="arabic_translator",
    instructions="You have to translate the user's input into Arabic.",
    model=model
)

french_translator = Agent(
    name="french_translator",
    instructions="You have to translate the user's input into French.",
    model=model
)

urdu_translator = Agent(
    name="urdu_translator",
    instructions="You have to translate the user's input into Urdu.",
    model=model
)

language_translator = Agent(
    name="translator",
    instructions="""You have to take the user's input and call the appropriate handoff to translate the user's input. 
    If you don't find the appropriate handoff then simply refuse the User.""",
    model=model,
    handoffs=[arabic_translator, french_translator, urdu_translator]
)

# ✅ This callback is required by Chainlit
@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(starting_agent=language_translator, input=message.content)
    await cl.Message(content=result.final_output).send()
