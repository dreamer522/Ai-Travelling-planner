#v-1
import os
import gradio as gr
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Define PlannerState
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "conversation history"]
    city: str
    interests: List[str]
    itinerary: str

# Define LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name='llama-3.3-70b-versatile'
)

# Prompt for itinerary
itinerary_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user\'s interests: {interests}. Provide a brief, bulleted itinerary.'),
    ('human', 'Create an itinerary for my day trip.'),
])

# Main generation function
def generate_itinerary(user_request: str, city: str, interests_input: str):
    interests = [i.strip() for i in interests_input.split(",") if i.strip()]
    
    state: PlannerState = {
        "messages": [HumanMessage(content=user_request)],
        "city": city,
        "interests": interests,
        "itinerary": ""
    }

    response = llm.invoke(itinerary_prompt.format_messages(
        city=state["city"],
        interests=", ".join(state["interests"])
    ))

    state["messages"].append(AIMessage(content=response.content))
    state["itinerary"] = response.content

    return state["itinerary"]

# Gradio UI
iface = gr.Interface(
    fn=generate_itinerary,
    inputs=[
        gr.Textbox(label="Initial Request", placeholder="e.g., I want to plan a road trip"),
        gr.Textbox(label="City", placeholder="e.g., Paris"),
        gr.Textbox(label="Your Interests (comma-separated)", placeholder="e.g., food, museums, nightlife")
    ],
    outputs=gr.Textbox(label="Day Trip Itinerary", lines=20),
    title="🗺️ Trip Planner",
    description="Enter your request, the city you want to visit, and your interests. The assistant will generate a one-day itinerary for you."
)

if __name__ == "__main__":
    iface.launch()