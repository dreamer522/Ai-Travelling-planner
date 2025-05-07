#v-1.1
import os
import gradio as gr
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

# Define planner state
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "conversation history"]
    start_city: str
    destination: str
    days: int
    itinerary: str
    interests: str  # Adding interests as part of the state

# Define the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name='llama3-70b-8192'
)

# Prompt template for itinerary
itinerary_prompt = ChatPromptTemplate.from_messages([
    ('system', 
     'You are a helpful travel assistant. Create a multi-day travel itinerary for a trip from {start_city} to {destination} lasting {days} days. Include travel details, sightseeing spots, and food suggestions. Break down the plan day by day. If the user has shared interests, include them as well.'),
    ('human', 'Create my travel itinerary.')
])

# Core travel planning function
def generate_itinerary(start_city: str, destination: str, days: int, interests: str = ''):
    user_message = f"Plan my {days}-day trip from {start_city} to {destination}. Interests: {interests}"
    state: PlannerState = {
        "messages": [HumanMessage(content=user_message)],
        "start_city": start_city,
        "destination": destination,
        "days": days,
        "itinerary": "",
        "interests": interests
    }

    prompt_messages = itinerary_prompt.format_messages(
        start_city=start_city,
        destination=destination,
        days=days,
        interests=interests
    )

    response = llm.invoke(prompt_messages)

    state["messages"].append(AIMessage(content=response.content))
    state["itinerary"] = response.content

    return response.content

# Function to generate PDF
def generate_pdf(itinerary: str, filename="itinerary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200, 10, itinerary)
    pdf.output(filename)
    return filename

# Gradio Interface
def create_interface(start_city: str, destination: str, days: int, interests: str):
    itinerary = generate_itinerary(start_city, destination, days, interests)
    pdf_filename = generate_pdf(itinerary)
    return itinerary, pdf_filename

# Gradio UI setup
iface = gr.Interface(
    fn=create_interface,
    inputs=[
        gr.Textbox(label="Start Location"),
        gr.Textbox(label="Destination"),
        gr.Slider(minimum=1, maximum=15, step=1, label="Number of Days"),
        gr.Textbox(label="Your Interests (Optional)", placeholder="e.g., nature, adventure, culture")  # Removed optional=True
    ],
    outputs=[gr.Textbox(label="Day-wise Travel Itinerary", lines=20), gr.File(label="Download PDF")],
    title="🧳 Multi-Day Travel Planner",
    description="Enter your trip details and interests to get a full day-by-day itinerary. You can also download the itinerary as a PDF."
)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)