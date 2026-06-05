import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# LangChain Framework
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- 1. Define Live LangChain Tools using SerpApi ---
@tool
def search_flights(origin: str, destination: str, date_str: str) -> list:
    """
    Queries Google Flights via SerpApi for real-time flight options and pricing.
    Expects clean 3-letter IATA codes (e.g., 'BOM', 'DEL') and ISO date 'YYYY-MM-DD'.
    Returns a structured list of matching flights with pricing in INR, airlines, and departure times.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return [{"error": "SerpApi key not configured in .env file."}]

    try:
        # Construct the query payload for Google Flights targeting INR
        params = {
            "engine": "google_flights",
            "departure_id": origin.upper(),
            "arrival_id": destination.upper(),
            "outbound_date": date_str,
            "type": "2",  # 2 = One-way flight
            "currency": "INR",  # Indian Rupees
            "hl": "en",
            "api_key": api_key
        }
        
        # Execute the request
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        
        # Google Flights via SerpApi categorizes top results into 'best_flights'
        raw_flights = data.get("best_flights", [])
        
        parsed_flights = []
        # Limit to top 4 to save LLM context tokens and maintain speed
        for flight in raw_flights[:4]:
            flight_segments = flight.get("flights", [{}])
            if not flight_segments:
                continue
                
            first_segment = flight_segments[0]
            airline = first_segment.get("airline", "Unknown Airline")
            departure_time = first_segment.get("departure_airport", {}).get("time", date_str)
            price = flight.get("price", 0)
            
            parsed_flights.append({
                "airline": airline,
                "departure": departure_time,
                "price": int(price),  # Integers look cleaner for INR
                "currency": "INR"
            })
            
        return sorted(parsed_flights, key=lambda x: x["price"])
        
    except requests.exceptions.RequestException as error:
        print(f"SerpApi Request Error: {error}")
        return [{"error": f"Failed to retrieve live flights for date {date_str}."}]

# --- 2. Initialize LangChain Agent Core ---
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

tools = [search_flights]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are VoyageAI, an expert computational AI travel agent specializing in route optimization.
Your primary directive is cost and timeline optimization. When a user requests a flight path:
1. Run the `search_flights` tool for their requested primary target date.
2. Proactively run the `search_flights` tool for the day before AND the day after that target date to analyze market variance.
3. Compare the live pricing metrics (all prices are in Indian Rupees - INR).
4. Construct an extremely neat, pointwise analysis using clear Markdown. 

Your response MUST follow this exact structural breakdown:
### Analysis of Routes from {origin} to {destination}
* **Primary Date Analysis:** [Brief bullet point summarizing availability and lowest price found for the target date]
* **Surrounding Date Deltas:** [Compare prices with the day before and day after. State explicitly which day is cheaper and by how many Rupees]
* **Actionable Recommendation:** [Conclude with a definitive, bold recommendation on whether the user should book the primary date or shift their schedule to save capital]

Keep your thoughts concise, structured, and strictly data-driven."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. Flask Server Control Routing ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search", methods=["POST"])
def search():
    data = request.json or {}
    origin = data.get("origin", "").strip().upper()
    destination = data.get("destination", "").strip().upper()
    date_str = data.get("date", "").strip()
    
    if not all([origin, destination, date_str]):
        return jsonify({"error": "Parameters 'origin', 'destination', and 'date' are mandatory."}), 400

    user_query = f"Analyze routes from {origin} to {destination} departing on {date_str}. Execute delta comparisons on surrounding dates."
    
    try:
        # Run autonomous agent loop
        response = agent_executor.invoke({
            "input": user_query,
            "origin": origin,
            "destination": destination
        })
        ai_suggestion = response.get("output", "Analysis engine failed to compute recommendations.")
        
        # Pull primary raw flight array data for the frontend UI components
        primary_flights = search_flights.invoke({"origin": origin, "destination": destination, "date_str": date_str})
        
        if primary_flights and "error" in primary_flights[0]:
            return jsonify({"error": primary_flights[0]["error"]}), 502
            
    except Exception as e:
        print(f"System Runtime Error: {e}")
        return jsonify({"error": "An internal system error occurred processing your routing."}), 500

    return jsonify({
        "flights": primary_flights,
        "ai_suggestion": ai_suggestion
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)