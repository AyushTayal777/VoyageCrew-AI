import os
from typing import TypedDict, Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_results:str
    hotel_results:str
    itinerary:str
    llm_calls:int

def flight_agent(state:TravelState):
    query=state['user_query']
    flight_data=search_flights(query)
    return {
        "flight_results":flight_data,
        "messages":[AIMessage(content='Flights fetched successfully')]
    }

def hotel_agent(state:TravelState):
    query=f'Best hotels for {state['user_query']}'
    hotel_data=tavily_search(query)
    return {
        "hotel_results":hotel_data,
        "messages":[AIMessage(content='Hotels fetched successfully')]
    }

def itinerary_agent(state:TravelState):
    prompt=f"""
    Create a travel itinerary for the following:
    User Query: {state['user_query']},
    Flight results: {state['flight_results']},
    Hotel_results:{state['hotel_results']}
    """

    response=llm.invoke([
        SystemMessage(
            content="You are a expert travel planner"
        ),
        HumanMessage(
            content=prompt
        )
    ])

    return {
        "itinerary":response.content,
        "messages":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }

def final_agent(state:TravelState):
    final_prompt=f"""
    Generate final travel response for the following:
    User Query: {state['user_query']},
    Flight results: {state['flight_results']},
    Hotel_results:{state['hotel_results']},
    Itinerary: {state['itinerary']} 
    """

    response=llm.invoke([
            HumanMessage(content=final_prompt)
            ])

    return {
        "messages":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }

graph =StateGraph(TravelState)

graph.add_node("flight_agent",flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_node("itinerary_agent",itinerary_agent)
graph.add_node("final_agent",final_agent)

graph.add_edge(START,"flight_agent")
graph.add_edge("flight_agent","hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent",END)

_conn=psycopg.connect(DATABASE_URL,autocommit=True)
checkpointer=PostgresSaver(_conn)
checkpointer.setup()

app=graph.compile(checkpointer=checkpointer) 


if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "user_aarohi"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)