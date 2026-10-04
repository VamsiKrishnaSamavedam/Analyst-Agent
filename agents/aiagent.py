# ruff: noqa: I001
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph

from agents.etl_analyst import DatasetETLAgent, etl_analyst
from agents.sql_analyst import LocalDatasetSQLAgent, sql_analyst_agent
from models.schema import AIAgentSchema, RouterSchema
from utils.llm_pick import pick_llm


llm = pick_llm("medium")

llm_router = llm.with_structured_output(RouterSchema)


# ---------------------------- AI AGENT GRAPH ---------------------------- #


def router_node(state:AIAgentSchema):

    message = state.messages[-1].content

    route_response_dict = llm_router.invoke(message).model_dump()

    route_response = route_response_dict['answer']

    state.route_response = route_response

    return state

def etl_node(state:AIAgentSchema):

    message = state.messages[-1].content

    response = etl_analyst.invoke(
             {"messages":[HumanMessage(content=f"""
            {message}
    """)]}
        ) 
    state.messages = state.messages + [response]

    return state

def sql_node(state:AIAgentSchema):

    message = state.messages[-1].content

    input_schema = {
        "messages": [],
        "user_question": f"{message}",
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    response = sql_analyst_agent.invoke(input_schema)

    state.messages = state.messages + [response]

    return state




data_agent_graph = StateGraph(AIAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")

def route_edge(state: AIAgentSchema) -> str:
    if state.route_response == "sql":
        return "sql_node"
    elif state.route_response == "etl":
        return "etl_node"
    else:
        raise ValueError(f"Invalid route response: {state.route_response}")


data_agent_graph.add_conditional_edges("router_node", route_edge,
                                      {
                                          "sql_node": "sql_node",
                                          "etl_node": "etl_node"
                                      })

data_agent = data_agent_graph.compile()


class ApplicationDataAgent:
    """Production coordinator built from the original ETL and SQL agents."""

    def __init__(self, gemini_api_key: str | None) -> None:
        self.etl = DatasetETLAgent(gemini_api_key)
        self.sql = LocalDatasetSQLAgent(gemini_api_key)

    def profile_dataset(self, frame):
        return self.etl.profile_dataset(frame)

    def understand_dataset(self, frame):
        return self.etl.understand_dataset(frame)

    def suggest_cleanup(self, frame):
        return self.etl.suggest_cleanup(frame)

    def apply_cleanup(self, frame, plan, approved_ids: list[str]):
        return self.etl.apply_approved_cleanup(frame, plan, approved_ids)

    def assess_load_readiness(self, frame, target_table, target_columns):
        return self.etl.assess_load_readiness(frame, target_table, target_columns)

    def reconcile_load(self, source_rows: int, target_rows: int, target_table: str):
        return self.etl.reconcile_load(source_rows, target_rows, target_table)

    def stage_raw_dataset(self, frame, source_name: str, source_format: str, database_config):
        return self.etl.stage_raw_dataset(frame, source_name, source_format, database_config)

    def analyze_dataset(self, frame, dataset_profile, question: str | None, sql: str | None):
        return self.sql.analyze(frame, dataset_profile, question, sql)



if __name__ == "__main__":

    response = data_agent.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
         "route_response": ""}
    )

    print(response)



