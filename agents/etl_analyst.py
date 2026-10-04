import os
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from models.schema import ETLAgentSchema

from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langchain.tools import tool


# ============================================================
# AGENT TOOL 1
# EXTRACT + LOAD
# ============================================================

@tool
def extract_load_tool(
    url: str,
    output_folder: str,
    output_format: str
) -> str:
    """
    Extract data from an API endpoint and save it locally.

    Args:
        url:
            API endpoint.

        output_folder:
            Destination folder.

        output_format:
            csv, json, or parquet.

    Returns:
        Operation result.
    """

    etl_tools = ETLTools()

    return etl_tools.extract_load(
        url=url,
        output_folder=output_folder,
        output_format=output_format
    )


# ============================================================
# AGENT TOOL 2
# TRANSFORM + LOAD
# ============================================================

@tool
def transform_load_tool(
    input_file_path: str,
    output_folder: str,
    output_format: str,
    output_file_name: str,
    user_question: str
) -> str:
    """
    Transform an existing dataset using Pandas and save
    the result to another file.

    Args:
        input_file_path:
            Path to source file.

        output_folder:
            Destination folder.

        output_format:
            csv, json, or parquet.

        output_file_name:
            Name of output file.

        user_question:
            Description of transformation.

    Returns:
        Transformation result.
    """

    etl_tools = ETLTools()

    # --------------------------------------------------------
    # Get source-data context
    # --------------------------------------------------------

    data_context = (
        etl_tools.transform_load_context(
            input_file_path
        )
    )

    # --------------------------------------------------------
    # Use Gemini to generate Pandas code
    # --------------------------------------------------------

    llm = (
        pick_llm("medium")
        | StrOutputParser()
    )

    prompt = f"""
You are an expert Python ETL engineer.

Your task is to generate executable Python Pandas code
that performs the requested ETL transformation.

Return ONLY executable Python code.


============================================================
SOURCE FILE
============================================================

{input_file_path}


============================================================
SOURCE DATA INFORMATION
============================================================

{data_context}


============================================================
USER TRANSFORMATION REQUEST
============================================================

{user_question}


============================================================
OUTPUT REQUIREMENTS
============================================================

Output folder:

{output_folder}

Output file name:

{output_file_name}

Output format:

{output_format}


============================================================
STRICT RULES
============================================================

1. Use Pandas.

2. Read data only from:

   {input_file_path}

3. Do not overwrite or modify the source file.

4. Apply exactly the transformation requested by the user.

5. Preserve all original columns unless the user explicitly
   requests columns to be removed.

6. Create the destination folder when necessary using:

   os.makedirs(output_folder, exist_ok=True)

7. Save the transformed dataset inside:

   {output_folder}

8. Save it with the exact file name:

   {output_file_name}

9. Use the requested file format:

   {output_format}

10. For text filtering, use case-insensitive matching when
    the user's intention indicates that capitalization should
    not matter.

11. Handle missing values safely when applying string operations.

For example:

    df["name"].fillna("").str.lower()

12. Use only columns that exist in the provided source-data
    information.

13. Do not invent column names.

14. Import all Python packages required by the code.

15. Use os.path.join() to construct the output path.

16. Do not include explanation.

17. Do not include comments outside the Python code.

18. Do not use Markdown.

19. Do not return ```python.

20. Do not return ```.

21. Return only executable Python code.

Generate the Pandas code now.
"""

    # --------------------------------------------------------
    # Generate code
    # --------------------------------------------------------

    pandas_code = llm.invoke(prompt)

    # --------------------------------------------------------
    # Clean possible Markdown
    # --------------------------------------------------------

    pandas_code = (
        pandas_code
        .replace("```python", "")
        .replace("```py", "")
        .replace("```", "")
        .strip()
    )

    # --------------------------------------------------------
    # Display generated code
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("GENERATED PANDAS CODE")
    print("=" * 80)
    print(pandas_code)
    print("=" * 80)
    print("\n")

    return f"""
ETL transformation was not executed.

Source:
{input_file_path}

Output Folder:
{output_folder}

Output File:
{output_file_name}

Output Format:
{output_format}

Generated Pandas Code:
{pandas_code}

For safety, model-generated Python is never executed. Use the reviewable
dataset workflow to apply approved deterministic cleanup actions.

Execution Result:
{execution_result}
""".strip()


# ============================================================
# TOOLKIT
# ============================================================

tools = [
    extract_load_tool,
    transform_load_tool
]


# ============================================================
# MAIN AGENT MODEL
# ============================================================

# Medium is sufficient for tool selection and saves quota.
agent_llm = pick_llm("medium")

llm_bind = agent_llm.bind_tools(tools)


# ============================================================
# LANGGRAPH NODE 1
# LLM DECISION NODE
# ============================================================

def llm_node(
    state: ETLAgentSchema
) -> ETLAgentSchema:
    """
    Determines which ETL tool should be called.
    """

    messages = state.messages

    system_instruction = HumanMessage(
        content="""
You are an ETL Analyst Agent.

Your job is to understand the user's ETL request
and use the available tools.

You have two tools:

1. extract_load_tool

Use this when the user wants to:
- call an API
- extract external data
- save API data into a local file


2. transform_load_tool

Use this when the user wants to:
- read an existing local file
- filter data
- clean data
- transform data
- aggregate data
- rename or manipulate columns
- save transformed data to another file


IMPORTANT:

For transform_load_tool, identify:

- input_file_path
- output_folder
- output_format
- output_file_name
- complete transformation request

Do not invent paths if the user has supplied them.

When a tool successfully completes the requested operation,
provide a concise final response to the user.
"""
    )

    model_messages = (
        [system_instruction]
        + messages
    )

    response = llm_bind.invoke(
        model_messages
    )

    state.messages = (
        messages
        + [response]
    )

    return state


# ============================================================
# LANGGRAPH NODE 2
# TOOL EXECUTION NODE
# ============================================================

def tool_node(
    state: ETLAgentSchema
) -> ETLAgentSchema:
    """
    Executes tools requested by the LLM.
    """

    tool_results = []

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    latest_message = state.messages[-1]

    tool_calls = (
        latest_message.tool_calls
    )

    for tool_call in tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        selected_tool = (
            tools_by_name[tool_name]
        )

        try:

            observation = (
                selected_tool.invoke(
                    tool_args
                )
            )

        except Exception as e:

            observation = (
                f"Tool execution failed.\n"
                f"Tool: {tool_name}\n"
                f"Error type: "
                f"{type(e).__name__}\n"
                f"Error message: {e}"
            )

        tool_results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id
            )
        )

    state.messages = (
        state.messages
        + tool_results
    )

    return state


# ============================================================
# ROUTING FUNCTION
# ============================================================

def is_tool_call(
    state: ETLAgentSchema
) -> str:
    """
    Determines whether the LLM requested a tool.
    """

    latest_message = (
        state.messages[-1]
    )

    tool_calls = getattr(
        latest_message,
        "tool_calls",
        []
    )

    if tool_calls:
        return "tool_node"

    return "end"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

etl_analyst_graph = StateGraph(
    ETLAgentSchema
)


# ------------------------------------------------------------
# Add Nodes
# ------------------------------------------------------------

etl_analyst_graph.add_node(
    "llm_node",
    llm_node
)

etl_analyst_graph.add_node(
    "tool_node",
    tool_node
)


# ------------------------------------------------------------
# START
# ------------------------------------------------------------

etl_analyst_graph.add_edge(
    START,
    "llm_node"
)


# ------------------------------------------------------------
# Conditional Routing
# ------------------------------------------------------------

etl_analyst_graph.add_conditional_edges(
    "llm_node",
    is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END
    }
)


# ------------------------------------------------------------
# Return to LLM after tool result
# ------------------------------------------------------------

etl_analyst_graph.add_edge(
    "tool_node",
    "llm_node"
)


# ============================================================
# COMPILE AGENT
# ============================================================

etl_analyst = (
    etl_analyst_graph.compile()
)


# ============================================================
# PRODUCTION DATASET ETL AGENT
# ============================================================
# The original graph above is retained as the portfolio prototype. The web
# application uses this agent facade instead: it keeps the ETL responsibility
# in this original agent module but delegates execution to deterministic,
# approval-based operations rather than executing model-generated Python.

from ai_agent.models import CleanupPlan, DatasetContext, DatasetProfile, ETLReadinessReport, ReconciliationReport, StagingResult, TargetColumn, ValidationReport
from ai_agent.services.postgres_etl import propose_mapping, reconcile_row_counts
from ai_agent.services.postgres_staging import stage_dataframe
from ai_agent.services.quality import apply_cleanup_plan, profile
from ai_agent.services.recommender import CleanupRecommender, DatasetGuide


class DatasetETLAgent:
    """Production-safe extension of the original ETL analyst."""

    def __init__(self, gemini_api_key: str | None) -> None:
        self._gemini_api_key = gemini_api_key

    def profile_dataset(self, frame) -> DatasetProfile:
        return profile(frame)

    def understand_dataset(self, frame) -> DatasetContext:
        return DatasetGuide(self._gemini_api_key).explain(self.profile_dataset(frame))

    def suggest_cleanup(self, frame) -> CleanupPlan:
        return CleanupRecommender(self._gemini_api_key).recommend(self.profile_dataset(frame))

    def apply_approved_cleanup(
        self, frame, plan: CleanupPlan, approved_ids: list[str]
    ) -> tuple[object, ValidationReport]:
        return apply_cleanup_plan(frame, plan, set(approved_ids))

    def assess_load_readiness(self, frame, target_table: str, target_columns: list[TargetColumn]) -> ETLReadinessReport:
        return propose_mapping(frame, target_table, target_columns)

    def reconcile_load(self, source_rows: int, target_rows: int, target_table: str) -> ReconciliationReport:
        return reconcile_row_counts(source_rows, target_rows, target_table)

    def stage_raw_dataset(self, frame, source_name: str, source_format: str, database_config: dict[str, object]) -> StagingResult:
        return stage_dataframe(frame, source_name, source_format, database_config)


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Optional graph visualization
    # --------------------------------------------------------

    try:

        from IPython.display import Image

        img = Image(
            etl_analyst
            .get_graph()
            .draw_mermaid_png()
        )

        with open(
            "etl_analyst_graph.png",
            "wb"
        ) as f:

            f.write(img.data)

    except Exception as e:

        print(
            "Graph visualization skipped:",
            e
        )


    # ========================================================
    # TEST 1 - EXTRACT
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TEST 1 - EXTRACT DATA")
    print("=" * 80)

    extract_response = (
        etl_analyst.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="""
Extract the data from this API:

https://pokeapi.co/api/v2/pokemon

Save the extracted data in:

data/extract

Output format:

csv
"""
                    )
                ]
            }
        )
    )

    print(
        extract_response["messages"][-1].content
    )


    # ========================================================
    # TEST 2 - TRANSFORM
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TEST 2 - TRANSFORM DATA")
    print("=" * 80)

    transform_response = (
        etl_analyst.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="""
Transform the source CSV file located at:

C:/Users/samav/OneDrive/Desktop/AI_Agent/data/extract/extracted_data.csv

Transformation requirements:

- Keep only records where the Pokemon name is Bulbasaur.
- Perform the Pokemon name comparison case-insensitively.
- Preserve all original columns.
- Do not modify the source file.

Output requirements:

Output folder:

C:/Users/samav/OneDrive/Desktop/AI_Agent/data/transform

Output file name:

bulbasaur_data.csv

Output format:

csv
"""
                    )
                ]
            }
        )
    )

    print(
        transform_response["messages"][-1].content
    )
