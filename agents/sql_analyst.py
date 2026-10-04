import os
import sys

# Add parent directory to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from models.schema import AgentState, judgeSchema

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from utils.llm_pick import pick_llm
from utils.database import DatabaseUtil

from langgraph.graph import StateGraph, START, END


# ============================================================
# SQL ANALYST AGENT
# ============================================================


# ------------------------------------------------------------
# 1. CURATE USER QUESTION
# ------------------------------------------------------------

def curate_question(state: AgentState) -> AgentState:
    """
    Cleans / rewrites the user's question so that it is easier
    for the SQL analyst agent to understand.
    """

    user_question = state.user_question

    # Pick a low-cost / lightweight LLM
    llm = pick_llm("low") | StrOutputParser()

    prompt = f"""
    Curate the following question for an AI SQL analyst agent.

    Keep the meaning exactly the same.
    Make the question clear and concise.

    User question:
    {user_question}
    """

    # StrOutputParser returns a normal Python string
    response = llm.invoke(prompt)

    state.curated_ques = response

    state.messages = state.messages + [
        HumanMessage(content=response)
    ]

    return state


# ------------------------------------------------------------
# 2. BUILD SQL PROMPT CONTEXT
# ------------------------------------------------------------

def prompt_context(state: AgentState) -> AgentState:
    """
    Builds a schema-grounded and constraint-aware PostgreSQL prompt.

    This function only prepares the prompt.
    SQL generation happens in generate_sql().
    """

    curated_question = state.curated_ques

    connection_details = {
        "host": os.getenv("host"),
        "port": os.getenv("port"),
        "user": os.getenv("user"),
        "password": os.getenv("password"),
        "dbname": os.getenv("dbname")
    }

    db_obj = DatabaseUtil(connection_details)

    # Fetch tables, columns, sample data, and ideally constraints
    schema_info_context = db_obj.schema_details("public")

    # Temporary debugging
    print("\n" + "=" * 80)
    print("SCHEMA AND CONSTRAINTS SENT TO LLM")
    print("=" * 80)
    print(schema_info_context)
    print("=" * 80 + "\n")

    prompt = f"""
You are an expert PostgreSQL SQL analyst.

Your task is to convert the user's natural-language question into a
VALID, READ-ONLY PostgreSQL SQL query.

You MUST generate the query using ONLY the database metadata provided below.

The DATABASE METADATA may contain:

- Schema names
- Table names
- Column names
- Data types
- Primary keys
- Foreign keys
- Unique constraints
- NOT NULL constraints
- CHECK constraints
- Sample data

The DATABASE METADATA is the ONLY source of truth.


================================================================
DATABASE USAGE RULES
================================================================

1. Use ONLY tables that exist in the supplied DATABASE METADATA.

2. Use ONLY columns that exist in the supplied DATABASE METADATA.

3. NEVER invent table names.

4. NEVER invent column names.

5. NEVER invent relationships.

6. Always use fully qualified PostgreSQL table names.

Correct:

    public.payments

    public.users

    public.rides

Incorrect:

    payments

    users

    rides


7. When using aliases, the original table must still be schema-qualified.

Correct:

    FROM public.payments AS p

    JOIN public.users AS u
        ON p.user_id = u.user_id


================================================================
CONSTRAINT-AWARE REASONING
================================================================

8. Carefully inspect PRIMARY KEY constraints before generating SQL.

A primary key uniquely identifies a row.

For example, if metadata contains:

    PRIMARY KEY (user_id)

then user_id should be treated as the unique identifier for that table.


9. Carefully inspect FOREIGN KEY constraints.

Foreign keys are the preferred source for determining table relationships.

If metadata contains:

    public.payments.user_id
        REFERENCES public.users.user_id

then use:

    JOIN public.users AS u
        ON p.user_id = u.user_id


10. ALWAYS prefer declared FOREIGN KEY relationships over guessing joins
based only on similarly named columns.


11. NEVER assume that two columns should be joined merely because they
have the same name.

For example:

    table_a.user_id
    table_b.user_id

does NOT automatically mean they are related unless the metadata
supports that relationship.


12. If an explicit FOREIGN KEY relationship exists, use that relationship
when the user's question requires those tables.


13. If multiple foreign-key paths exist between tables, select the path
that semantically matches the user's question.


14. Pay attention to UNIQUE constraints.

If a column or combination of columns is declared UNIQUE, treat those
values as uniquely identifying records according to the database schema.


15. Pay attention to NOT NULL constraints.

Do not add unnecessary:

    IS NOT NULL

conditions when the schema guarantees that the column cannot be NULL,
unless required by the user's question.


16. Pay attention to CHECK constraints.

Use CHECK constraints to understand valid domain values and business rules.

For example, if metadata contains:

    CHECK (rating >= 1 AND rating <= 5)

then understand that valid ratings range from 1 through 5.


17. Do NOT violate any database constraint in your reasoning.


================================================================
JOIN RULES
================================================================

18. Before creating a JOIN, determine:

    - Source table
    - Target table
    - Foreign key column
    - Referenced primary/unique key
    - Relationship direction


19. Generate joins primarily from declared database constraints.

Example metadata:

    Table: payments
    Primary Key:
        payment_id

    Foreign Keys:
        ride_id REFERENCES public.rides(ride_id)
        user_id REFERENCES public.users(user_id)

Then:

    public.payments AS p

can join:

    public.rides AS r
        ON p.ride_id = r.ride_id

and:

    public.users AS u
        ON p.user_id = u.user_id


20. Do not create unnecessary joins.

If all requested information exists in one table, query that table directly.


Example:

Question:

    What payment methods are available?

If payment_method exists in public.payments, use:

    SELECT DISTINCT p.payment_method
    FROM public.payments AS p
    LIMIT 10;

Do NOT join other tables unnecessarily.


21. If information requires several tables, use the minimum number of
tables necessary to answer the question.


================================================================
QUERY GENERATION RULES
================================================================

22. Use PostgreSQL syntax only.

23. Generate READ-ONLY SQL only.

Allowed operations include:

    SELECT
    WITH
    JOIN
    WHERE
    GROUP BY
    HAVING
    ORDER BY
    LIMIT
    DISTINCT


24. NEVER generate database-changing operations including:

    INSERT
    UPDATE
    DELETE
    DROP
    ALTER
    TRUNCATE
    CREATE
    MERGE
    GRANT
    REVOKE


25. For questions asking for unique or different values,
use DISTINCT when appropriate.


26. For aggregation questions, use appropriate PostgreSQL functions:

    COUNT
    SUM
    AVG
    MIN
    MAX


27. When aggregation is used, correctly determine whether GROUP BY
is required.


28. When filtering aggregated values, use HAVING rather than WHERE
when appropriate.


29. When ranking results, use ORDER BY appropriately.


30. Unless the user explicitly asks for a different number of results,
limit result sets to 10 rows when appropriate.


31. Do not add LIMIT to queries where it would change the intended
meaning unnecessarily, such as a single aggregate value:

    SELECT COUNT(*)
    FROM public.users;


================================================================
SCHEMA INTERPRETATION
================================================================

32. Analyze the database metadata BEFORE generating the SQL.

Internally determine:

    A. What is the user asking for?

    B. Which table contains the requested information?

    C. Which columns are required?

    D. What are the relevant primary keys?

    E. Are foreign keys required?

    F. Are joins required?

    G. Which declared constraints define those relationships?

    H. Are filters required?

    I. Is aggregation required?

    J. Is GROUP BY required?

    K. Is DISTINCT required?

    L. Is ordering required?

    M. Is LIMIT appropriate?


33. Do this reasoning internally.

Do NOT expose this reasoning in the output.


================================================================
IMPORTANT RELATIONSHIP EXAMPLE
================================================================

Suppose the metadata contains:

    Table: users
        user_id
        first_name
        last_name

    PRIMARY KEY:
        user_id


    Table: payments
        payment_id
        user_id
        amount

    PRIMARY KEY:
        payment_id

    FOREIGN KEY:
        payments.user_id
        REFERENCES users.user_id


And the user asks:

    Which users made the highest total payments?


Correct:

    SELECT
        u.user_id,
        u.first_name,
        u.last_name,
        SUM(p.amount) AS total_payment
    FROM public.users AS u
    JOIN public.payments AS p
        ON p.user_id = u.user_id
    GROUP BY
        u.user_id,
        u.first_name,
        u.last_name
    ORDER BY total_payment DESC
    LIMIT 10;


The JOIN exists because it is supported by the declared FOREIGN KEY.


================================================================
NO RELATIONSHIP HALLUCINATION
================================================================

34. Never create a relationship that is not supported by the supplied
database metadata.

35. If there is no declared relationship between two required tables,
do not arbitrarily join them.

36. Prefer explicit database constraints over assumptions derived from
sample data.

37. Sample data may help understand values but MUST NOT override schema
definitions or constraints.

38. Primary keys and foreign keys have higher authority than sample data
for determining table relationships.


================================================================
OUTPUT FORMAT
================================================================

39. Return ONLY the SQL query.

40. Do NOT explain the query.

41. Do NOT return Markdown.

42. Do NOT include:

    ```sql

or:

    ```


43. Do not return analysis or reasoning.

44. The output must be executable directly by PostgreSQL.


================================================================
USER QUESTION
================================================================

{curated_question}


================================================================
DATABASE METADATA
================================================================

{schema_info_context}


Generate the PostgreSQL SQL query now.
"""

    state.prompt_context = prompt

    return state

# ------------------------------------------------------------
# 3. GENERATE SQL
# ------------------------------------------------------------

def generate_sql(state: AgentState) -> AgentState:
    """
    Generates the PostgreSQL query from the prompt context.
    """

    prompt = state.prompt_context

    # StrOutputParser converts model output into plain string
    llm = pick_llm("medium") | StrOutputParser()

    generated_sql_query = llm.invoke(prompt)

    # Clean possible Markdown formatting just in case
    generated_sql_query = generated_sql_query.strip()

    if generated_sql_query.startswith("```sql"):
        generated_sql_query = generated_sql_query[6:]

    elif generated_sql_query.startswith("```"):
        generated_sql_query = generated_sql_query[3:]

    if generated_sql_query.endswith("```"):
        generated_sql_query = generated_sql_query[:-3]

    generated_sql_query = generated_sql_query.strip()

    state.generated_sql_query = generated_sql_query

    return state


# ------------------------------------------------------------
# 4. SQL SAFETY CHECK
# ------------------------------------------------------------

def is_safe(state: AgentState) -> AgentState:
    """
    Uses an LLM judge to determine whether the generated SQL
    is safe to execute.

    Only read-only queries should be allowed.
    """

    sql_query = state.generated_sql_query

    llm = pick_llm("medium")

    # Structured output using your judgeSchema
    llm_judge = llm.with_structured_output(judgeSchema)

    prompt = f"""
    You are a SQL security judge.

    Determine whether the following SQL query is safe to execute.

    A safe query must ONLY retrieve data.

    Safe examples:
    - SELECT
    - SELECT with JOIN
    - SELECT with GROUP BY
    - SELECT with ORDER BY
    - WITH / CTE used only for reading data

    Unsafe commands include:

    INSERT
    UPDATE
    DELETE
    DROP
    ALTER
    TRUNCATE
    CREATE
    GRANT
    REVOKE
    MERGE
    CALL

    The query must not modify the database in any way.

    SQL Query:

    {sql_query}
    """

    response = llm_judge.invoke(prompt)

    # Convert Pydantic result to dictionary
    response_dict = response.model_dump()

    state.is_safe = response_dict["answer"]
    state.comments = response_dict["comments"]

    return state


# ------------------------------------------------------------
# 5. CONDITIONAL ROUTING FUNCTION
# ------------------------------------------------------------

def is_safe_edge_condition(state: AgentState) -> str:
    """
    Routes the graph depending on whether the SQL is safe.
    """

    if state.is_safe.lower().strip() == "yes":
        return "execute_sql"

    return "cancelled_sql"


# ------------------------------------------------------------
# 6. CANCEL UNSAFE SQL
# ------------------------------------------------------------

def cancelled_sql(state: AgentState) -> AgentState:
    """
    Stops execution when SQL is unsafe.
    """

    comments = state.comments

    state.final_answer = (
        "The generated SQL query is deemed unsafe to execute. "
        f"Reason: {comments}. "
        "Please modify your query and try again."
    )

    state.messages = state.messages + [
        AIMessage(content=state.final_answer)
    ]

    return state


# ------------------------------------------------------------
# 7. EXECUTE SQL
# ------------------------------------------------------------

def execute_sql(state: AgentState) -> AgentState:
    """
    Executes the approved SQL query.
    """

    sql_query = state.generated_sql_query

    connection_details = {
        "host": os.getenv("host"),
        "port": os.getenv("port"),
        "user": os.getenv("user"),
        "password": os.getenv("password"),
        "dbname": os.getenv("dbname")
    }

    db_obj = DatabaseUtil(connection_details)

    sql_query_result = db_obj.execute_sql(sql_query)

    state.sql_query_result = sql_query_result

    return state


# ------------------------------------------------------------
# 8. GENERATE USER-FRIENDLY RESPONSE
# ------------------------------------------------------------

def representation(state: AgentState) -> AgentState:
    """
    Converts the SQL execution result into a simple
    natural-language response.
    """

    execution_result = state.sql_query_result
    curated_question = state.curated_ques

    llm = pick_llm("low") | StrOutputParser()

    prompt = f"""
    You are an SQL analyst assistant.

    Generate a concise and informative answer for the user
    based on the SQL query result.

    Do not include SQL commands.

    Do not explain database implementation details.

    Do not simply repeat the raw SQL result.

    Summarize the information clearly in plain language.

    If no rows were returned, clearly tell the user that
    no matching data was found.

    User Question:

    {curated_question}


    SQL Execution Result:

    {execution_result}
    """

    llm_response = llm.invoke(prompt)

    state.final_answer = llm_response

    state.messages = state.messages + [
        AIMessage(content=llm_response)
    ]

    return state


# ============================================================
# BUILD LANGGRAPH
# ============================================================

sql_agent_graph = StateGraph(AgentState)


# ------------------------------------------------------------
# ADD NODES
# ------------------------------------------------------------

sql_agent_graph.add_node(
    "curate_question",
    curate_question
)

sql_agent_graph.add_node(
    "prompt_context",
    prompt_context
)

sql_agent_graph.add_node(
    "generate_sql",
    generate_sql
)

sql_agent_graph.add_node(
    "is_safe",
    is_safe
)

sql_agent_graph.add_node(
    "cancelled_sql",
    cancelled_sql
)

sql_agent_graph.add_node(
    "execute_sql",
    execute_sql
)

sql_agent_graph.add_node(
    "representation",
    representation
)


# ------------------------------------------------------------
# ADD EDGES
# ------------------------------------------------------------

sql_agent_graph.add_edge(
    START,
    "curate_question"
)

sql_agent_graph.add_edge(
    "curate_question",
    "prompt_context"
)

sql_agent_graph.add_edge(
    "prompt_context",
    "generate_sql"
)

sql_agent_graph.add_edge(
    "generate_sql",
    "is_safe"
)


# ------------------------------------------------------------
# CONDITIONAL EDGE
# ------------------------------------------------------------

sql_agent_graph.add_conditional_edges(
    "is_safe",
    is_safe_edge_condition,
    {
        "execute_sql": "execute_sql",
        "cancelled_sql": "cancelled_sql"
    }
)


# ------------------------------------------------------------
# FINAL EDGES
# ------------------------------------------------------------

sql_agent_graph.add_edge(
    "cancelled_sql",
    END
)

sql_agent_graph.add_edge(
    "execute_sql",
    "representation"
)

sql_agent_graph.add_edge(
    "representation",
    END
)

sql_analyst_agent = sql_agent_graph.compile()


# ============================================================
# PRODUCTION LOCAL SQL AGENT
# ============================================================
# This keeps local-dataset analysis in the original SQL agent module while the
# actual query validator and executor remain deterministic and read-only.

from ai_agent.models import DatasetProfile
from ai_agent.services.local_analysis import (
    LocalSQLAssistant,
    apply_requested_row_limit,
    execute_local_query,
)


class LocalDatasetSQLAgent:
    """Production-safe SQL extension of the original SQL analyst."""

    def __init__(self, gemini_api_key: str | None) -> None:
        self._gemini_api_key = gemini_api_key

    def analyze(
        self, frame, dataset_profile: DatasetProfile, question: str | None, sql: str | None
    ) -> tuple[str, list[str], list[list[object]]]:
        query = sql or LocalSQLAssistant(self._gemini_api_key).generate(question or "", dataset_profile)
        if not sql:
            query = apply_requested_row_limit(question or "", query)
        return execute_local_query(frame, query)
# ============================================================
# RUN AGENT
# ============================================================

if __name__ == "__main__":

    # Compile graph
    

    # --------------------------------------------------------
    # Optional: Save graph image
    # --------------------------------------------------------

    try:

        from IPython.display import Image

        img = Image(
            sql_analyst_agent
            .get_graph()
            .draw_mermaid_png()
        )

        with open(
            "sql_analyst_agent_graph.png",
            "wb"
        ) as f:

            f.write(img.data)

    except Exception as e:
        print(
            f"Graph visualization skipped: {e}"
        )


    # --------------------------------------------------------
    # Initial Agent State
    # --------------------------------------------------------

    input_schema = {

        "messages": [],

        "user_question":
            "What are the different types of payment methods available in the database?",

        "curated_ques": "",

        "prompt_context": "",

        "generated_sql_query": "",

        "is_safe": "No",

        "comments": "",

        "sql_query_result": "",

        "final_answer": ""

    }


    # --------------------------------------------------------
    # Execute Agent
    # --------------------------------------------------------

    try:

        sql_analyst_response = (
            sql_analyst_agent.invoke(
                input_schema
            )
        )

        print("\n")
        print("=" * 60)
        print("SQL ANALYST AGENT RESULT")
        print("=" * 60)

        print(
            "\nCurated Question:\n",
            sql_analyst_response.get(
                "curated_ques"
            )
        )

        print(
            "\nGenerated SQL:\n",
            sql_analyst_response.get(
                "generated_sql_query"
            )
        )

        print(
            "\nSQL Safe:\n",
            sql_analyst_response.get(
                "is_safe"
            )
        )

        print(
            "\nFinal Answer:\n",
            sql_analyst_response.get(
                "final_answer"
            )
        )

    except Exception as e:

        print("\nSQL Analyst Agent failed.")

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {e}"
        )
