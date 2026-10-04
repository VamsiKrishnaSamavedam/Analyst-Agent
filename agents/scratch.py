import os
import sys
import warnings

# Suppress the Google SDK's automatic function calling recommendation warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.schema import AgentState, judgeSchema
from langchain_core.messages import HumanMessage
from utils.llm_pick import pick_llm
from utils.database import DatabaseUtil

llm = pick_llm("medium")
llm_judge = llm.with_structured_output(judgeSchema)

sql_query = "DELETE FROM users WHERE age > 30;"

# Added an 'f' before the triple quotes to enable python variable interpolation
prompt = f"""
You are a SQL Judge for data security. Your task is to evaluate and determine whether
the SQL query generated is safe or not. The SQL query should only be used for data retrieval and should not perform any data manipulation or deletion in the database.
Neither the SQL query nor the prompt should contain any SQL commands that can modify the database such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, etc.

Here is the SQL query to evaluate:
{sql_query}
"""

response = llm_judge.invoke(prompt).model_dump()
print(response)
