import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()


def pick_llm(level: str):
    """
    Picks the appropriate Gemini model.

    low:
        Cheap / lightweight tasks

    medium:
        SQL generation and normal reasoning

    high:
        More complex reasoning tasks
    """

    level = level.lower()

    if level in {"low", "medium"}:

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            max_retries=5,
            temperature=0.2,
            thinking_level=level,
        )

    elif level == "high":

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            max_retries=5,
            temperature=0.2,
            thinking_level=level,
        )

    else:

        raise ValueError(
            "Invalid level provided. "
            "Choose from 'low', 'medium', or 'high'."
        )

    return llm


if __name__ == "__main__":

    llm_object = pick_llm("low")

    response = llm_object.invoke(
        "What is the capital of France?"
    )

    print(response)
