import os
import requests
import pandas as pd


class ETLTools:
    """
    Utility class containing ETL operations used by the ETL Analyst Agent.
    """

    def __init__(self):
        pass

    # ============================================================
    # EXTRACT + LOAD
    # ============================================================

    def extract_load(
        self,
        url: str,
        output_folder: str,
        output_format: str
    ) -> str:
        """
        Extracts JSON data from an API and saves it locally.

        Args:
            url:
                API endpoint.

            output_folder:
                Destination folder.

            output_format:
                csv, json, or parquet.

        Returns:
            Result message.
        """

        try:
            # ----------------------------------------------------
            # Resolve output folder
            # ----------------------------------------------------

            if os.path.isabs(output_folder):
                final_output_folder = output_folder
            else:
                project_root = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        ".."
                    )
                )

                final_output_folder = os.path.join(
                    project_root,
                    output_folder
                )

            os.makedirs(
                final_output_folder,
                exist_ok=True
            )

            # ----------------------------------------------------
            # Call API
            # ----------------------------------------------------

            response = requests.get(
                url,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            # ----------------------------------------------------
            # Convert API response to DataFrame
            # ----------------------------------------------------

            if isinstance(data, dict) and "results" in data:
                df = pd.json_normalize(
                    data["results"]
                )

            elif isinstance(data, list):
                df = pd.json_normalize(data)

            elif isinstance(data, dict):
                df = pd.json_normalize([data])

            else:
                return (
                    "Unsupported API response structure."
                )

            # ----------------------------------------------------
            # Normalize format
            # ----------------------------------------------------

            output_format = (
                output_format
                .lower()
                .strip()
                .replace(".", "")
            )

            filename = os.path.join(
                final_output_folder,
                f"extracted_data.{output_format}"
            )

            # ----------------------------------------------------
            # Save file
            # ----------------------------------------------------

            if output_format == "csv":

                df.to_csv(
                    filename,
                    index=False
                )

            elif output_format == "json":

                df.to_json(
                    filename,
                    orient="records",
                    lines=True
                )

            elif output_format == "parquet":

                df.to_parquet(
                    filename,
                    index=False
                )

            else:

                return (
                    "Unsupported output format. "
                    "Choose csv, json, or parquet."
                )

            return (
                f"Data extracted successfully.\n"
                f"Source: {url}\n"
                f"Rows extracted: {len(df)}\n"
                f"Output file: {filename}"
            )

        except requests.exceptions.RequestException as e:

            return (
                f"API extraction failed.\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {e}"
            )

        except Exception as e:

            return (
                f"Extraction/load failed.\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {e}"
            )

    # ============================================================
    # TRANSFORM CONTEXT
    # ============================================================

    def transform_load_context(
        self,
        file_path: str
    ) -> str:
        """
        Reads the source file and creates useful context for the LLM.

        Includes:
        - file path
        - columns
        - data types
        - row count
        - first three records
        """

        try:

            df = self.read_dataframe(file_path)

            context = f"""
SOURCE FILE:
{file_path}

TOTAL ROWS:
{len(df)}

COLUMNS:
{list(df.columns)}

DATA TYPES:
{df.dtypes.astype(str).to_dict()}

SAMPLE RECORDS:
{df.head(3).to_dict(orient="records")}
"""

            return context.strip()

        except Exception as e:

            return (
                f"Could not read transformation context.\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {e}"
            )

    # ============================================================
    # COMMON FILE READER
    # ============================================================

    def read_dataframe(
        self,
        file_path: str
    ) -> pd.DataFrame:
        """
        Reads CSV, JSON, or Parquet into a Pandas DataFrame.
        """

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Source file does not exist: {file_path}"
            )

        file_extension = (
            os.path.splitext(file_path)[1]
            .lower()
        )

        if file_extension == ".csv":

            return pd.read_csv(file_path)

        elif file_extension == ".json":

            try:
                return pd.read_json(
                    file_path,
                    lines=True
                )

            except ValueError:
                return pd.read_json(file_path)

        elif file_extension == ".parquet":

            return pd.read_parquet(file_path)

        else:

            raise ValueError(
                f"Unsupported input file format: "
                f"{file_extension}"
            )

    # ============================================================
    # GENERATED CODE EXECUTION
    # ============================================================

    def execute_code(
        self,
        code: str
    ) -> str:
        """Refuse model-generated code; use the approval-based ETL workflow."""

        del code
        return (
            "Generated Python is never executed. Review a cleanup plan in the "
            "dataset workflow and apply only approved deterministic actions."
        )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    obj = ETLTools()

    test_path = (
        "C:/Users/samav/OneDrive/Desktop/"
        "AI_Agent/data/extract/extracted_data.csv"
    )

    print(
        obj.transform_load_context(
            test_path
        )
    )
