import os
import psycopg2
from psycopg2 import sql


class DatabaseUtil:

    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None

        try:
            self.connection = psycopg2.connect(**db_config)

        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.connection = None


    def schema_details(self, schema_name):
        """
        Returns detailed PostgreSQL schema metadata including:

        - Tables
        - Columns in correct physical order
        - Data types
        - Nullability
        - Primary keys
        - Foreign keys
        - Unique constraints
        - Check constraints
        - Labeled sample rows
        """

        if self.connection is None:
            return "Database connection is not available."

        schema_info_context = f"Database Schema: {schema_name}\n"

        cursor = None

        try:
            cursor = self.connection.cursor()

            # =========================================================
            # 1. GET TABLES
            # =========================================================

            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """,
                (schema_name,)
            )

            tables_list = cursor.fetchall()

            # =========================================================
            # LOOP THROUGH TABLES
            # =========================================================

            for table in tables_list:

                table_name = table[0]

                schema_info_context += (
                    f"\n{'=' * 60}\n"
                    f"Table: {schema_name}.{table_name}\n"
                    f"{'=' * 60}\n"
                )

                # =====================================================
                # 2. GET COLUMNS IN CORRECT ORDER
                # =====================================================

                cursor.execute(
                    """
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        ordinal_position
                    FROM information_schema.columns
                    WHERE table_schema = %s
                      AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (schema_name, table_name)
                )

                columns_list = cursor.fetchall()

                schema_info_context += "\nColumns:\n"

                column_names = []

                for column in columns_list:

                    column_name = column[0]
                    data_type = column[1]
                    is_nullable = column[2]

                    column_names.append(column_name)

                    nullable_text = (
                        "NULL"
                        if is_nullable == "YES"
                        else "NOT NULL"
                    )

                    schema_info_context += (
                        f"  - {column_name}: "
                        f"{data_type}, "
                        f"{nullable_text}\n"
                    )

                # =====================================================
                # 3. PRIMARY KEY CONSTRAINTS
                # =====================================================

                cursor.execute(
                    """
                    SELECT
                        kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                       AND tc.table_schema = kcu.table_schema
                       AND tc.table_name = kcu.table_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND tc.table_schema = %s
                      AND tc.table_name = %s
                    ORDER BY kcu.ordinal_position;
                    """,
                    (schema_name, table_name)
                )

                primary_keys = cursor.fetchall()

                schema_info_context += "\nPrimary Key:\n"

                if primary_keys:

                    for primary_key in primary_keys:
                        schema_info_context += (
                            f"  - {primary_key[0]}\n"
                        )

                else:
                    schema_info_context += "  - None\n"

                # =====================================================
                # 4. FOREIGN KEY CONSTRAINTS
                # =====================================================

                cursor.execute(
                    """
                    SELECT
                        kcu.column_name,
                        ccu.table_schema AS foreign_table_schema,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name,
                        tc.constraint_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                       AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage ccu
                        ON ccu.constraint_name = tc.constraint_name
                       AND ccu.constraint_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = %s
                      AND tc.table_name = %s
                    ORDER BY kcu.ordinal_position;
                    """,
                    (schema_name, table_name)
                )

                foreign_keys = cursor.fetchall()

                schema_info_context += "\nForeign Keys:\n"

                if foreign_keys:

                    for fk in foreign_keys:

                        local_column = fk[0]
                        foreign_schema = fk[1]
                        foreign_table = fk[2]
                        foreign_column = fk[3]
                        constraint_name = fk[4]

                        schema_info_context += (
                            f"  - {local_column} "
                            f"-> {foreign_schema}.{foreign_table}"
                            f"({foreign_column}) "
                            f"[Constraint: {constraint_name}]\n"
                        )

                else:
                    schema_info_context += "  - None\n"

                # =====================================================
                # 5. UNIQUE CONSTRAINTS
                # =====================================================

                cursor.execute(
                    """
                    SELECT
                        tc.constraint_name,
                        kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                       AND tc.table_schema = kcu.table_schema
                       AND tc.table_name = kcu.table_name
                    WHERE tc.constraint_type = 'UNIQUE'
                      AND tc.table_schema = %s
                      AND tc.table_name = %s
                    ORDER BY
                        tc.constraint_name,
                        kcu.ordinal_position;
                    """,
                    (schema_name, table_name)
                )

                unique_constraints = cursor.fetchall()

                schema_info_context += "\nUnique Constraints:\n"

                if unique_constraints:

                    for constraint in unique_constraints:

                        constraint_name = constraint[0]
                        column_name = constraint[1]

                        schema_info_context += (
                            f"  - {column_name} "
                            f"[Constraint: {constraint_name}]\n"
                        )

                else:
                    schema_info_context += "  - None\n"

                # =====================================================
                # 6. CHECK CONSTRAINTS
                # =====================================================

                cursor.execute(
                    """
                    SELECT
                        tc.constraint_name,
                        cc.check_clause
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.check_constraints cc
                        ON tc.constraint_name = cc.constraint_name
                       AND tc.constraint_schema = cc.constraint_schema
                    WHERE tc.constraint_type = 'CHECK'
                      AND tc.table_schema = %s
                      AND tc.table_name = %s
                    ORDER BY tc.constraint_name;
                    """,
                    (schema_name, table_name)
                )

                check_constraints = cursor.fetchall()

                schema_info_context += "\nCheck Constraints:\n"

                if check_constraints:

                    for check_constraint in check_constraints:

                        constraint_name = check_constraint[0]
                        check_clause = check_constraint[1]

                        schema_info_context += (
                            f"  - {check_clause} "
                            f"[Constraint: {constraint_name}]\n"
                        )

                else:
                    schema_info_context += "  - None\n"

                # =====================================================
                # 7. SAMPLE DATA
                # =====================================================

                query = sql.SQL(
                    "SELECT * FROM {}.{} LIMIT 5;"
                ).format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name)
                )

                cursor.execute(query)

                sample_data = cursor.fetchall()

                # Use cursor.description so values are mapped to the
                # exact column order returned by SELECT *
                sample_column_names = [
                    description[0]
                    for description in cursor.description
                ]

                schema_info_context += "\nSample Data:\n"

                if sample_data:

                    for row in sample_data:

                        schema_info_context += "  Row:\n"

                        for column_name, value in zip(
                            sample_column_names,
                            row
                        ):

                            schema_info_context += (
                                f"    {column_name}: {value}\n"
                            )

                else:
                    schema_info_context += "  - No sample data\n"

        except Exception as e:

            print(f"Error fetching schema details: {e}")

            schema_info_context = (
                f"Error fetching schema details: {e}"
            )

        finally:

            if cursor:
                cursor.close()

            if self.connection:
                self.connection.close()

        return schema_info_context


    def execute_sql(self, query):
        """
        Executes SQL and returns the result.
        """

        if self.connection is None:
            return None

        cursor = None

        try:

            cursor = self.connection.cursor()

            cursor.execute(query)

            result = cursor.fetchall()

            # SELECT queries don't actually need commit,
            # but keeping this will not hurt.
            self.connection.commit()

            return str(result)

        except Exception as e:

            print(f"Error executing query: {e}")

            if self.connection:
                self.connection.rollback()

            return None

        finally:

            if cursor:
                cursor.close()

            if self.connection:
                self.connection.close()


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    obj = DatabaseUtil(
        {
            "host": os.getenv("host"),
            "port": os.getenv("port"),
            "user": os.getenv("user"),
            "password": os.getenv("password"),
            "dbname": os.getenv("dbname")
        }
    )

    result = obj.schema_details("public")

    print(result)

    with open(
        "test_schema_details.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(result)