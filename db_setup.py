import logging
import re
import mysql.connector
from mysql.connector import Error
import config

log = logging.getLogger(__name__)

_SAFE_DB_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _validate_db_name(name: str) -> str:
    """Return *name* if it is a safe identifier, otherwise raise ValueError."""
    if not _SAFE_DB_NAME_RE.match(name):
        raise ValueError(
            f"DB_NAME '{name}' contains invalid characters. "
            "Only letters, digits, and underscores are allowed."
        )
    return name


def setup_database():
    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )

        if connection.is_connected():
            cursor = connection.cursor()
            safe_name = _validate_db_name(config.DB_NAME)
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{safe_name}`")  # noqa: S608
            log.info("Database '%s' checked/created.", safe_name)

            cursor.execute(f"USE `{safe_name}`")  # noqa: S608

            create_table_query = """
            CREATE TABLE IF NOT EXISTS violations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                track_id INT,
                vehicle_class VARCHAR(50),
                violation_type VARCHAR(100),
                plate_number VARCHAR(50),
                confidence FLOAT,
                frame_number INT,
                timestamp DATETIME,
                is_exempted BOOLEAN
            )
            """
            cursor.execute(create_table_query)
            log.info("Table 'violations' created or already exists.")
            connection.commit()

    except Error as e:
        log.error("MySQL Error during setup: %s", e)
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None and connection.is_connected():
            connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    setup_database()
