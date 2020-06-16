import pathlib
import sqlalchemy


def create_engine_str(dialect, username, password, host, port, database):
    return f'{dialect}://{username}:{password}@{host}:{port}/{database}'


PATH = pathlib.Path(__file__).parent.resolve()
DATASET = 'tests/test_data.csv'
DIALECT = 'db2'
USERNAME = 'kfn42270'
PASSWORD = '6kg39fqcqk+tqqpf'
HOST = 'dashdb-txn-sbox-yp-dal09-03.services.dal.bluemix.net'
PORT = 50000
DATABASE = 'BLUDB'
COLUMNS = '*'
SCHEMA = 'kfn42270'
TABLE = 'OON_SCORES'
FILTER = ''

path = PATH

file_path = PATH.joinpath(f'{DATASET}')
# test_file_datasource(file_path)

engine_str = create_engine_str(DIALECT, USERNAME, PASSWORD, HOST, PORT, DATABASE)
# test_sql_connection(engine_str)

query = f'SELECT {COLUMNS} FROM {SCHEMA}.{TABLE}'
engine = sqlalchemy.create_engine(engine_str)
session = engine.connect()
# sql_datasource(query, session)
