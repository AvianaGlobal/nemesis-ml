import pathlib

import sqlalchemy


def create_engine_str(dialect, username, password, host, port, database):
    return f'{dialect}://{username}:{password}@{host}:{port}/{database}'


path = pathlib.Path(__file__).parent.resolve()
dataset = 'tests/test_data.csv'
dialect = 'db2'
username = 'kfn42270'
password = '6kg39fqcqk+tqqpf'
host = 'dashdb-txn-sbox-yp-dal09-03.services.dal.bluemix.net'
port = 50000
database = 'BLUDB'
columns = '*'
schema = 'kfn42270'
table = 'OON_SCORES'

file_path = path.joinpath(f'{dataset}')
engine_str = create_engine_str(dialect, username, password, host, port, database)
engine = sqlalchemy.create_engine(engine_str)
session = engine.connect()
query = f'select {columns} from {schema}.{table} limit 10'
