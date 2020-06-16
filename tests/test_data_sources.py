#!/usr/bin/env python
import pandas

import app.feature_engineering.data_sources as data_sources
from app.defaults import *


def test_file_path(path):
    print(f'Testing {path}')
    assert path


def test_file_datasource(file_path):
    print(f'Testing {file_path} as DataFrame')
    datasource = data_sources.FileDataSource(file_path)
    test = datasource.load_data()
    assert type(test) == pandas.DataFrame


def test_sql_connection(engine_str):
    print(f'Testing database connection using {engine_str}')
    engine = sqlalchemy.create_engine(engine_str, echo=True)
    session = engine.connect()
    assert type(session) == sqlalchemy.engine.base.Connection


def test_sql_datasource(query, session):
    print(f'Testing {query} on {session}')
    datasource = data_sources.SqlDataSource(query, session)
    test = datasource.load_data()
    assert type(test) == pandas.DataFrame

file_path = path.joinpath(f'{dataset}')
engine_str = create_engine_str(dialect, username, password, host, port, database)
query = f'select {columns} from {schema}.{table} limit 10'
engine = sqlalchemy.create_engine(engine_str)
session = engine.connect()

test_file_path(file_path)
test_sql_connection(engine_str)
test_sql_datasource(query, session)
