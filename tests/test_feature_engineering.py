#!/usr/bin/env python
import os
import pathlib
import unittest

import pandas
import sqlalchemy

import app.feature_engineering.build_features as build_features
import app.feature_engineering.data_sources as data_sources

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
    engine = sqlalchemy.create_engine(engine_str)
    session = engine.connect()
    assert type(session) == sqlalchemy.engine.base.Connection

def test_sql_datasource(query, session):
    print(f'Testing {query} on {session}')
    datasource = data_sources.SqlDataSource(query, session)
    test = datasource.load_data()
    assert type(test) == pandas.DataFrame


def create_engine_str(dialect, username, password, host, port, database):
    return f'{dialect}://{username}:{password}@{host}:{port}/{database}'

# PATH = pathlib.Path(__file__).parent.resolve()
# DATASET = 'tests/test_data.csv'
# DIALECT = 'db2'
# USERNAME = 'kfn42270'
# PASSWORD = '6kg39fqcqk+tqqpf'
# HOST = 'dashdb-txn-sbox-yp-dal09-03.services.dal.bluemix.net'
# PORT = 50000
# DATABASE = 'BLUDB'
# COLUMNS = '*'
# SCHEMA = 'kfn42270'
# TABLE = 'OON_SCORES'
# FILTER = ''
#
# test_file_path = PATH.joinpath(f'{DATASET}')
# test_file_datasource(test_file_path)
#
# test_engine_str = create_engine_str(DIALECT, USERNAME, PASSWORD, HOST, PORT, DATABASE)
# test_sql_connection(test_engine_str)
#
# test_query = f'SELECT {COLUMNS} FROM {SCHEMA}.{TABLE}'
# test_engine = sqlalchemy.create_engine(test_engine_str)
# test_session = test_engine.connect()
# test_sql_datasource(test_query, test_session)
