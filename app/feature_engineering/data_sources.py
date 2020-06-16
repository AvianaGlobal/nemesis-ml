# -*- encoding: utf-8 -*-
import pandas


class DataSource(object):

    def __init__(self, Parent=None):
        super(DataSource, self).__init__(parent)

    def load_data(self):
        raise NotImplementedError


class FileDataSource(DataSource):

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pandas.read_csv(self.file_path)
        return df


class SqlDataSource(DataSource):

    def __init__(self, query, session):
        self.session = session
        self.query = query

    def load_data(self):
        df = pandas.read_sql(self.query, self.session)
        return df
