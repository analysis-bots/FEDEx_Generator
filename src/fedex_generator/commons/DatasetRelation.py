from pandas import Series


class DatasetRelation(object):
    """
    A class that represents the relation between the source and the result of an operation.
    """

    def __init__(self, source_df, result_df, source_table_name):
        """
        :param source_df: The source DataFrame
        :param result_df: The result DataFrame
        :param source_table_name: The name of the source table
        """
        self.source_df = source_df
        self.result_df = result_df
        self.source_table_name = source_table_name

    def get_source(self, attr) -> Series | None:
        """
        Get an attribute from the source DataFrame
        """
        # GroupBy
        if self.source_df is None or attr not in self.source_df:
            return None
        return self.source_df[attr]

    def get_result(self, attr) -> Series | None:
        """
        Get an attribute from the result DataFrame
        """
        return self.result_df[attr]

    def get_source_name(self) -> str:
        """
        Get the name of the source table
        """
        return self.source_table_name
