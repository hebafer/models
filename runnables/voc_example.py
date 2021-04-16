import argparse
import mars.dataframe as md
import os
import pandas as pd

from run_io.db_adapter import convertDSNToRfc1738
from sqlalchemy import create_engine

import numpy as np

def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", type=str, required=False)
    return parser

if __name__ == "__main__":
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    output_tables = output.split(',')
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    assert len(output_tables) == 1, "The output tables shouldn't be null and can contain only one."

    print("Connecting to database...")
    url = convertDSNToRfc1738(datasource, args.dataset)
    engine = create_engine(url)

    print("Printing result from SELECT statement as DataFrame...")
    input_md = md.read_sql(
        sql=select_input,
        con=engine)
    input_md.execute()
    print(input_md)

    print("Persist the statement into the table {}".format(output_tables[0]))
    result_table = input_md.to_sql(
        name=output_tables[0],
        con=engine,
        index=False
    )
    result_table.execute()