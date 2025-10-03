import os
import random

import polars as pl

from ..async_loader.data_loader import AsyncS3DataLoader


class PageRankDataProcessor:
    """Processes data ready to be loaded into graph.
    Works with AsyncS3DataLoader

    Attributes:
        info_prefix: str - AWS prefix for publication info used to filter edges
        edges_prefix: str - AWS prefix for edge list
        bucket_name: str - name of the AWS bucket
        chunks: int - number of chunks split out processing
        edge_path: str - Optional local storage of processed edge list
        info_path: str - Optinal local storage of processed info dataframe
        save_locally: bool - Optional arg whether to save data locally
        date_cutoff: str - Optional date cutoff to filter publications
    """

    def __init__(
        self,
        info_prefix,
        edges_prefix,
        bucket_name,
        chunks,
        edge_path=None,
        info_path=None,
        save_locally=False,
        date_cutoff=None,
        info_publication_id='id',
        edge_publication_id_target='pub_id',
        edge_publication_id_source='references',
        info_date_field='publication_date',
        info_publication_type_field='publication_type',
        publication_type_value='publication',
    ):
        """Initializes the data processor

        Args:
            info_prefix: str - AWS prefix for publication info used to filter edges
            edges_prefix: str - AWS prefix for edge list
            bucket_name: str - name of the AWS bucket
            chunks: int - number of chunks split out processing
            edge_path: str - Optional local storage of processed edge list
            info_path: str - Optinal local storage of processed info dataframe
            save_locally: bool - Optional arg whether to save data locally
            date_cutoff: str - Optional date cutoff to filter publications
        """
        self.info_prefix = info_prefix
        self.edges_prefix = edges_prefix
        self.bucket_name = bucket_name
        self.chunks = chunks
        self.edge_path = edge_path
        self.info_path = info_path
        self.save_locally = save_locally
        self.info_df = None
        self.df = None
        self.date_cutoff = date_cutoff
        self.info_publication_id = info_publication_id
        self.edge_publication_id_source = edge_publication_id_target
        self.edge_publication_id_target = edge_publication_id_source
        self.info_date_field = info_date_field
        self.info_publication_type_field = info_publication_type_field
        self.publication_type_value = publication_type_value

    async def load_publication_info(self):
        """Loads the publication info dataframe"""
        print(self.info_prefix)
        data_loader = AsyncS3DataLoader(
            prefix=self.info_prefix,
            chunks=self.chunks,
            polars_args=self.pub_info_polars_args,
            bucket_name=self.bucket_name,
            tqdm_desc='publication info',
        )

        self.info_df = await data_loader.async_chunk_run()

    async def load_graph_data(self):
        """Loads the edge list needed for pagerank"""
        data_loader = AsyncS3DataLoader(
            prefix=self.edges_prefix,
            chunks=self.chunks,
            polars_args=self.graph_data_polars_args,
            bucket_name=self.bucket_name,
            tqdm_desc='edge list',
        )

        self.df = await data_loader.async_chunk_run()

    def pub_info_polars_args(self, df):
        """Defines specific polars args used for pagerank info dataset.
        Types are handled via cast due to some inconsistencies
        in the bulk extract. Data as pl.String to allow for later
        manipulation.

        Args:
            df - A Polars dataframe

        Returns:
            df - A filtered polars dataframe
        """
        return df.filter(
            pl.col(self.info_publication_type_field) == self.publication_type_value
        ).select(
            [
                pl.col(self.info_publication_id).cast(pl.String, strict=False),
                pl.col(self.info_date_field).cast(pl.String, strict=False),
            ]
        )

    def graph_data_polars_args(self, df):
        """Defines polars args used for pagerank edge list

        Args:
            df - A polars dataframe

        Returns:
            df - A filtered polars dataframe
        """
        return df.select(
            [
                pl.col(self.edge_publication_id_source),
                pl.col(self.edge_publication_id_target),
            ]
        )

    @staticmethod
    def fill_date(date_str):
        """Applies random day and or month for pubs with incomplete publish
        dates. Avoids huge paper clusters at 1st Jan for example.

        Args:
            date_str - column - Date column (str date) of pub publish date

        Returns:
            filled date column
        """
        parts = date_str.split('-')
        if len(parts) == 1:
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            return f'{parts[0]}-{month:02d}-{day:02d}'
        elif len(parts) == 2:
            day = random.randint(1, 28)
            return f'{parts[0]}-{parts[1]}-{day:02d}'
        return date_str

    def clean_date(self):
        """Cleans date calling fill_date and converting output to pl.Date col"""
        self.info_df = self.info_df.with_columns(
            pl.col(self.info_date_field)
            .map_elements(self.fill_date, return_dtype=pl.Utf8)
            .str.strptime(pl.Date, '%Y-%m-%d', strict=False)
            .alias(self.info_date_field)
        ).drop_nulls(subset=pl.col(self.info_date_field))

    def filter_date_cutoff(self):
        """Filters the dataframe to only include publications before a certain date."""
        if self.date_cutoff is not None:
            self.info_df = self.info_df.filter(
                pl.col('date')
                < pl.lit(self.date_cutoff).str.strptime(pl.Date, '%Y-%m-%d')
            )

    def filter_set(self):
        """Keeps only edges which can be linked to publications defined in
        the info set. Practical effect of keeping only research articles.
        """
        filter_ids = self.info_df.get_column(self.info_publication_id)
        self.df = self.df.filter(
            (pl.col(self.edge_publication_id_source).is_in(filter_ids))
            | (pl.col(self.edge_publication_id_target).is_in(filter_ids))
        )

    def remove_edges_before_publish_date(self):
        """This function removes impossible edges which exist due to
        data quality issues in the bulk extract. This should make the
        graph acyclic. Still check later with graph-tool is_DAG.
        """
        self.df = self.df.join(
            self.info_df[[self.info_publication_id, self.info_date_field]],
            left_on=self.edge_publication_id_target,
            right_on=self.info_publication_id,
            how='left',
        )

        info_df_copy = self.info_df.clone().rename(
            {
                self.info_publication_id: f'{self.info_publication_id}1',
                self.info_date_field: f'{self.info_date_field}1',
            }
        )

        self.df = self.df.join(
            info_df_copy[[f'{self.info_publication_id}1', f'{self.info_date_field}1']],
            left_on=self.edge_publication_id_source,
            right_on=f'{self.info_publication_id}1',
            how='left',
        )

        self.df = self.df.filter(
            pl.col(self.info_date_field) < pl.col(f'{self.info_date_field}1')
        )

        self.df = self.df.select(
            [
                pl.col(self.edge_publication_id_source),
                pl.col(self.edge_publication_id_target),
            ]
        )

    @staticmethod
    def ids_to_numeric(df, column_name):
        """Removed 'pub' and turns into an integer
        Used for much faster hashing in graph construction

        Args:
            df - A polars dataframe (edge list)
            column_name - ID column

        Returns:
            df - A polars dataframe with numeric ids
        """
        return df.with_columns(
            pl.col(column_name)
            .str.replace('^pub\\.', '')
            .cast(pl.Int64)
            .alias(column_name)
        )

    def define_source_target(self):
        """Explicitly defines source and target for graph construction.
        PageRank needs to be a directed graph, however via a 'cites' not
        'cited by' relationship i.e., reverse direction of the WAG
        """
        self.df = self.df.select(
            [
                pl.col(self.edge_publication_id_source).alias('source'),
                pl.col(self.edge_publication_id_target).alias('target'),
            ]
        )

    async def process_data(self):
        """Methods to run data_processor:
        - load_publication_info
        - load_graph_data
        - clean_date
        - filter_date_cutoff
        - removes not possible edges
        - converts dimensions ids to numeric version
        - defines source and target for graph construction
        - optional saving of processed data
        """
        await self.load_publication_info()
        await self.load_graph_data()
        self.clean_date()
        if self.date_cutoff:
            self.filter_date_cutoff()
        self.filter_set()
        self.remove_edges_before_publish_date()
        self.df = PageRankDataProcessor.ids_to_numeric(
            self.df, self.edge_publication_id_target
        )
        self.df = PageRankDataProcessor.ids_to_numeric(
            self.df, self.edge_publication_id_source
        )
        self.info_df = PageRankDataProcessor.ids_to_numeric(
            self.info_df, self.info_publication_id
        )
        self.define_source_target()
        if self.save_locally:
            os.makedirs(os.path.dirname(self.edge_path), exist_ok=True)
            self.df.write_parquet(self.edge_path)
            self.info_df.write_parquet(self.info_path)
        return self.df, self.info_df
