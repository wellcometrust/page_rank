import os
import random

import polars as pl


class PageRankDataProcessor:
    """Processes data ready to be loaded into graph.

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
        edge_publication_id_citing='pub_id',
        edge_publication_id_cited='references',
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
        self.edge_publication_id_citing = edge_publication_id_citing
        self.edge_publication_id_cited = edge_publication_id_cited
        self.info_date_field = info_date_field
        self.info_publication_type_field = info_publication_type_field
        self.publication_type_value = publication_type_value

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
            pl.col(self.info_publication_type_field).is_in(
                [self.publication_type_value]
            )
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
                pl.col(self.edge_publication_id_citing),
                pl.col(self.edge_publication_id_cited),
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
        if self.info_df is not None:
            self.info_df = self.info_df.with_columns(
                pl.col(self.info_date_field)
                .map_elements(self.fill_date, return_dtype=pl.Utf8)
                .str.strptime(pl.Date, '%Y-%m-%d', strict=False)
                .alias(self.info_date_field)
            ).drop_nulls(subset=self.info_date_field)

    def filter_date_cutoff(self):
        """Filters the dataframe to only include publications before a certain date."""
        if self.date_cutoff is not None and self.info_df is not None:
            cutoff = pl.lit(self.date_cutoff).str.strptime(pl.Date, '%Y-%m-%d')
            date_col = (
                '_min_date'
                if '_min_date'
                in (
                    self.info_df.schema
                    if isinstance(self.info_df, pl.LazyFrame)
                    else self.info_df.columns
                )
                else self.info_date_field
            )
            self.info_df = self.info_df.filter(pl.col(date_col) < cutoff)

    def prepare_min_date(self):
        """Create a deterministic minimal date column (_min_date) from partial date strings.
        This is used solely for temporal edge pruning to avoid randomness affecting DAG determination.
        """
        if self.info_df is not None:
            year = pl.col(self.info_date_field).str.slice(0, 4)
            month = pl.when(pl.col(self.info_date_field).str.len_chars() >= 7)
            month = month.then(pl.col(self.info_date_field).str.slice(5, 2)).otherwise(
                pl.lit('01')
            )
            day = pl.when(pl.col(self.info_date_field).str.len_chars() >= 10)
            day = day.then(pl.col(self.info_date_field).str.slice(8, 2)).otherwise(
                pl.lit('01')
            )

            self.info_df = self.info_df.with_columns(
                (year + pl.lit('-') + month + pl.lit('-') + day)
                .str.strptime(pl.Date, '%Y-%m-%d', strict=False)
                .alias('_min_date')
            )

    def remove_edges_before_publish_date(self):
        """This function removes impossible edges which exist due to
        data quality issues in the bulk extract. This should make the
        graph acyclic. Still check later with graph-tool is_DAG.
        """
        if self.info_df is not None and self.df is not None:
            info_min = self.info_df.select(
                [
                    pl.col(self.info_publication_id).alias('pid'),
                    pl.col('_min_date'),
                ]
            )

            self.df = self.df.join(
                info_min.rename({'pid': 'citing_pid', '_min_date': 'citing_min_date'}),
                left_on=self.edge_publication_id_citing,
                right_on='citing_pid',
                how='inner',
            ).join(
                info_min.rename({'pid': 'cited_pid', '_min_date': 'cited_min_date'}),
                left_on=self.edge_publication_id_cited,
                right_on='cited_pid',
                how='inner',
            )

            self.df = self.df.filter(
                pl.col('cited_min_date') < pl.col('citing_min_date')
            )

            self.df = self.df.select(
                [
                    pl.col(self.edge_publication_id_citing),
                    pl.col(self.edge_publication_id_cited),
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
        """Rename original id columns to generic source/target for graph construction."""
        if self.df is not None:
            self.df = self.df.rename(
                {
                    self.edge_publication_id_citing: 'source',
                    self.edge_publication_id_cited: 'target',
                }
            )

    def process_data(self):
        """Methods to run data_processor:
        - load_publication_info
        - load_graph_data
        - prepare_min_date & filter_date_cutoff (deterministic minimal date)
        - removes not possible edges (using minimal dates)
        - clean_date (random date augmentation applied AFTER pruning)
        - converts dimensions ids to numeric version
        - defines source and target for graph construction
        - optional saving of processed data
        """
        self.info_df = pl.scan_parquet(f's3://datalabs-data/{self.info_prefix}')
        self.df = pl.scan_parquet(f's3://datalabs-data/{self.edges_prefix}')

        self.info_df = self.pub_info_polars_args(self.info_df)
        self.df = self.graph_data_polars_args(self.df)

        self.prepare_min_date()
        if self.date_cutoff:
            self.filter_date_cutoff()

        self.remove_edges_before_publish_date()

        self.clean_date()

        self.df = PageRankDataProcessor.ids_to_numeric(
            self.df, self.edge_publication_id_cited
        )
        self.df = PageRankDataProcessor.ids_to_numeric(
            self.df, self.edge_publication_id_citing
        )
        self.info_df = PageRankDataProcessor.ids_to_numeric(
            self.info_df, self.info_publication_id
        )

        self.define_source_target()

        self.info_df = self.info_df.collect(streaming=True)
        self.df = self.df.collect(streaming=True)
        if self.save_locally:
            if self.edge_path is not None:
                os.makedirs(os.path.dirname(self.edge_path), exist_ok=True)
                self.df.write_parquet(self.edge_path)
            if self.info_path is not None:
                self.info_df.write_parquet(self.info_path)
        return self.df, self.info_df
