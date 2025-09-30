import asyncio
import io

import aioboto3
import awswrangler as wr
import polars as pl
from tqdm import tqdm


class AsyncS3DataLoader:
    """General use asynchronous s3 data loader

    Attributes:
        bucket_name: str - Name of the s3 bucket
        prefix: str - AWS prefix of file location
        chunks: int - Number of chunks to process the data in
        polars_args: function/method - list of arguments to passed to polars read
        tqdm_desc: str - Description to be shown in tqdm progress bar
    """

    def __init__(self, bucket_name, prefix, chunks, polars_args=None, tqdm_desc=''):
        """Initialises asynchronous s3 data loader
        Args:
            bucket_name: str - Name of the s3 bucket
            prefix: str - AWS prefix of file location
            chunks: int - Number of chunks to process the data in
            polars_args: function/method - list of arguments to passed to polars read
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.chunks = chunks
        self.path_list = wr.s3.list_objects(
            path=f's3://{self.bucket_name}/{self.prefix}'
        )
        self.polars_args = polars_args
        self.tqdm_desc = tqdm_desc

    async def read_from_s3(self, s3_client, full_path):
        """Reads a file from s3

        First extracts s3 key from the bucket to be processed by the client
        Retreives and loads data asynchronously into memory
        Data is then read sequentially by polars with any arguments applied

        Args:
            s3_client: An aioboto3.Session()
            full_path: The full directory path of the file on s3

        Returns:
            Polars dataframe
        """
        key = full_path.replace(f's3://{self.bucket_name}/', '')
        try:
            response = await s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = await response['Body'].read()
            df = pl.read_parquet(io.BytesIO(data))
            if self.polars_args:
                df = self.polars_args(df)
            else:
                new_cols = []
                for col in df.columns:
                    dtype = df.schema[col]
                    if isinstance(dtype, pl.List):
                        new_cols.append(pl.col(col).list.join(',').alias(col))
                    elif isinstance(dtype, pl.Struct):
                        new_cols.append(pl.col(col).struct.json_encode().alias(col))
                    else:
                        new_cols.append(
                            pl.col(col).cast(pl.String, strict=False).alias(col)
                        )
                df = df.with_columns(new_cols)

            return df
        except s3_client.exceptions.NoSuchKey:
            return None

    async def process_chunks(self, s3_client, chunk_list):
        """Processes a chunk of paths to be read_from_s3

        Args:
            s3_client: An aioboto3.Session()
            chunk_list: The full directory of all files in a chunk

        Returns:
            A list of Polars dataframes
        """
        tasks = [
            self.read_from_s3(s3_client=s3_client, full_path=full_path)
            for full_path in chunk_list
        ]
        results = await asyncio.gather(*tasks)
        return [df for df in results if df is not None]

    async def async_chunk_run(self):
        """Chunks data, processes, and save to output_dir

        Returns:
            Unified Polars dataframe
        """
        all_dataframes = []
        async with aioboto3.Session().client('s3') as s3_client:
            for i in tqdm(
                range(0, len(self.path_list), self.chunks),
                desc=f'Loading {self.tqdm_desc} chunked data from s3',
            ):
                chunk_list = self.path_list[i : i + self.chunks]
                chunk_dataframes = await self.process_chunks(s3_client, chunk_list)
                if chunk_dataframes:
                    df = pl.concat(chunk_dataframes)
                    all_dataframes.append(df)
        return pl.concat(all_dataframes)
