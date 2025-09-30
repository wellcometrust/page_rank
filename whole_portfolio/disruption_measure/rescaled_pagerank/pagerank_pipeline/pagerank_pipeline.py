import argparse
import asyncio
from datetime import datetime, timezone

import awswrangler as wr
import polars as pl

from .edge_list_loader.data_processor import PageRankDataProcessor
from .graph_metrics.run_pagerank import PageRank
from .graph_metrics.time_normalise import TimeNormalise


def parse_args():
    parser = argparse.ArgumentParser(description='Run the PageRank pipeline')
    parser.add_argument(
        '--resume-locally',
        action='store_true',
        help='Resume processing from local files',
    )
    parser.add_argument(
        '--save-locally',
        action='store_true',
        help='Save processed data locally',
    )
    parser.add_argument(
        '--iterations',
        default=1000,
        help='Max number of iterations for PageRank',
    )
    parser.add_argument(
        '--damping',
        default=0.5,
        help='Damping factor for PageRank',
    )
    parser.add_argument(
        '--epsilon',
        default=1e-13,
        help='Convergence tolerance for PageRank',
    )
    parser.add_argument(
        '--out-degree',
        default='out_degree',
        help='Column name for out degree',
    )
    parser.add_argument(
        '--date',
        default='date',
        help='Column name for date',
    )
    parser.add_argument(
        '--aggregation',
        default='quarter',
        help='Aggregation period for time normalisation',
    )
    parser.add_argument(
        '--field',
        default='page_rank',
        help='Field to normalize in time normalisation',
    )
    parser.add_argument(
        '--filter-out-degree',
        action='store_true',
        help='Filter out degree for time normalisation',
    )
    parser.add_argument(
        '--time-normalise',
        action='store_true',
        help='Apply time normalisation',
    )
    parser.add_argument(
        '--output-base-path',
        default='s3://datalabs-data/funding_impact_measures/page_rank/processed_data/',
        help='Base S3 path for non-test outputs; a timestamped folder will be created inside',
    )
    parser.add_argument(
        '--date-cutoff',
        default=None,
        help='Cutoff date for filtering data',
    )
    parser.add_argument(
        '--info-prefix',
        default='dimensions_2024_04/nodes/publications/publications/article/',
        help='S3 prefix for the info data',
    )
    parser.add_argument(
        '--edges-prefix',
        default='dimensions_2024_04/edges/Publication_CITED_BY_Publication/Article/',
        help='S3 prefix for the edges data',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run pipeline in test mode (PRE year=2000, test bucket)',
    )
    return parser.parse_args()


def save_to_s3(df, path):
    df = df.to_pandas()
    wr.s3.to_parquet(df=df, path=path, dataset=False)


def clean_df(df):
    df = df.select(
        [
            'dimensions_publication_id',
            'page_rank',
            'in_degree',
            'out_degree',
            'relative_citation_ratio',
            'for',
            'date',
            'rescaled_pr',
            'nn_rescaled_pr',
        ]
    )
    return df.drop_nulls(subset=[pl.col('rescaled_pr')])


async def load_data(handler, resume_locally):
    if resume_locally:
        df = pl.read_parquet('data/processed_edge_data.parquet')
        info_df = pl.read_parquet('data/processed_info_data.parquet')
    else:
        df, info_df = await handler.process_data()
    return df, info_df


async def main(args):
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    base = args.output_base_path.rstrip('/')
    run_dir = f'{base}/{ts}'
    if args.test:
        info_prefix = (
            'dimensions_2024_04/nodes/publications/publications/article/year=2000/'
        )
        edges_prefix = 'dimensions_2024_04/edges/Publication_CITED_BY_Publication/Article/CITED_BY_2000_*.parquet'
        full_output_path = f'{run_dir}/test/pr_norm_full.parquet'
        clean_output_path = f'{run_dir}/test/pr_norm_clean.parquet'
        non_norm_output_path = f'{run_dir}/test/pr_non_norm.parquet'
    else:
        info_prefix = args.info_prefix
        edges_prefix = args.edges_prefix
        full_output_path = f'{run_dir}/pr_norm_full.parquet'
        clean_output_path = f'{run_dir}/pr_norm_clean.parquet'
        non_norm_output_path = f'{run_dir}/pr_non_norm.parquet'

    handler = PageRankDataProcessor(
        bucket_name='datalabs-data',
        info_prefix=info_prefix,
        edges_prefix=edges_prefix,
        chunks=16,
        edge_path='data/processed_edge_data.parquet',
        info_path='data/processed_info_data.parquet',
        save_locally=args.save_locally,
        date_cutoff=args.date_cutoff,
    )
    df, info_df = await load_data(handler, args.resume_locally)

    pagerank_processor = PageRank(
        df=df,
        iterations=args.iterations,
        damping=args.damping,
        epsilon=args.epsilon,
    )
    df = pagerank_processor.process_pagerank()
    if df is not None:
        df = df.join(info_df, on='dimensions_publication_id')

    if args.time_normalise:
        normaliser = TimeNormalise(
            df,
            out_degree=args.out_degree,
            date=args.date,
            aggregation=args.aggregation,
            field=args.field,
            filter_out_degree=args.filter_out_degree,
        )
        df = normaliser.process_normalisation()
        save_to_s3(df, full_output_path)
        df_clean = clean_df(df)
        save_to_s3(df_clean, clean_output_path)
    else:
        save_to_s3(df, non_norm_output_path)


if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args=args))
