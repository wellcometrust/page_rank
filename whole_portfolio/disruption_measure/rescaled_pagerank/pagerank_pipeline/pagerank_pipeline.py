import argparse
import asyncio
from datetime import datetime, timezone

import awswrangler as wr
import polars as pl

from .edge_list_loader.data_processor import PageRankDataProcessor
from .graph_metrics.run_pagerank import PageRank
from .graph_metrics.time_normalise import TimeNormalise


def parse_args():
    parser = argparse.ArgumentParser()
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
        default='dimensions_2025_05/publications/output/*/publications/*.parquet',
        help='S3 prefix for the info data',
    )
    parser.add_argument(
        '--edges-prefix',
        default='dimensions_2025_05/publications/output/*/pubs_references/*.parquet',
        help='S3 prefix for the edges data',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run pipeline in test mode (PRE year=2000, test bucket)',
    )
    parser.add_argument(
        '--info-publication-id',
        default='id',
        help='Publication ID field for the info data',
    )
    parser.add_argument(
        '--edge-publication-id-target',
        default='pub_id',
        help='Publication ID field for the target in the edge data',
    )
    parser.add_argument(
        '--edge-publication-id-source',
        default='references',
        help='Publication ID field for the source in the edge data',
    )
    parser.add_argument(
        '--info-date-field',
        default='publication_date',
        help='Date field for the info data',
    )
    parser.add_argument(
        '--info-publication-type-field',
        default='publication_type',
        help='Publication type field for the info data',
    )
    parser.add_argument(
        '--publication-type-value',
        default='article',
        help='Value to filter the publication type field on',
    )
    parser.add_argument(
        '--edge_local_path',
        default='data/processed_edge_data.parquet',
        help='Local path to store processed edge data',
    )
    parser.add_argument(
        '--info_local_path',
        default='data/processed_info_data.parquet',
        help='Local path to store processed info data',
    )
    return parser.parse_args()


def save_to_s3(df, path):
    df = df.to_pandas()
    wr.s3.to_parquet(df=df, path=path, dataset=False)


def clean_df(df):
    df = df.select(
        [
            'id',
            'page_rank',
            'in_degree',
            'out_degree',
            args.info_date_field,
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
        info_prefix = 'dimensions_2025_05/publications/output/*/publications/0.parquet'
        edges_prefix = (
            'dimensions_2025_05/publications/output/*/pubs_references/0.parquet'
        )
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
        edge_path=args.edge_local_path,
        info_path=args.info_local_path,
        save_locally=args.save_locally,
        date_cutoff=args.date_cutoff,
        info_publication_id=args.info_publication_id,
        edge_publication_id_target=args.edge_publication_id_target,
        edge_publication_id_source=args.edge_publication_id_source,
        info_date_field=args.info_date_field,
        info_publication_type_field=args.info_publication_type_field,
        publication_type_value=args.publication_type_value,
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
        df = df.join(info_df, on='id')

    if args.time_normalise:
        normaliser = TimeNormalise(
            df,
            out_degree=args.out_degree,
            date=args.info_date_field,
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
