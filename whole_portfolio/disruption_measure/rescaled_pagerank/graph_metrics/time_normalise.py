import polars as pl
from sklearn.preprocessing import MinMaxScaler


class TimeNormalise:
    """Takes a polars dataframe and a given field and time normalises it

    Attributes:
        df - a Polars dataframe
        out_degree - a column in a Polars dataframe with an int out_degree of a pub
        date - a string date of pub published date
        aggregation - period to aggregate, year, quarter, month, minimum_value
        field - a metric to time normalise i.e., pagerank in a polars column
    """

    AGGREGATIONS = ['year', 'quarter', 'month', 'minimum_valid']

    def __init__(
        self, df, out_degree, date, aggregation, field, filter_out_degree=False
    ):
        """Initialises TimeNormalise class. Aggregation must be valid.

        Args:
            df - a Polars dataframe
            out_degree - a column in a Polars dataframe with an int out_degree of a pub
            date - a string date of pub published date
            aggregation - period to aggregate, year, quarter, month, minimum_value
            field - a metric to time normalise i.e., pagerank in a polars column

        Returns:
            None
        """
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(f'Invalid aggregation, choose one of {self.AGGREGATIONS}')
        self.df = df
        self.out_degree = out_degree
        self.date = date
        self.aggregation = aggregation
        self.field = field
        self.filter_out_degree = filter_out_degree

    def filter_zero_out_degree(self):
        """Removes 0 out_degree from list to not impact normalisation

        Returns:
            Filtered Polars dataframe without 0 out degree
        """
        return self.df.filter(pl.col(self.out_degree) > 0)

    def create_month(self):
        """Creates a month column (str) from a polars date col"""
        return self.df.with_columns(
            pl.col(self.date).dt.strftime('%Y-%m').alias('month')
        )

    def create_quarter(self):
        """Create a year-quarter column from a polars date col"""
        return self.df.with_columns(
            pl.concat_str(
                [
                    pl.col(self.date).dt.strftime('%Y'),
                    pl.col(self.date).dt.quarter().cast(pl.String),
                ],
                separator='-',
            ).alias('quarter')
        )

    def find_max_period(self):
        """Finds the maximum number of papers in a given date aggregation

        Returns:
            Integer of paper count
        """
        return (
            self.df.group_by(self.aggregation)
            .len()
            .max()
            .select(pl.first('len'))
            .item()
        )

    def compute_rescaled_scores(self, window_size):
        """Computes rolling mean, deviation, rolling std, and z scores for a
        given metric. This is computed on a date ordered dataframe overa given window.

        Args:
            window_size: int - given window to aggregate over

        Rerturns:
            Enriched dataframe with rescaled_pr (z score) as well as the mean and std.
        """
        self.df = self.df.sort(self.date)
        result_df = self.df.with_columns(
            [
                pl.col(self.field)
                .rolling_mean(window_size, center=True)
                .alias('rolling_mean')
            ]
        )
        result_df = result_df.with_columns(
            [
                (pl.col(self.field) - pl.col('rolling_mean')).alias('deviation'),
                pl.col(self.field)
                .rolling_std(window_size, center=True)
                .alias('rolling_std'),
            ]
        ).with_columns(
            [(pl.col('deviation') / pl.col('rolling_std')).alias('z_scores')]
        )

        z_scores = result_df['z_scores'].fill_nan(None)
        rolling_means = result_df['rolling_mean']
        rolling_stds = result_df['rolling_std']

        self.df = self.df.with_columns(
            [
                z_scores.alias('rescaled_pr'),
                rolling_means.alias('rescaled_mean_pr'),
                rolling_stds.alias('rescaled_std_pr'),
            ]
        )

        return self.df.drop_nulls(subset=['rescaled_pr'])

    def compute_zero_one_scores(self):
        """Converts rescaled scores to 0 -1. This avoid papers having negative
        scores which is unintuitive for analysis. Small constant added to avoid 0
        values to allow for potential log transformation if desired. Distance should
        be maintained via this method.

        Returns:
            Enriched polars dataframe with non negative rescaled scores 'nn_rescaled_pr'
        """
        scaler = MinMaxScaler()
        rescaled_values = scaler.fit_transform(
            self.df['rescaled_pr'].to_numpy().reshape(-1, 1)
        )
        self.df = self.df.with_columns(
            pl.Series('nn_rescaled_pr', rescaled_values.flatten())
        )
        self.df = self.df.with_columns(
            (pl.col('nn_rescaled_pr') + 1e-13).alias('nn_rescaled_pr')
        )
        return self.df

    def process_normalisation(self):
        """Processes TimeNormalisation class"""
        if self.filter_out_degree:
            self.df = self.filter_zero_out_degree()
        if self.aggregation == 'month':
            self.df = self.create_month()
        if self.aggregation == 'quarter':
            self.df = self.create_quarter()

        window_size = self.find_max_period()
        print(f'Aggregation at rolling {window_size}')
        self.df = self.compute_rescaled_scores(window_size)
        self.df = self.compute_zero_one_scores()
        return self.df
