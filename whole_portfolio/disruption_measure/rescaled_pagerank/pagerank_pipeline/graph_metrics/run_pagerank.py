import polars as pl
from graph_tool.all import Graph, is_DAG, pagerank


class PageRank:
    """Takes an edge list and calculates PageRank

    Attributes:
        df: A polars dataframe (edge list)
        iterations: int - max number of iterations to run
        damping: float - damping (telportation) for PageRank
        epsilon: float - epsilon for PageRank to determine convergence
    """

    def __init__(self, df, iterations, damping, epsilon):
        """Initialises PageRank class

        Args:
            df: A polars dataframe (edge list)
            iterations: int - max number of iterations to run
            damping: float - damping (telportation) for PageRank
            epsilon: float - epsilon for PageRank to determine convergence

        Returns:
            None
        """
        self.df = df
        self.iterations = iterations
        self.damping = damping
        self.epsilon = epsilon

    def load_graph(self):
        """Loads polars dataframe (edge list) into a graph

        Returns:
            g - A graph-tool directed graph
            hash_map - VertexPropertyMap of hashed graph vertices to publication ids
        """
        g = Graph(directed=True)
        edge_list = self.df.to_numpy()
        hash_map = g.add_edge_list(edge_list, hashed=True)
        return g, hash_map

    def run_pagerank(self, g, pers=None):
        """Calculates PageRank for a given graph
        Uses graph-tool library

        Args:
            g - A graph-tool directed graph
            pers - Optional extra personalisation arguments

        Returns:
            page_rank_scores - VertexPropertyMap of pageranks for each node
            has_converged - number of iterations to converge or max
        """
        page_rank_scores, has_converged = pagerank(
            g,
            damping=self.damping,
            pers=pers,
            weight=None,
            prop=None,
            epsilon=self.epsilon,
            max_iter=self.iterations,
            ret_iter=True,
        )
        return page_rank_scores, has_converged

    def get_hashed_ids(self, g, hash_map):
        """Converts hashed vertices to publication ids

        Args:
            g - a graph-tool graph
            hash_map - graph-tool map between hashed ids and ids

        Returns:
            list of publication_ids in order
        """
        return [hash_map[v] for v in g.vertices()]

    def get_in_degree(self, g):
        """Calculates in degree of a graph (citations)

        Args:
            g - a graph-tool graph
        Returns:
            list of in_degrees in order
        """
        return g.get_in_degrees(g.get_vertices())

    def get_out_degree(self, g):
        """Calculates out degree of a graph (references)

        Args:
            g - a graph-tool graph

        Returns:
            list of out degrees in order
        """
        return g.get_out_degrees(g.get_vertices())

    def get_pagerank_scores(self, g, page_rank_scores):
        """Returns pagerank scores from a VertexPropertyMap

        Args:
            g - a graph-tool graph
            page_rank_scores - a VertexPropertyMap

        Returns:
            list of pagerank scores in order
        """
        return [page_rank_scores[v] for v in g.vertices()]

    def combine_into_dataframe(self, g, hash_map, page_rank_scores):
        """Combines order list of metrics into a polars dataframe

        Args:
            g - a graph-tool graph
            hash_map - VertexPropertyMap of hashed vertices to pub ids
            page_rank_scores - VertexPropertyMap of pagerank scores

        Returns:
            Polars dataframe of given metrics
        """
        return pl.DataFrame(
            {
                'id': self.get_hashed_ids(g, hash_map),
                'page_rank': self.get_pagerank_scores(g, page_rank_scores),
                'in_degree': self.get_in_degree(g),
                'out_degree': self.get_out_degree(g),
            }
        )

    def process_pagerank(self):
        """Processes pagerank class if constructed graph as a DAG

        Retuns:
            Enriched PageRank polars dataframe or None
        """
        g, hash_map = self.load_graph()
        if is_DAG(g):
            page_rank_scores, has_converged = self.run_pagerank(g)
            print(f'Pagerank converged after {has_converged} iterations')
            return self.combine_into_dataframe(g, hash_map, page_rank_scores)
        else:
            print('Graph is not a DAG, please investigate the data source')
            return None
