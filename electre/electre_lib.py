import pandas as pd
from copy import deepcopy
import numpy as np
from jinja2 import Template

debug = False
distillation_counter = 0

result_template = Template("""
<html>
<script src="https://unpkg.com/mermaid@7.1.2/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>
<div class="mermaid">
    graph TD
{% for node_label in nodes %}{% set node = nodes[node_label] %}{% if not node.hidden %}
{% for after_node in node.after %}{% if not after_node.hidden %}
        {{node.id}}({{node.label}}) --> {{after_node.id}}({{after_node.label}})
{% endif %}{% endfor %}{% endif %}{% endfor %}
</div>
</html>""")

def read_csv(file_path):
    """ Reads CSV file into pandas dataframe

    Args:
        file_path (string) : path to CSV file
    Kwargs:
        None
    Returns:
        pandas dataframe. CSV contents

    """
    return pd.read_csv(file_path, index_col=0)

class electre(object):

    def __init__(self, df_alternatives, df_thresholds_weights):
        """ Constructs electre class for running the ELECTRE III process

        Args:
            df_alternatives (Dataframe) : alternatives with criteria measures
            df_thresholds_weights (Dataframe) : thresholds and weights per criteria
        Kwargs:
            None
        Returns:
            electre instance
        """
        if debug:
            self.writer = pd.ExcelWriter('matrices.xlsx')

        self.alternatives = df_alternatives
        self.criteria = df_alternatives.columns
        self.thresholds, self.weights = self._separate_thresholds_weights(df_thresholds_weights)
        self.weights = self._normalize_weights()
        self.concordance_matrices, self.discordance_matrices = self.calculate_partial_concordance_discordance()
        self.global_concordance = self.calculate_global_concordance()
        self.credibility = self.calculate_credibility()
        self.descending_order = self.distillation()
        self.ascending_order = self.distillation(ascending=True)
        self.final_ranking_matrix = self.compute_final_ranking_matrix()
        self.nodes = self.compute_final_order()

        if debug:
            self.writer.save()
            self.writer.close()

    def write_results(self):
        """ Creates visual HTML file of resulting order
        Args:
            None
        Kwargs:
            None
        Returns:
            string. path to result file
        """
        result_path = 'order.html'
        with open(result_path, 'w') as fh:
            fh.write(result_template.render(nodes=self.nodes))
        return result_path

    def compute_final_order(self):
        """ Returns final order

        Args:
            None
        Kwargs:
            None
        Returns:
            dict. dictionary of alternative_nodes
        """
        nodes = dict() # not in order
        for alternative in self.alternatives.index:
            nodes[alternative] = self.alternative_node(alternative)
        final_ranking_matrix_dict = self.final_ranking_matrix.to_dict()
        for alternative_a in final_ranking_matrix_dict:
            alternative_a_relationships = final_ranking_matrix_dict[alternative_a]
            for alternative_b in alternative_a_relationships:
                if alternative_a_relationships[alternative_b] == 'P-':
                    nodes[alternative_a].add_after(nodes[alternative_b])
                elif alternative_a_relationships[alternative_b] == 'I':
                    nodes[alternative_a].append_name(nodes[alternative_b].id)
                    if not nodes[alternative_a].hidden:
                        # Want to leave one visible
                        nodes[alternative_b].hide()
                    nodes[alternative_a].add_equals(nodes[alternative_b])
        self._remove_duplicate_mappings(nodes)
        return nodes

    def _remove_duplicate_mappings(self, node_list):
        """ Removes duplicate mappings from all nodes
        Args:
            node_list (dict) : dictionary of alternative_nodes, id : node
        Kwargs:
            None
        Returns:
            list. new list of cleaned up nodes
        """
        for node_label in node_list:
            node = node_list[node_label]
            nodes_after = list()
            for after_node in node.after:
                nodes_after.append(after_node.id)
            for after_node in node.after:
                for after_after_node in after_node.after:
                    if after_after_node.id in nodes_after:
                        try:
                            node.after.remove(after_after_node)
                        except ValueError:
                            pass # The value was already removed
        return node_list

    class alternative_node(object):

        def __init__(self, id):
            filtered_id = id.replace(' ', '_')
            self.after = list()
            self.equals = list()
            self.id = filtered_id
            self.label = filtered_id
            self.hidden = False

        def __str__(self):
            return '{} -> {}'.format(self.id, self.after)

        def __repr__(self):
            return '{} -> {}'.format(self.id, self.after)

        def hide(self):
            """ Sets node to hidden
            Args:
                None
            Kwargs:
                None
            Returns:
                boolean. state of hidden on node
            """
            self.hidden = True
            return self.hidden

        def add_after(self, alternative_node_inst):
            """ Adds node to before list
            Args:
                alternative_node_inst (alternative_node) : node that is ranked after this node
            Kwargs:
                None
            Returns:
                int. number of nodes after this node
            """
            self.after.append(alternative_node_inst)
            return len(self.after)

        def add_equals(self, alternative_node_inst):
            """ Adds node to equals list
            Args:
                alternative_node_inst (alternative_node) : node that is ranked equal to this node
            Kwargs:
                None
            Returns:
                int. number of nodes equal to this node
            """
            self.equals.append(alternative_node_inst)
            return len(self.equals)

        def append_name(self, name_string):
            """ Append string to label
            Args:
                name_string (string) : string to be appended to label
            Kwargs:
                None
            Returns:
                string. new label
            """
            self.label = '{}, {}'.format(self.label, name_string)
            return self.label

    def compute_final_ranking_matrix(self):
        """ Computes final ranking matrix
        Args:
            None
        Kwargs:
            None
        Returns:
            dataframe. final ranking matrix
        """
        descending_ranks = self._compute_ranks(self.descending_order)
        ascending_ranks = self._compute_ranks(self.ascending_order)

        final_order_matrix = dict()
        for alternative_a in self.alternatives.index:
            for alternative_b in self.alternatives.index:
                if alternative_b not in final_order_matrix:
                    final_order_matrix[alternative_b] = dict()
                if alternative_a == alternative_b:
                    final_order_matrix[alternative_b][alternative_a] = '-'
                elif ((ascending_ranks[alternative_a] < ascending_ranks[alternative_b] and
                    descending_ranks[alternative_a] < descending_ranks[alternative_b]) or
                   (ascending_ranks[alternative_a] < ascending_ranks[alternative_b] and
                    descending_ranks[alternative_a] == descending_ranks[alternative_b]) or
                   (ascending_ranks[alternative_a] == ascending_ranks[alternative_b] and
                    descending_ranks[alternative_a] < descending_ranks[alternative_b])):
                    final_order_matrix[alternative_b][alternative_a] = 'P+'
                elif ((ascending_ranks[alternative_a] > ascending_ranks[alternative_b] and
                    descending_ranks[alternative_a] > descending_ranks[alternative_b]) or
                   (ascending_ranks[alternative_a] > ascending_ranks[alternative_b] and
                    descending_ranks[alternative_a] == descending_ranks[alternative_b]) or
                   (ascending_ranks[alternative_a] == ascending_ranks[alternative_b] and
                    descending_ranks[alternative_a] > descending_ranks[alternative_b])):
                    final_order_matrix[alternative_b][alternative_a] = 'P-'
                elif (descending_ranks[alternative_a] == descending_ranks[alternative_b] and
                      ascending_ranks[alternative_a] == ascending_ranks[alternative_b]):
                    final_order_matrix[alternative_b][alternative_a] = 'I'
                elif((ascending_ranks[alternative_a] < ascending_ranks[alternative_b] and
                      descending_ranks[alternative_a] > descending_ranks[alternative_b]) or
                     (ascending_ranks[alternative_a] > ascending_ranks[alternative_b] and
                      descending_ranks[alternative_a] < descending_ranks[alternative_b])):
                    final_order_matrix[alternative_b][alternative_a] = 'R'
                else:
                    print('ERROR: I missed a case')

        final_order_matrix = pd.DataFrame(final_order_matrix)
        if debug:
            final_order_matrix.to_excel(self.writer, sheet_name='final')

        return final_order_matrix

    def _compute_ranks(self, alternative_list):
        """ Returns dictionary of rank order based on ordered list
        Args:
            alternative_list (list) : list from distillation
        Kwargs:
            None
        Returns:
            dict. alternative keys to integer ranks in distillation list provided
        """
        rank_dict = dict()
        list_length = len(alternative_list)
        for rank in range(list_length):
            alternative = alternative_list[rank]
            if isinstance(alternative, tuple):
                for alt in alternative:
                    rank_dict[alt] = rank
            else: #string
                rank_dict[alternative] = rank

        return rank_dict

    def distillation(self, ascending=False):
        """ Performs distillation to identify order
        Args:
            None
        Kwargs:
            ascending (boolean) : order of alternatives
        Returns:
            list. alternatives in order
        """
        order_list = list()
        credibility = deepcopy(self.credibility)
        while len(credibility) > 0:
            alternative_count, next_alternative = self._distill(credibility, ascending=ascending)
            order_list.append(next_alternative)
            if alternative_count > 1:
                next_alternative_list = list(next_alternative)
            else:
                next_alternative_list = next_alternative
            credibility = credibility.drop(next_alternative_list)
            credibility = credibility.drop(next_alternative_list, axis=1)
        if ascending:
            order_list = list(reversed(order_list))

        if debug:
            if ascending:
                sheet_name = 'asc_order'
            else:
                sheet_name = 'dec_order'
            pd.DataFrame(order_list).to_excel(self.writer, sheet_name=sheet_name)

        return order_list

    def _distill(self, matrix, second_pass=False, ascending=False):
        """ Distills the provided matrix and returns next alternative
        Args:
            matrix (Dataframe) : credibility matrix or distilled version
        Kwargs:
            second_pass (boolean) : is a tie being evaluated
            ascending (boolean) : order of alternatives
        Returns:
            (#_values, string or tuple). # of alternatives and alternative label of next alternative(s)
        """
        global distillation_counter
        lambda_max = matrix.values.max()
        S_lambda_max = 0.3 - 0.15*lambda_max
        _lambda = self._compute_lambda(matrix, lambda_max, S_lambda_max)
        S_matrix = 0.3 - 0.15*matrix
        matrix_transpose = matrix.transpose() + S_matrix

        domination_matrix = (matrix > _lambda) & (matrix > matrix_transpose)

        if debug:
            domination_matrix.to_excel(self.writer, sheet_name='domination{}'.format(distillation_counter))
            distillation_counter += 1

        row_sums = domination_matrix.sum(axis=1)
        column_sums = domination_matrix.sum()

        Q_values = row_sums - column_sums

        if not ascending:
            Q_max = Q_values.max()
        else:
            Q_max = Q_values.min()

        max_occurences = (Q_values == Q_max).sum()

        Q_values_dict = Q_values.to_dict()
        if max_occurences == 1:
            for key in Q_values_dict:
                if Q_values_dict[key] == Q_max:
                    return (1, key)
        else:
            tie_alternatives = self._get_occurences(Q_values_dict, Q_max)
            if not second_pass:
                tie_matrix = matrix[tie_alternatives] # keep columns
                tie_matrix = tie_matrix.loc[tie_alternatives] # keep rows
                return self._distill(tie_matrix, second_pass=True)
            else:
                return (len(tie_alternatives), tuple(tie_alternatives))

    def _get_occurences(self, dictionary, value):
        """ Return keys of dictionary where value occurs

        Args:
            dictionary (dict) : dictionary to be searched
            value (anything) : value to be found for keys
        Kwargs:
            None
        Returns:
            list. list of keys where value matched
        """
        result_list = list()
        for key in dictionary:
            if dictionary[key] == value:
                result_list.append(key)
        return result_list

    def _compute_lambda(self, matrix, lambda_max, S_lambda_max):
        """ Returns lambda of the matrix

        Args:
            matrix (Dataframe) : credibility matrix or distilled version
            lambda_max (number) : max value of the above matrix
            S_lambda_max (number) : 0.3 - 0.15 * lambda_max
        Kwargs:
            None
        Returns:
            number. computed lambda
        """
        lambda_diff = lambda_max - S_lambda_max
        acceptable_matrix = matrix[matrix < lambda_diff]
        return acceptable_matrix.fillna(0).values.max()

    def calculate_credibility(self):
        """ Calculates the credibility matrix

        Args:
            None
        Kwargs:
            None
        Returns:
            dataframe. credibility matrix
        """
        global_concordance = deepcopy(self.global_concordance)
        first = True
        for criterion in self.criteria:
            product_value = (1 - self.discordance_matrices[criterion])/(1 - global_concordance)
            deviations_from_concordance = global_concordance < self.discordance_matrices[criterion]
            credibility_multiplier = product_value[deviations_from_concordance]
            # 1's will have no effect in multiplication
            credibility_multiplier = credibility_multiplier.fillna(1)
            if first:
                first = False
                credibility = credibility_multiplier
            else:
                credibility = credibility * credibility_multiplier

        credibility = credibility * global_concordance
        # zero the diagonal
        credibility.values[[np.arange(len(self.alternatives))] * 2] = 0

        if debug:
            credibility.to_excel(self.writer, sheet_name='Credibility')

        return credibility

    def calculate_global_concordance(self):
        """ Calculates the global concordance matrix

        Args:
            None
        Kwargs:
            None
        Returns:
            dataframe. Global concordance matrix
        """
        first = True
        for criterion in self.criteria:
            next_matrix = self.weights[criterion]*self.concordance_matrices[criterion]
            if first:
                first = False
                global_matrix = next_matrix
            else:
                global_matrix = global_matrix + next_matrix

        if debug:
            global_matrix.to_excel(self.writer, sheet_name='GC')

        return global_matrix

    def _separate_thresholds_weights(self, df_thresholds_weights):
        """ Returns separate dictionaries for thresholds and weights

        Args:
            df_thresholds_weights (Dataframe) : Contains both thresholds and weights
        Kwargs:
            None
        Returns:
            tuple - (dict, dict). (thresholds, weights). Dictionaries are keyed by criteria.
            Thresholds then keyed by threshold
        """
        thresholds_dict = df_thresholds_weights.to_dict()
        weights_dict = dict()
        for criterion_label in thresholds_dict:
            criterion_fields_dict = thresholds_dict[criterion_label]
            weights_dict[criterion_label] = criterion_fields_dict['Weights']
            del criterion_fields_dict['Weights']

        return thresholds_dict, weights_dict

    def _normalize_weights(self):
        """ Normalizes weights to sum of 1
        Args:
            None
        Kwargs:
            None
        Returns:
            dict. New weights keyed by criteria
        """
        weight_sum = 0.0
        for key in self.weights:
            weight_sum += self.weights[key]

        if weight_sum != 1.0:
            new_weights = dict()
            for key in self.weights:
                new_weights[key] = self.weights[key]/weight_sum
            return new_weights
        else:
            return self.weights

    def calculate_partial_concordance_discordance(self):
        """ Calculates the concordance and discordance matrices

        Args:
            None
        Kwargs:
            None
        Returns:
            tuple (dict, dict). concordance, discordance.
            Dictionary by criteria of dataframe concordance matrices
        """
        concordance_matrices = dict()
        discordance_matrices = dict()

        alternative_scores = self.alternatives.to_dict()

        for criterion in self.alternatives:
            diff_matrix = self._calculate_diff_matrix(criterion)
            preference_threshold = self.thresholds[criterion]['P']
            weak_preference_threshold = self.thresholds[criterion]['Q']
            veto_threshold = self.thresholds[criterion]['V']
            partial_agreement_dict = dict()
            partial_discord_dict = dict()
            for alternative_column in diff_matrix.columns:
                for alternative_row in diff_matrix.columns: # diff matrix is square, same rows, same columns
                    concordance_value = self._concordance_piecewise(diff_matrix.get_value(alternative_column,
                                                                                      alternative_row),
                                                                preference_threshold,
                                                                weak_preference_threshold,
                                                                alternative_scores[criterion][alternative_column],
                                                                alternative_scores[criterion][alternative_row])
                    discordance_value = self._discordance_piecewise(diff_matrix.get_value(alternative_column,
                                                                                      alternative_row),
                                                                preference_threshold,
                                                                veto_threshold,
                                                                alternative_scores[criterion][alternative_column],
                                                                alternative_scores[criterion][alternative_row])
                    if alternative_row not in partial_agreement_dict:
                        partial_agreement_dict[alternative_row] = dict()
                        partial_discord_dict[alternative_row] = dict()
                    partial_agreement_dict[alternative_row][alternative_column] = concordance_value
                    partial_discord_dict[alternative_row][alternative_column] = discordance_value
            concordance_matrices[criterion] = pd.DataFrame(partial_agreement_dict)
            discordance_matrices[criterion] = pd.DataFrame(partial_discord_dict)
            if debug:
                concordance_matrices[criterion].to_excel(self.writer, sheet_name='C_'+criterion)
                discordance_matrices[criterion].to_excel(self.writer, sheet_name='D_'+criterion)

        return concordance_matrices, discordance_matrices

    def _discordance_piecewise(self, score_difference, preference_threshold, veto_threshold,
                                      alternative_a_score, alternative_b_score):
        """ Returns result of discordance piecewise on value
        Args:
            score_difference (number) : difference between alternative's scores for the criterion
            preference_threshold (number) : criterion threshold for strict preference
            veto_threshold (number) : criterion threshold for veto
            alternative_a_score (number) : alternative A score for criterion
            alternative_b_score (number) : alternative B score for criterion
        Kwargs:
            None
        Returns:
            number. Discordance calculated number
        """
        if score_difference < preference_threshold:
            return 0
        elif score_difference >= preference_threshold and score_difference < veto_threshold:
            return ((-1*preference_threshold - alternative_a_score + alternative_b_score) /
                    (veto_threshold - preference_threshold))
        else: # >= veto_threshold
            return 1

    def _concordance_piecewise(self, score_difference, preference_threshold, weak_preference_threshold,
                                      alternative_a_score, alternative_b_score):
        """ Returns result of concordance piecewise on value
        Args:
            score_difference (number) : difference between alternative's scores for the criterion
            preference_threshold (number) : criterion threshold for strict preference
            weak_preference_threshold (number) : criterion threshold for weak preference
            alternative_a_score (number) : alternative A score for criterion
            alternative_b_score (number) : alternative B score for criterion
        Kwargs:
            None
        Returns:
            number. Concordance calculated number
        """
        if score_difference >= preference_threshold:
            return 0
        elif score_difference < preference_threshold and score_difference >= weak_preference_threshold:
            return ((preference_threshold - alternative_b_score + alternative_a_score) /
                    (preference_threshold - weak_preference_threshold))
        else: # less than preference but not greater than weak
            return 1

    def _calculate_diff_matrix(self, criterion_label):
        """ Returns the diff matrix for the purposes of calculating the concordance matrices

        Args:
            criterion_label (string) : criterion being calculated
        Kwargs:
            None
        Returns:
            Dataframe. Matrix of alternative diffs for criterion
        """
        diff_matrix = pd.DataFrame()
        for (alternative, score) in zip(self.alternatives.index, self.alternatives[criterion_label]):
            diff_matrix[alternative] = score - self.alternatives[criterion_label]
        return diff_matrix


