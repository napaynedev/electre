import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
import os

from electre import read_csv, electre

class electre_tests(unittest.TestCase):

    def test_read_csv(self):
        df = read_csv('test/example_thresholds_weights.csv')
        self.assertEqual(type(df), type(pd.DataFrame()))

        dict_df = df.to_dict()
        self.verify_example_thresholds_weights(dict_df)

    def verify_example_thresholds_weights(self, thresholds_weights_dict):
        self.assertTrue('g1' in thresholds_weights_dict)
        criteria_count = len(thresholds_weights_dict.keys())
        self.assertTrue(criteria_count == 4, 'Actual length: {}'.format(criteria_count))
        self.assertTrue('Q' in thresholds_weights_dict['g1'], 'Thresholds and weights found: {}'.format(thresholds_weights_dict['g1'].keys()))
        self.assertTrue(len(thresholds_weights_dict['g1'].keys()) == 4)

    def setup_class(self):
        threshold_weights_df = read_csv('test/example_thresholds_weights.csv')
        alternatives_df = read_csv('test/example_alternatives.csv')
        electre_handler = electre(alternatives_df, threshold_weights_df)
        return electre_handler

    def test_electre_constructor(self):
        electre_handler = self.setup_class()
        self.assertEqual(type(electre_handler), electre, 'Actual type: {}'.format(type(electre_handler)))
        self.assertTrue(hasattr(electre_handler, 'alternatives'))
        self.assertTrue(hasattr(electre_handler, 'weights'))
        self.assertTrue(hasattr(electre_handler, 'thresholds'))

    def test_separate_thresholds_weights(self):
        electre_handler = self.setup_class()
        threshold_weights_dict = {'g1' :
                                      {'P': 1,
                                       'Q':3,
                                       'Weights':2},
                                  'g2' :
                                      {'P': 4,
                                       'Q': 5,
                                       'Weights' : 6}}
        threshold_weights_df = pd.DataFrame(threshold_weights_dict)
        thresholds_dict, weights_dict = electre_handler._separate_thresholds_weights(threshold_weights_df)
        thresholds_test_dict = {'g1' :
                                      {'P': 1,
                                       'Q':3},
                                  'g2' :
                                      {'P': 4,
                                       'Q': 5}}
        self.assertDictEqual(thresholds_dict, thresholds_test_dict)
        weights_test_dict = {'g1' :2,
                             'g2' : 6}
        self.assertDictEqual(weights_dict, weights_test_dict)

    def test_calculate_partial_concordance_discordance(self):
        electre_handler = self.setup_class()
        concordance_matrices, discordance_matrices = electre_handler.calculate_partial_concordance_discordance()
        g1_concordance = {'a1': {'a1': 1, 'a4': 0, 'a6': 0, 'a2': 1, 'a3': 0, 'a5': 1},
                          'a4': {'a1': 1, 'a4': 1, 'a6': 0, 'a2': 1, 'a3': 1, 'a5': 1},
                          'a6': {'a1': 1, 'a4': 1, 'a6': 1, 'a2': 1, 'a3': 1, 'a5': 1},
                          'a2': {'a1': 1, 'a4': 0, 'a6': 0, 'a2': 1, 'a3': 0, 'a5': 1},
                          'a3': {'a1': 1.0, 'a4': 1.0, 'a6': 0.75000000000000178, 'a2': 1.0, 'a3': 1.0, 'a5': 1.0},
                          'a5': {'a1': 1.0, 'a4': 0.0, 'a6': 0.0, 'a2': 0.20000000000000462, 'a3': 0.0, 'a5': 1.0}}
        self.assertDictEqual(g1_concordance, concordance_matrices['g1'].to_dict())
        self.assertEqual(len(concordance_matrices.keys()), 4)
        g1_discordance ={'a2': {'a2': 0.0, 'a5': 0.0, 'a1': 0.0, 'a3': 1.0,
                                'a4': 0.50000000000000722, 'a6': 1.0},
                         'a5': {'a2': 0, 'a5': 0, 'a1': 0, 'a3': 1, 'a4': 1, 'a6': 1},
                         'a1': {'a2': 0, 'a5': 0, 'a1': 0, 'a3': 1, 'a4': 1, 'a6': 1},
                         'a3': {'a2': 0, 'a5': 0, 'a1': 0, 'a3': 0, 'a4': 0, 'a6': 0},
                         'a4': {'a2': 0.0, 'a5': 0.0, 'a1': 0.0, 'a3': 0.0,
                                'a4': 0.0, 'a6': 0.2999999999999981},
                         'a6': {'a2': 0, 'a5': 0, 'a1': 0, 'a3':0, 'a4': 0, 'a6': 0}}

        self.assertDictEqual(g1_discordance, discordance_matrices['g1'].to_dict(),
                             'Discordance: {}'.format(discordance_matrices['g1'].to_dict()))
        self.assertEqual(len(discordance_matrices.keys()), 4)


    def test_calculate_diff_matrix(self):
        electre_handler = self.setup_class()
        diff_matrix_df = electre_handler._calculate_diff_matrix('g1')
        diff_matrix_test = {'a1': {'a1': 0.0,
                                  'a2': 0.26999999999999957,
                                  'a3': 1.0800000000000001,
                                  'a4': 0.87000000000000011,
                                  'a5': -0.1899999999999995,
                                  'a6': 1.4299999999999997},
                                 'a2': {'a1': -0.26999999999999957,
                                  'a2': 0.0,
                                  'a3': 0.8100000000000005,
                                  'a4': 0.60000000000000053,
                                  'a5': -0.45999999999999908,
                                  'a6': 1.1600000000000001},
                                 'a3': {'a1': -1.0800000000000001,
                                  'a2': -0.8100000000000005,
                                  'a3': 0.0,
                                  'a4': -0.20999999999999996,
                                  'a5': -1.2699999999999996,
                                  'a6': 0.34999999999999964},
                                 'a4': {'a1': -0.87000000000000011,
                                  'a2': -0.60000000000000053,
                                  'a3': 0.20999999999999996,
                                  'a4': 0.0,
                                  'a5': -1.0599999999999996,
                                  'a6': 0.55999999999999961},
                                 'a5': {'a1': 0.1899999999999995,
                                  'a2': 0.45999999999999908,
                                  'a3': 1.2699999999999996,
                                  'a4': 1.0599999999999996,
                                  'a5': 0.0,
                                  'a6': 1.6199999999999992},
                                 'a6': {'a1': -1.4299999999999997,
                                  'a2': -1.1600000000000001,
                                  'a3': -0.34999999999999964,
                                  'a4': -0.55999999999999961,
                                  'a5': -1.6199999999999992,
                                  'a6': 0.0}}
        self.assertDictEqual(diff_matrix_test, diff_matrix_df.to_dict())

    def assert_close_numbers(self, reference, value, percentage=0.01):
        self.assertTrue(abs(reference-value) < percentage * abs(reference))

    def test_concordance_piecewise(self):
        electre_handler = self.setup_class()
        test_0_equal = electre_handler._concordance_piecewise(.4,.4,.2,15,16)
        self.assertEqual(test_0_equal, 0)
        test_0_greater = electre_handler._concordance_piecewise(.5,.4,.2,15,16)
        self.assertEqual(test_0_greater, 0)
        test_1 = electre_handler._concordance_piecewise(.1,.4,.2,15,16)
        self.assertEqual(test_1, 1)
        test_in_between = electre_handler._concordance_piecewise(.3,.4,.2,15,16)
        # make sure answer is really close, less than 1% error
        self.assert_close_numbers(-3, test_in_between)

    def test_discordance_piecewise(self):
        electre_handler = self.setup_class()
        test_0 = electre_handler._discordance_piecewise(.2,.3,.6,15,16)
        self.assertEqual(test_0, 0)
        test_1_equal = electre_handler._discordance_piecewise(.6,.3,.6,15,16)
        self.assertEqual(test_1_equal, 1)
        test_1_greater = electre_handler._discordance_piecewise(.7,.3,.6,15,16)
        self.assertEqual(test_1_greater, 1)
        test_in_between = electre_handler._discordance_piecewise(.4,.3,.6,15,16)
        self.assert_close_numbers(2.33333333, test_in_between)

    def test_normalize_weights(self):
        electre_handler = self.setup_class()
        self.assert_close_numbers(0.283911672, electre_handler._normalize_weights()['g1'])

    def test_calculate_global_concordance(self):
        electre_handler = self.setup_class()
        global_concordance = electre_handler.calculate_global_concordance()
        global_concordance_test = {'a1': {'a1': 1.0, 'a2': 0.81135646687697149, 'a6': 0.45615141955835964,
                                          'a5': 1.0, 'a4': 0.52744479495268137, 'a3': 0.26750788643533124},
                                   'a2': {'a1': 1.0, 'a2': 1.0, 'a6': 0.45615141955835964,
                                          'a5': 1.0, 'a4': 0.71608832807570977, 'a3': 0.45615141955835964},
                                   'a6': {'a1': 0.96227129337539452, 'a2': 0.78460567823343808,
                                          'a6': 1.0, 'a5': 1.0, 'a4': 0.81135646687697149, 'a3': 0.81135646687697149},
                                   'a5': {'a1': 0.54384858044164031, 'a2': 0.10876971608832826, 'a6': 0.0,
                                          'a5': 1.0, 'a4': 0.52744479495268137, 'a3': 0.26750788643533124},
                                   'a4': {'a1': 0.69350157728706607, 'a2': 0.32164037854889588,
                                          'a6': 0.1886435331230284, 'a5': 1.0, 'a4':1.0, 'a3': 0.55141955835962142},
                                   'a3': {'a1': 0.7324921135646687, 'a2': 0.7324921135646687, 'a6': 0.6615141955835967,
                                          'a5': 0.7324921135646687, 'a4': 0.7324921135646687, 'a3': 1.0}}

        #self.assertDictEqual(global_concordance.to_dict(), global_concordance_test,
        #                     'Global Concordance: {}'.format(global_concordance.to_dict()))
        assert_frame_equal(global_concordance, pd.DataFrame(global_concordance_test))

    def test_calculate_credibility(self):
        electre_handler = self.setup_class()
        credibility = electre_handler.calculate_credibility()
        credibility_test = {'a1': {'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 1, 'a6': 0},
                             'a2': {'a1': 1.0,
                              'a2': 0.0,
                              'a3': 0.0,
                              'a4': 0.716088328,
                              'a5': 1.0,
                              'a6': 0.0},
                             'a3': {'a1': 0.0,
                              'a2': 0.0,
                              'a3': 0.0,
                              'a4': 0.41073113200000005,
                              'a5': 0.54764150899999997,
                              'a6': 0.0},
                             'a4': {'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 1, 'a6': 0},
                             'a5': {'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 0, 'a6': 0},
                             'a6': {'a1': 0.96227129299999992,
                              'a2': 0.0,
                              'a3': 0.0,
                              'a4': 0.0,
                              'a5': 1.0,
                              'a6': 0.0}}

        #self.assertDictEqual(credibility.to_dict(), credibility_test,
        #                     'Credibility Matrix: {}'.format(credibility.to_dict()))
        assert_frame_equal(credibility, pd.DataFrame(credibility_test).astype(np.float64))

    def test_compute_lambda(self):
        electre_handler = self.setup_class()
        matrix = pd.DataFrame({'a1':{'a1':1, 'a2':2},
                               'a2': {'a1': 1, 'a2': 2}})
        test = electre_handler._compute_lambda(matrix, 10, 0)
        self.assertEqual(test, 2)
        test = electre_handler._compute_lambda(matrix, 5, 3)
        self.assertEqual(test, 1)

    def test_distill(self):
        electre_handler = self.setup_class()
        credibility_test = {'a1': {'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 1, 'a6': 0},
                             'a2': {'a1': 1.0,
                              'a2': 0.0,
                              'a3': 0.0,
                              'a4': 0.71608830000000012,
                              'a5': 1.0,
                              'a6': 0.0},
                             'a3': {'a1': 0.0,
                              'a2': 0.0,
                              'a3': 0.0,
                              'a4': 0.41073109999999996,
                              'a5': 0.5476415,
                              'a6': 0.0},
                             'a4': {'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 1, 'a6': 0},
                             'a5': {'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 0, 'a6': 0},
                             'a6': {'a1': 0.96227129999999994,
                              'a2': 0.0,
                              'a3': 0.0,
                              'a4': 0.0,
                              'a5': 1.0,
                              'a6': 0.0}}
        alternative_count, alternative = electre_handler._distill(pd.DataFrame(credibility_test))
        self.assertEqual(alternative, 'a5')
        self.assertTrue(alternative_count==1)
        # Returning two alternatives effectively tests second_pass
        alternative_count, alternative = electre_handler._distill(pd.DataFrame(credibility_test), ascending=True)
        self.assertTrue('a2' in alternative)
        self.assertTrue('a6' in alternative)
        self.assertTrue(alternative_count == 2)

    def test_distillation(self):
        electre_handler = self.setup_class()
        chosen_order = electre_handler.distillation()
        self.assertTrue(chosen_order[0] == 'a5')
        self.assertTrue(chosen_order[1] == 'a1')
        self.assertTrue(chosen_order[2] == 'a4')
        # Tuple order below made simply comparing the list not work
        self.assertTrue('a2' in chosen_order[3])
        self.assertTrue('a3' in chosen_order[3])
        self.assertTrue('a6' in chosen_order[3])
        self.assertTrue(len(chosen_order) == 4)

        chosen_order = electre_handler.distillation(ascending=True)
        self.assertTrue(chosen_order[0] == 'a5')
        self.assertTrue(chosen_order[1] == 'a3')
        self.assertTrue('a1' in chosen_order[2])
        self.assertTrue('a4' in chosen_order[2])
        self.assertTrue('a2' in chosen_order[3])
        self.assertTrue('a6' in chosen_order[3])

    def test_get_occurences(self):
        electre_handler = self.setup_class()
        test = {1:1, 2:2}
        occurences = electre_handler._get_occurences(test, 1)
        self.assertListEqual(occurences, [1])
        test = {1: 1, 2: 1}
        occurences = electre_handler._get_occurences(test, 1)
        self.assertListEqual(occurences, [1, 2])

    def test_compute_ranks(self):
        electre_handler = self.setup_class()
        order = ['a5', 'a1', ('a2', 'a3')]
        rank_dict = electre_handler._compute_ranks(order)
        self.assertDictEqual(rank_dict,
                             {'a5':0, 'a1':1, 'a2':2, 'a3':2})

    def test_compute_final_ranking_matrix(self):
        electre_handler = self.setup_class()
        final_matrix_test = {'a1': {'a1': '-', 'a2': 'P-', 'a3': 'R', 'a4': 'P-', 'a5': 'P+', 'a6': 'P-'},
                     'a2': {'a1': 'P+', 'a2': '-', 'a3': 'P+', 'a4': 'P+', 'a5': 'P+', 'a6': 'I'},
                     'a3': {'a1': 'R', 'a2': 'P-', 'a3': '-', 'a4': 'R', 'a5': 'P+', 'a6': 'P-'},
                     'a4': {'a1': 'P+', 'a2': 'P-', 'a3': 'R', 'a4': '-', 'a5': 'P+', 'a6': 'P-'},
                     'a5': {'a1': 'P-', 'a2': 'P-', 'a3': 'P-', 'a4': 'P-', 'a5': '-', 'a6': 'P-'},
                     'a6': {'a1': 'P+', 'a2': 'I', 'a3': 'P+', 'a4': 'P+', 'a5': 'P+', 'a6': '-'}}
        final_matrix = electre_handler.compute_final_ranking_matrix()
        assert_frame_equal(final_matrix, pd.DataFrame(final_matrix_test))

    def test_alternative_node(self):
        electre_handler = self.setup_class()
        test_id = 'a5'
        node = electre_handler.alternative_node(test_id)
        self.assertEqual(node.id, test_id)
        self.assertTrue(hasattr(node, 'after'))
        self.assertTrue(hasattr(node, 'equals'))
        self.assertEqual(node.label, test_id)
        self.assertFalse(node.hidden)

        self.assertEqual('{}'.format(node), 'a5 -> []')
        self.assertEqual(str(node), 'a5 -> []')

        node.hide()
        self.assertTrue(node.hidden)

        self.assertEqual(len(node.after), 0)
        node_b = electre_handler.alternative_node('a6')
        node.add_after(node_b)
        self.assertEqual(len(node.after), 1)
        self.assertEqual(type(node.after[0]), electre_handler.alternative_node)

        self.assertEqual(len(node.equals), 0)
        node.add_equals(node_b)
        self.assertEqual(len(node.equals), 1)
        self.assertEqual(type(node.equals[0]), electre_handler.alternative_node)

        node.append_name('b')
        self.assertEqual(node.label, 'a5, b')
        self.assertEqual(node.id, 'a5')

    def test_remove_duplicate_mappings(self):
        electre_handler = self.setup_class()
        test_id = 'a5'
        node = electre_handler.alternative_node(test_id)
        node_b = electre_handler.alternative_node('a6')
        node_c = electre_handler.alternative_node('a7')

        node_b.add_after(node_c)
        node.add_after(node_b)
        node.add_after(node_c)

        self.assertEqual(len(node.after), 2)

        electre_handler._remove_duplicate_mappings({'a5': node})

        self.assertEqual(len(node.after), 1)
        self.assertEqual(node.after[0], node_b)
        self.assertEqual(node_b.after[0], node_c)

    def test_compute_final_order(self):
        electre_handler = self.setup_class()
        nodes = electre_handler.compute_final_order()

        self.node_checker(nodes['a5'], 2, 'a5', 'a5', 0)
        self.node_checker(nodes['a1'], 1, 'a1', 'a1', 0, single_after_check='a4')
        self.node_checker(nodes['a4'], 2, 'a4', 'a4', 0)
        self.node_checker(nodes['a3'], 2, 'a3', 'a3', 0)
        self.node_checker(nodes['a2'], 0, 'a2', 'a2, a6', 1, single_equals_check='a6')
        self.node_checker(nodes['a6'], 0, 'a6', 'a6, a2', 1, single_equals_check='a2')

        self.assertFalse(nodes['a5'].hidden or nodes['a3'].hidden or nodes['a1'].hidden or nodes['a4'].hidden)

        self.assertTrue(nodes['a2'].hidden or nodes['a6'].hidden)

    def node_checker(self, node, after_length, id, label, equals_length, single_after_check=None,
                     single_equals_check=None):
        self.assertEqual(len(node.after), after_length, '{}'.format(node.after))
        self.assertEqual(node.id, id)
        self.assertEqual(node.label, label)
        self.assertEqual(len(node.equals), equals_length)
        if single_after_check != None:
            self.assertEqual(node.after[0].id, single_after_check)
        if single_equals_check != None:
            self.assertEqual(node.equals[0].id, single_equals_check)

    def test_write_results(self):
        result_path = 'order.html'
        if os.path.exists(result_path):
            os.unlink(result_path)
        self.assertFalse(os.path.exists(result_path))
        electre_handler = self.setup_class()
        electre_handler.write_results()
        self.assertTrue(os.path.exists(result_path))