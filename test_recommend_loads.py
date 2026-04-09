import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.recommend_loads import RecommendLoadsService


class TestRecommendLoadsService(unittest.TestCase):
    
    def setUp(self):
        self.mock_scoring_strategy = Mock()
        self.mock_constraints = Mock()
        self.service = RecommendLoadsService.__new__(RecommendLoadsService)
        self.service.scoring_strategy = self.mock_scoring_strategy
        self.service.feasibility_checker = Mock()
        self.service.constraints = self.mock_constraints
        self.service.evaluate_loads_service = Mock()
        
    def test_init(self):
        """Test RecommendLoadsService initialization"""
        self.assertEqual(self.service.scoring_strategy, self.mock_scoring_strategy)
        self.assertIsNotNone(self.service.feasibility_checker)
        self.assertIsNotNone(self.service.constraints)
        self.assertIsNotNone(self.service.evaluate_loads_service)
        
    def test_recommend_loads_with_feasible_loads(self):
        """Test recommend_loads returns feasible loads sorted by score"""
        # Setup
        mock_load = Mock()
        mock_load.load_id = 1
        loads = [mock_load]
        truck_state = Mock()

        mock_evaluation = Mock()
        mock_evaluation.load.load_id = 1
        self.service.evaluate_loads_service.evaluate_loads.return_value = [mock_evaluation]
        self.service.feasibility_checker.return_value = (True, None)
        self.mock_scoring_strategy.score_load.return_value = 10
        
        # Execute
        result = self.service.recommend_loads(loads, truck_state)
        
        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], mock_evaluation)
        self.assertEqual(result[0][1], 10)
        self.service.feasibility_checker.assert_called_once_with(mock_evaluation, truck_state, self.mock_constraints)
        self.mock_scoring_strategy.score_load.assert_called_once_with(mock_evaluation)
        
    @patch('builtins.print')
    def test_recommend_loads_with_non_feasible_loads(self, mock_print):
        """Test recommend_loads filters out non-feasible loads"""
        # Setup
        mock_load = Mock()
        mock_load.load_id = 1
        loads = [mock_load]
        truck_state = Mock()

        mock_evaluation = Mock()
        mock_evaluation.load.load_id = 1
        self.service.evaluate_loads_service.evaluate_loads.return_value = [mock_evaluation]
        self.service.feasibility_checker.return_value = (False, "Too heavy")
        
        # Execute
        result = self.service.recommend_loads(loads, truck_state)
        
        # Verify - returns empty list when no feasible loads
        self.assertEqual(result, [])
        mock_print.assert_called_once_with("Load 1 is not feasible after evaluation: Too heavy")
        
    def test_recommend_loads_empty_list(self):
        """Test recommend_loads with empty loads list"""
        # Setup
        loads = []
        truck_state = Mock()
        
        # Execute
        result = self.service.recommend_loads(loads, truck_state)
        
        # Verify - empty list returns empty list
        self.assertEqual(result, [])

    def test_recommend_loads_multiple_loads_sorted_by_score(self):
        """Test that all feasible loads are processed and sorted by score descending"""
        # Setup
        mock_load1 = Mock()
        mock_load1.load_id = 1
        mock_load2 = Mock()
        mock_load2.load_id = 2
        loads = [mock_load1, mock_load2]
        truck_state = Mock()

        mock_eval1 = Mock()
        mock_eval1.load.load_id = 1
        mock_eval2 = Mock()
        mock_eval2.load.load_id = 2

        self.service.evaluate_loads_service.evaluate_loads.side_effect = [
            [mock_eval1], [mock_eval2]
        ]
        self.service.feasibility_checker.return_value = (True, None)
        self.mock_scoring_strategy.score_load.side_effect = [10, 20]
        
        # Execute
        result = self.service.recommend_loads(loads, truck_state)
        
        # Verify - both loads processed, sorted descending by score
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], mock_eval2)
        self.assertEqual(result[0][1], 20)
        self.assertEqual(result[1][0], mock_eval1)
        self.assertEqual(result[1][1], 10)

    @patch('builtins.print')
    def test_recommend_loads_mixed_feasibility(self, mock_print):
        """Test with a mix of feasible and non-feasible loads"""
        # Setup
        mock_load1 = Mock()
        mock_load1.load_id = 1
        mock_load2 = Mock()
        mock_load2.load_id = 2
        loads = [mock_load1, mock_load2]
        truck_state = Mock()

        mock_eval1 = Mock()
        mock_eval1.load.load_id = 1
        mock_eval2 = Mock()
        mock_eval2.load.load_id = 2

        self.service.evaluate_loads_service.evaluate_loads.side_effect = [
            [mock_eval1], [mock_eval2]
        ]
        self.service.feasibility_checker.side_effect = [
            (False, "Too heavy"), (True, "Load is feasible")
        ]
        self.mock_scoring_strategy.score_load.return_value = 15

        # Execute
        result = self.service.recommend_loads(loads, truck_state)

        # Verify - only feasible load returned
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], mock_eval2)
        self.assertEqual(result[0][1], 15)
        mock_print.assert_called_once_with("Load 1 is not feasible after evaluation: Too heavy")


if __name__ == '__main__':
    unittest.main()