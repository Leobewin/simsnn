from snn import SpikingNueralNetwork
import numpy as np
import unittest
 
class TestSNN(unittest.TestCase):
    """
    Test the varoius components of SNN class 
    """
    def test_input_next_iteration_reset(self):
        """
        Test that all the  values of x are being reset if y is not above
        threshold
        """
        network = SpikingNueralNetwork(input=3,output=3,threshold=15)
        network.feed_forward()
        network.after_forward()
        self.assertEqual(np.any(network.x),False)
 
    def test_output_reset_after_threshold(self):
        """
        Test that values of y are being reset if y is above
        threshold
        """
        network = SpikingNueralNetwork(input=3,output=3,threshold=5)
        network.y = np.array([4.0,5.0,1]).reshape(3,1)
        network.feed_forward()
        network.after_forward()
        self.assertEqual(network.y[1],0)

if __name__ == '__main__':
    unittest.main()
