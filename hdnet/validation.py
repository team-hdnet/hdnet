import numpy as np

class Validation:
    """
    Validation base class.
    
    """

    def __init__(self,spikes_true,spikes_pred):
        """
        Initializes `Validation` class.

        Args:
            spikes_true: Experimental Spike Train data
            spikes_pred: Predicted Spike Train data
        """

        self.spikes_true = spikes_true
        self.spikes_pred = spikes_pred

    def call(self):
        """
        Invokes the `Validation` instance.
        

        Returns:
            Matrix corresponding to Validation
            Shape varies according to the Validation method used 
        """
        raise NotImplementedError('Must be implemented in subclasses.')