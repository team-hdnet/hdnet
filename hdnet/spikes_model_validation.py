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

class LogProbabilityRatio(Validation):
    """
    Compares log-likelihoods of occurrence of codewords 
    between experimental data and predicted data
    """
    def call(self):

        original_code_freq = self.spikes_true.get_frequencies(self.spikes_true)
        predicted_code_freq = self.spikes_pred.get_frequencies(self.spikes_pred)

        #since total timebins same for both spike trains, 
        # log(P(model)/P(data)) = log(freq(model)/freq(data))
        #calculating only for the common codes between predicted and experimental

        common_codes = list( set(original_code_freq.keys()) & set(predicted_code_freq.keys()) )
        log_ratios = {}
        for code in common_codes:
            log_ratios[code] = np.log(predicted_code_freq[code]/original_code_freq[code])

        return log_ratios

class MostFrequentCommonCode(Validation):
    """
    Compares top 'N' codewords from experimental data and 
    predicted data
    """
    def call(self,N=10):
        """
        Args:
            N: Number of most frequent codewords to check overlap with

        Returns:
            common_codes: List of common codewords from Top 'N' occurring of each spiketrain
        """
        original_code_freq = self.spikes_true.get_frequencies(self.spikes_true)
        predicted_code_freq = self.spikes_pred.get_frequencies(self.spikes_pred)

        common_codes = list( set(np.array(original_code_freq.most_common(N))[:,0]) & 
        set(np.array(predicted_code_freq.most_common(N))[:,0]) )
        count_common = len(common_codes)

        return common_codes