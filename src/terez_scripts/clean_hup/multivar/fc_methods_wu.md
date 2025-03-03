The coherence function computes the magnitude‐squared coherence between every pair of channels over specified time windows. It uses the scipy.signal.coherence function on segments of the data (using parameters for segment length and overlap) and then averages the resulting coherence values over each frequency range defined by the freqs array.

