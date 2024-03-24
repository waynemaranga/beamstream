from utilities.beamanalysis import BeamAnalysis
from utilities.plots import Plots


class Beam:
    def __init__(self, spans, udls, EI_values=[1, 1]):
        self.span_12, self.span_23 = spans
        self.udl_12, self.udl_23 = udls
        self.EI_12, self.EI_23 = EI_values

    def analysis(self):
        self.beam_analysis = BeamAnalysis(
            [self.span_12, self.span_12],
            [self.udl_12, self.udl_23],
            [self.EI_12, self.EI_23],
        )  # when variable had same name as method, unkown error from Pylance. Check #CHECK
        return self.beam_analysis

    def results(self):
        self.analysis()
        self.analysis_results = self.beam_analysis.get_results()
        return self.analysis_results

    def print_results(self):
        self.beam_analysis.print_results()

    def sfd(self):
        self.results()
