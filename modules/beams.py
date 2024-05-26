"""# BEAMS: Two Span Continuous"""

# pylint: disable = C0103
# pylint: disable = W0201
# pylint: disable = W0511

from typing import Any, Union
from utilities.beamanalysis import BeamAnalysis

Number = Union[float, int]

# from utilities.plots import Plots


class Beam:
    """Beams"""

    def __init__(
        self, spans: list[Number], udls: list[Number], EI_values=[1, 1]
    ) -> None:
        #! FIXME:  Dangerous default value [] as argumentPylintW0102:dangerous-default-value
        self.span_12, self.span_23 = spans
        self.udl_12, self.udl_23 = udls
        self.EI_12, self.EI_23 = EI_values

    def analysis(self) -> BeamAnalysis:
        """Beam Analysis by SSM & MDM"""
        self.beam_analysis = BeamAnalysis(
            spans=[self.span_12, self.span_12],
            udls=[self.udl_12, self.udl_23],
            EI_values=[self.EI_12, self.EI_23],
        )  # when variable had same name as method, unkown error from Pylance. Check #CHECK
        return self.beam_analysis

    def results(self) -> list[list[Any]]:
        """Get Results"""
        self.analysis()
        self.analysis_results: list[list[Any]] = self.beam_analysis.get_results()  # type: ignore
        return self.analysis_results

    def print_results(self) -> None:
        """Print results to std. output"""
        self.beam_analysis.print_results()  # type: ignore

    def sfd(self) -> None:
        """NotImplemented"""
        # FIXME:
        self.results()
