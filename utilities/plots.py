"""Plotting: Shear Force and Bending Moment Diagrams"""

# pylint: disable = C0103
# pylint: disable = C0301
# pylint: disable = W0511
# pylint: disable = W0201


from math import ceil
from typing import Union
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# from operator import neg as neg
# from math import ceil, floor

Number = Union[float, int]


class Plots:
    """Plot object..."""

    def __init__(
        self, spans: list[Number], udls: list[Number], reactions: list[Number]
    ) -> None:
        """FIXME: refactor all inits to separate turn types for use as functions. This will reduce class attrs exponentially as well"""
        self.span_12, self.span_23 = spans[0], spans[1]
        self.w_12, self.w_23 = udls[0], udls[1]
        self.R_1, self.R_2, self.R_3 = reactions[0], reactions[1], reactions[2]

        # -> Shear Force values #FIXME: should be separate fn, for calls elsewhere
        self.V_1 = self.R_1  # shear at support 1
        self.V_2: Number = (
            self.R_1 - self.w_12 * self.span_12
        )  # shear at support 2, j-LHS
        self.V_3: Number = (
            self.R_1 - self.w_12 * self.span_12 + self.R_2
        )  # shear at support 2 j-RHS
        self.V_4: Number = (
            self.R_1 - self.w_12 * self.span_12 + self.R_2 - self.w_23 * self.span_23
        )  # shear at support 3

        self.x_c1: float = (
            self.V_1 / self.w_12
        )  # location of change in shear on span 1-2
        self.x_c2: float = self.span_12 - self.x_c1
        self.x_c3: float = (
            self.V_3 / self.w_23
        )  # KEY: location of change in shear on span 2-3 from support
        self.x_c4: float = self.span_23 - self.x_c3

        self.A_c1: float = (
            self.V_1 * self.x_c1 / 2
        )  # area of shear diagram on span 1-2, LHS
        self.A_c2: float = (
            self.V_2 * self.x_c2 / 2
        )  # area of shear diagram on span 1-2, RHS
        self.A_c3: float = (
            self.V_3 * self.x_c3 / 2
        )  # area of shear diagram on span 2-3, LHS
        self.A_c4: float = (
            self.V_4 * self.x_c4 / 2
        )  # area of shear diagram on span 2-3, RHS

        # -> Bending Moment values
        self.M_1: float = self.A_c1  # max moment on span 1-2
        self.M_2: float = (
            self.M_1 + self.A_c2
        )  # min moment i.e max hogging on about support 2
        self.M_3: float = self.M_2 + self.A_c3  # max moment on span 2-3

    def plot_strings(self) -> None:
        """Labels and axis stuff, as str"""
        self.x_c1s: str = f"{str(object=round(number=self.x_c1, ndigits=2))} m"
        self.x_c2s: str = f"{str(object=round(number=self.x_c2, ndigits=2))} m"
        self.x_c3s: str = (
            f"{str(object=round(number=(self.span_12+self.x_c3), ndigits=2))} m"  # added L_12 since on RHS
        )
        self.x_c4s: str = (
            f"{str(object=round(number=(self.span_12+self.x_c4), ndigits=2))} m"
        )

        self.V_1s: str = f"{str(object=round(number=self.V_1, ndigits=2))} kN"
        self.V_2s: str = f"{str(object=round(number=self.V_2, ndigits=2))} kN"
        self.V_3s: str = f"{str(object=round(number=self.V_3, ndigits=2))} kN"
        self.V_4s: str = f"{str(object=round(number=self.V_4, ndigits=2))} kN"

        self.M_1s: str = f"{str(object=round(number=self.M_1, ndigits=2))} kNm"
        self.M_2s: str = f"{str(object=round(number=self.M_2, ndigits=2))} kNm"
        self.M_3s: str = f"{str(object=round(number=self.M_3, ndigits=2))} kNm"

    def sfd(self) -> None:
        """Creates but does not plot a shear force diagram. That's handled by the
        `plotly_sfd` and `matplotlib_sfd` instance methods.
        """
        self.x_LHSv: ndarray = np.linspace(start=0, stop=self.span_12, num=1000)
        self.x_RHSv: ndarray = np.linspace(
            start=self.span_12, stop=(self.span_12 + self.span_23), num=1000
        )

        self.y_LHSv: list[Number] = [self.V_1 - self.w_12 * x for x in self.x_LHSv]
        self.y_RHSv: list[Number] = [
            self.V_3 - self.w_23 * (x - self.span_12) for x in self.x_RHSv
        ]

    def plotly_sfd(self) -> None:
        """Shear Force Diagram, with Plotly's graph objects"""
        self.sfd()
        self.plot_strings()

        fig = go.Figure()
        fig.add_trace(
            trace=go.Scatter(
                x=self.x_LHSv,
                y=self.y_LHSv,
                name="LHS",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            trace=go.Scatter(
                x=self.x_RHSv,
                y=self.y_RHSv,
                name="RHS",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            )
        )
        fig.update_layout(title="SFD", xaxis_title="x (m)", yaxis_title="V (kN)")
        fig.show()

    def matplotlib_sfd(self) -> None:
        """SFD with Matplotib's pyplot"""
        self.sfd()
        self.plot_strings()

        # -
        plt.figure(figsize=(10, 6))

        # -> Plot limits
        plt.xlim(0, self.span_12 + self.span_23)  # x-axis limits = beam length

        # -
        plt.fill_between(x=self.x_LHSv, y1=self.y_LHSv, color="skyblue", alpha=0.4)
        plt.fill_between(x=self.x_RHSv, y1=self.y_RHSv, color="skyblue", alpha=0.4)
        plt.plot(self.x_LHSv, self.y_LHSv, color="blue", label="SFD")
        plt.plot(self.x_RHSv, self.y_RHSv, color="blue", label="SFD")
        plt.axhline(y=0, color="black", linewidth=2)

        # -
        # -> Axes ticks
        plt.minorticks_on()
        plt.xticks(
            ticks=np.arange(
                start=0,
                stop=(self.span_12 + self.span_23),
                step=ceil((self.span_12 + self.span_23) / 5),
            )
        )  # make x-axis ticks at reasonable intervals based on beam length
        plt.tick_params(axis="x", which="minor", bottom=True)

        # -> Grid
        plt.grid(which="major", linestyle="-", linewidth="0.5", color="black")
        plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")

        # -> Labels on important points
        plt.annotate(
            text=self.V_1s,
            xy=(0, self.V_1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            text=self.V_2s,
            xy=(self.span_12, self.V_2),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            text=self.V_3s,
            xy=(self.span_12, self.V_3),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            text=self.V_4s,
            xy=((self.span_12 + self.span_23), self.V_4),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        plt.annotate(
            text=self.x_c1s,
            xy=(self.x_c1, 0),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            text=self.x_c3s,
            xy=((self.span_12 + self.x_c3), 0),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        plt.title(label="Shear Force Diagram")
        plt.show()

    def bmd(self) -> None:
        """BMD values"""
        # --- interpolation points for bmd curves
        self.x_LHS: list[Number] = [0, self.x_c1, self.span_12]
        self.x_RHS: list[Number] = [
            self.span_12,
            self.span_12 + self.x_c3,
            self.span_12 + self.span_23,
        ]

        self.y_LHS: list[Number] = [0, self.M_1, self.M_2]
        self.y_RHS: list[Number] = [self.M_2, self.M_3, 0]

        self.coefficients_LHS: ndarray = np.polyfit(
            x=self.x_LHS, y=self.y_LHS, deg=2
        )  # coeffs for TODO: complete -- typing error/lint
        self.coefficients_RHS: ndarray = np.polyfit(x=self.x_RHS, y=self.y_RHS, deg=2)

        self.polynomial_LHS = np.poly1d(c_or_r=self.coefficients_LHS)
        self.polynomial_RHS = np.poly1d(c_or_r=self.coefficients_RHS)

        self.x_range_LHS: ndarray = np.linspace(start=0, stop=self.span_12, num=1000)
        self.x_range_RHS: ndarray = np.linspace(
            start=self.span_12, stop=self.span_12 + self.span_23, num=1000
        )
        self.y_range_LHS: ndarray = self.polynomial_LHS(self.x_range_LHS)
        self.y_range_RHS: ndarray = self.polynomial_RHS(self.x_range_RHS)

    def plotly_bmd(self) -> None:
        """Bending Moment Diagram, with Plotly's Graph Objects"""
        self.plot_strings()
        self.bmd()

        # create figure
        fig = go.Figure()

        # -> add traces
        fig.add_trace(
            trace=go.Scatter(
                x=self.x_range_LHS,
                y=self.y_range_LHS,
                fill="tozeroy",
                name="LHS",
                fillcolor="rgba(135, 206, 235, 0.5)",
                line=dict(color="skyblue"),
            )
        )  # LHS
        fig.add_trace(
            trace=go.Scatter(
                x=self.x_range_RHS,
                y=self.y_range_RHS,
                fill="tozeroy",
                name="RHS",
                fillcolor="rgba(135, 206, 235, 0.5)",
                line=dict(color="skyblue"),
            )
        )  # RHS

        # -> annotating important points
        fig.add_annotation(
            x=self.x_c1,
            y=self.M_1,
            text=self.M_1s,
            showarrow=False,
            arrowhead=1,
            ax=0,
            ay=-40,
        )  # M_1
        fig.add_annotation(
            x=self.x_c1,
            y=0,
            text=self.x_c1s,
            showarrow=False,
            arrowhead=1,
            ax=0,
            ay=-40,
        )  # x_c1
        fig.add_annotation(
            x=self.span_12,
            y=self.M_2,
            text=self.M_2s,
            showarrow=False,
            arrowhead=1,
            ax=0,
            ay=-40,
        )  # M_2
        fig.add_annotation(
            x=self.span_12 + self.x_c3,
            y=self.M_3,
            text=self.M_3s,
            showarrow=False,
            arrowhead=1,
            ax=0,
            ay=-40,
        )  # M_3
        fig.add_annotation(
            x=self.span_12 + self.x_c3,
            y=0,
            text=self.x_c3s,
            showarrow=False,
            arrowhead=1,
            ax=0,
            ay=-40,
        )  # x_c4

        # TODO: annotate points of contraflexure

        # -> add title and axis labels
        fig.update_layout(
            title="Bending Moment Diagram", xaxis_title="x (m)", yaxis_title="M (kNm)"
        )
        fig.update_yaxes(autorange="reversed")  # flip y-axis

        fig.show()

    def matplotlib_bmd(self) -> None:
        """plot bmd with Matplotlib"""
        self.plot_strings()
        self.bmd()

        # ->
        plt.figure(
            figsize=(10, 6)
        )  # TODO: check if affects appearance of max and min values

        # -> Plot limits
        plt.xlim(0, self.span_12 + self.span_23)  # x-axis limits = beam length
        # plt.ylim(min(y_points_LHS) - 20, max(y_points_RHS) + 20) # shear values +/- 20 on either side of y-axis

        # -> Plot line and fill
        plt.plot(
            self.x_range_LHS, self.y_range_LHS, label="LHS", color="skyblue"
        )  # plot the curve
        plt.plot(self.x_range_RHS, self.y_range_RHS, label="RHS", color="skyblue")
        plt.fill_between(
            x=self.x_range_LHS, y1=self.y_range_LHS, y2=0, alpha=0.2, color="skyblue"
        )  # fill the area under the curve
        plt.fill_between(
            x=self.x_range_RHS, y1=self.y_range_RHS, y2=0, alpha=0.2, color="skyblue"
        )
        plt.axhline(y=0, color="black", linewidth=2)

        # -> Axes ticks
        plt.minorticks_on()
        plt.xticks(
            ticks=np.arange(
                start=0,
                stop=(self.span_12 + self.span_23),
                step=ceil((self.span_12 + self.span_23) / 5),
            )
        )  # make x-axis ticks at reasonable intervals based on beam length
        plt.tick_params(axis="x", which="minor", bottom=True)

        # -> Grid
        plt.grid(which="major", linestyle=":", linewidth="0.5", color="grey")
        plt.grid(which="minor", linestyle=":", linewidth="0.5", color="grey")

        # -> Labels on important points
        # for M_1 i.e max moment on span 1-2 and x_c1 i.e location of change in shear on span 1-2
        # plt.scatter(x_c1, M_1, color='red')
        plt.annotate(
            text=self.x_c1s,
            xy=(self.x_c1, 0),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            text=self.M_1s,
            xy=(self.x_c1, self.M_1),
            textcoords="offset points",
            xytext=(-10, 0),
            va="center",
        )

        # for M_2 i.e min moment i.e max hogging on about support 2 and x_c2 i.e location of change in shear on span 1-2
        # plt.scatter(l_12, M_2, color='red')
        plt.annotate(
            text=self.M_2s,
            xy=(self.span_12, self.M_2),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
        )

        # for M_3 i.e max moment on span 2-3 and x_c3 i.e location of change in shear on span 2-3
        # plt.scatter(l_12+x_c3, M_3, color='red')
        plt.annotate(
            text=self.x_c3s,
            xy=(self.span_12 + self.x_c3, 0),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            text=self.M_3s,
            xy=(self.span_12 + self.x_c3, self.M_3),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
        )

        plt.title(label="Bending Moment Diagram")
        plt.gca().invert_yaxis()  # Flip the y-axis
        plt.show()
