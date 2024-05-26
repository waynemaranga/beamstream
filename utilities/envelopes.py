# pylint: disable = C0301


import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from plots import Plots


class ShearForceEnvelope:
    """Shear Force Envelope for all 3 cases"""

    def __init__(self, case_1: Plots, case_2: Plots, case_3: Plots) -> None:
        """
        Initializes the ComparePlots class with three instances of the Plots class.
        """
        self.plot1: Plots = case_1
        self.plot2: Plots = case_2
        self.plot3: Plots = case_3

        self.plot1.sfd()
        self.plot2.sfd()
        self.plot3.sfd()

    def matplotlib(self) -> None:
        """Plot Envelope with Matplotlib's Pyplot"""
        # -> Plot
        plt.figure(figsize=(10, 5))
        plt.grid(which="major", linestyle=":", linewidth="0.5", color="black")
        plt.grid(which="minor", linestyle=":", linewidth="0.5", color="grey")
        plt.minorticks_on()
        plt.plot(self.plot1.x_RHSv, np.zeros(shape=1000), color="black")
        plt.plot(self.plot1.x_LHSv, np.zeros(shape=1000), color="black")

        # -> Case A
        plt.plot(self.plot1.x_LHSv, self.plot1.y_LHSv, label="Case A", color="green")
        plt.plot(self.plot1.x_RHSv, self.plot1.y_RHSv, label="Case A", color="green")
        plt.fill_between(x=self.plot1.x_LHSv, y1=self.plot1.y_LHSv, color="blue", alpha=0.2)  # fmt: skip
        plt.fill_between(x=self.plot1.x_RHSv, y1=self.plot1.y_RHSv, color="blue", alpha=0.2)  # fmt: skip

        # -> Case B
        plt.plot(self.plot2.x_LHSv, self.plot2.y_LHSv, label="Case B", color="blue")
        plt.plot(self.plot2.x_RHSv, self.plot2.y_RHSv, label="Case B", color="blue")
        plt.fill_between(x=self.plot2.x_LHSv, y1=self.plot2.y_LHSv, color="orange", alpha=0.2)  # fmt: skip
        plt.fill_between(x=self.plot2.x_RHSv, y1=self.plot2.y_RHSv, color="orange", alpha=0.2)  # fmt: skip

        # -> Case C
        plt.plot(self.plot3.x_LHSv, self.plot3.y_LHSv, label="Case C", color="red")
        plt.plot(self.plot3.x_RHSv, self.plot3.y_RHSv, label="Case C", color="red")
        plt.fill_between(x=self.plot3.x_LHSv, y1=self.plot3.y_LHSv, color="green", alpha=0.2)  # fmt: skip
        plt.fill_between(x=self.plot3.x_RHSv, y1=self.plot3.y_RHSv, color="green", alpha=0.2)  # fmt: skip

        # Adding labels, title and legend
        plt.xlabel(xlabel="Position along the beam")
        plt.ylabel(ylabel="Shear Force")
        plt.title(label="Comparison of Shear Force Diagrams")
        plt.legend()

        # Display the plot
        plt.show()

    def plotly(self) -> None:
        """Plot envelope with Plotly's Graph Objects"""
        # ->
        plot_layout = go.Layout(width=900, height=600)
        fig = go.Figure(layout=plot_layout)
        fig.add_hline(y=0)

        # -> Case A
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot1.x_LHSv,
                y=self.plot1.y_LHSv,
                fill="tozeroy",
                fillcolor="rgba(0, 200, 0, 0.2)",
                name="Case A",
                line=dict(color="green"),
                hovertemplate="x: %{x:.2f}<br>V: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot1.x_RHSv,
                y=self.plot1.y_RHSv,
                fill="tozeroy",
                fillcolor="rgba(0, 200, 0, 0.2)",
                name="Case A",
                line=dict(color="green"),
                hovertemplate="x: %{x:.2f}<br>V: %{y:.2f}<extra></extra>",
            )
        )

        # -> Case B
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot2.x_LHSv,
                y=self.plot2.y_LHSv,
                fill="tozeroy",
                fillcolor="rgba(0, 0, 200, 0.2)",
                name="Case B",
                line=dict(color="blue"),
                hovertemplate="x: %{x:.2f}<br>V: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot2.x_RHSv,
                y=self.plot2.y_RHSv,
                fill="tozeroy",
                fillcolor="rgba(0, 0, 200, 0.2)",
                name="Case B",
                line=dict(color="blue"),
                hovertemplate="x: %{x:.2f}<br>V: %{y:.2f}<extra></extra>",
            )
        )

        # -> Case C
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot3.x_LHSv,
                y=self.plot3.y_LHSv,
                fill="tozeroy",
                fillcolor="rgba(200, 0, 0, 0.2)",
                name="Case C",
                line=dict(color="red"),
                hovertemplate="x: %{x:.2f}<br>V: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot3.x_RHSv,
                y=self.plot3.y_RHSv,
                fill="tozeroy",
                fillcolor="rgba(200, 0, 0, 0.2)",
                name="Case C",
                line=dict(color="red"),
                hovertemplate="x: %{x:.2f}<br>V: %{y:.2f}<extra></extra>",
            )
        )

        # ->
        fig.update_layout(title="Shear Force Envelope")
        fig.show()

        # return fig


class BendingMomentEnvelope:
    """Plot envelope of all 3 BMDs"""

    def __init__(self, case_1, case_2, case_3) -> None:
        """""" ""
        self.plot1 = case_1
        self.plot2 = case_2
        self.plot3 = case_3

        self.plot1.bmd()
        self.plot2.bmd()
        self.plot3.bmd()

    def matplotlib(self) -> None:
        """Plot envelope with Matplotlib's Pyplot"""
        # -> Plot
        plt.figure(figsize=(10, 5))
        plt.grid(which="major", linestyle=":", linewidth="0.5", color="black")
        plt.grid(which="minor", linestyle=":", linewidth="0.5", color="grey")
        plt.minorticks_on()
        plt.plot(self.plot1.x_RHSv, np.zeros(shape=1000), color="black")
        plt.plot(self.plot1.x_LHSv, np.zeros(shape=1000), color="black")

        # ->
        plt.axhline(y=0, color="k", linestyle="--")
        # plt.xlim(0, self.plot1.length)

        # -> Case A
        plt.plot(
            self.plot1.x_range_LHS,
            self.plot1.y_range_LHS,
            label="Case A",
            color="green",
        )
        plt.plot(self.plot1.x_range_RHS, self.plot1.y_range_RHS, color="green")

        # -> Case B
        plt.plot(
            self.plot2.x_range_LHS, self.plot2.y_range_LHS, label="Case B", color="blue"
        )
        plt.plot(self.plot2.x_range_RHS, self.plot2.y_range_RHS, color="blue")

        # -> Case C
        plt.plot(
            self.plot3.x_range_LHS, self.plot3.y_range_LHS, label="Case C", color="red"
        )
        plt.plot(self.plot3.x_range_RHS, self.plot3.y_range_RHS, color="red")

        # ->
        plt.title(label="Bending Moment Envelope")
        plt.gca().invert_yaxis()
        plt.show()

    def plotly(self):
        """Plot Envelope with Plotly's Graph Objects"""
        # ->
        plot_layout = go.Layout(width=800, height=500)
        fig = go.Figure(layout=plot_layout)
        fig.add_hline(y=0)

        # -> Case A
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot1.x_range_LHS,
                y=self.plot1.y_range_LHS,
                fill="tozeroy",
                fillcolor="rgba(0, 200, 0, 0.0)",
                name="Case A",
                line=dict(color="green"),
                hovertemplate="x: %{x:.2f}<br>M: %{y:.2f}<extra></extra>",
            )
        )  # Note hover templ in HTML
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot1.x_range_RHS,
                y=self.plot1.y_range_RHS,
                fill="tozeroy",
                fillcolor="rgba(0, 200, 0, 0.0)",
                name="Case A",
                line=dict(color="green"),
                hovertemplate="x: %{x:.2f}<br>M: %{y:.2f}<extra></extra>",
            )
        )

        # -> Case B
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot2.x_range_LHS,
                y=self.plot2.y_range_LHS,
                fill="tozeroy",
                fillcolor="rgba(0, 0, 200, 0.0)",
                name="Case B",
                line=dict(color="blue"),
                hovertemplate="x: %{x:.2f}<br>M: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot2.x_range_RHS,
                y=self.plot2.y_range_RHS,
                fill="tozeroy",
                fillcolor="rgba(0, 0, 200, 0.0)",
                name="Case B",
                line=dict(color="blue"),
                hovertemplate="x: %{x:.2f}<br>M: %{y:.2f}<extra></extra>",
            )
        )

        # -> Case C
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot3.x_range_LHS,
                y=self.plot3.y_range_LHS,
                fill="tozeroy",
                fillcolor="rgba(200, 0, 0, 0.0)",
                name="Case C",
                line=dict(color="red"),
                hovertemplate="x: %{x:.2f}<br>M: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            trace=go.Scatter(
                x=self.plot3.x_range_RHS,
                y=self.plot3.y_range_RHS,
                fill="tozeroy",
                fillcolor="rgba(200, 0, 0, 0.0)",
                name="Case C",
                line=dict(color="red"),
                hovertemplate="x: %{x:.2f}<br>M: %{y:.2f}<extra></extra>",
            )
        )

        # ->
        fig.update_layout(
            title="Bending Moment Envelope", yaxis=dict(autorange="reversed")
        )
        fig.show()

        # return fig


# This class can now be used to compare the shear force diagrams of three different cases.
# Note: The actual plotting part in the 'plot_sfd_comparison' method requires matplotlib, which should be installed in your environment.
