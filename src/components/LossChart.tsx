import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend
);

const LossChart = ({
  lossHistory,
  problemType,
}: {
  lossHistory: {
    loss: number;
    metric: number;
    val_loss?: number;
    val_metric?: number;
  }[];
  problemType: string;
}) => {

  if (lossHistory.length === 0) {
    return (
      <div style={{ marginTop: 20, textAlign: "center" }}>
        <h3>Training Metrics</h3>
        <p>No training data available yet. Please train the model.</p>
      </div>
    );
  }

  const metricLabel = problemType === "Regression" ? "MAE" : "Accuracy";
  const valMetricLabel = problemType === "Regression" ? "RMSE" : "Val Accuracy";

  const data = {
    labels: lossHistory.map((_, i) => `${i + 1}`),
    datasets: [
      {
        label: "Loss",
        data: lossHistory.map((h) => h.loss),
        borderColor: "#e74c3c",
        backgroundColor: "rgba(231, 76, 60, 0.2)",
        yAxisID: "y",
        fill: false,
        tension: 0.4,
      },
      {
        label: metricLabel,
        data: lossHistory.map((h) => h.metric || 0),
        borderColor: "#2ecc71",
        backgroundColor: "rgba(46, 204, 113, 0.2)",
        yAxisID: "y1",
        fill: false,
        tension: 0.4,
      },
      {
        label: "Validation Loss",
        data: lossHistory.map((h) => h.val_loss ?? null),
        borderColor: "#f1c40f",
        backgroundColor: "rgba(241, 196, 15, 0.2)",
        yAxisID: "y",
        fill: false,
        tension: 0.4,
      },
      {
        label: valMetricLabel,
        data: lossHistory.map((h) => h.val_metric ?? null),
        borderColor: "#9b59b6",
        backgroundColor: "rgba(155, 89, 182, 0.2)",
        yAxisID: problemType === "Regression" ? "y2" : "y1",
        fill: false,
        tension: 0.4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true, // Show legend
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        title: { display: false, text: "Loss" },
        ticks: { font: { size: 8 } },
        grid: {
          drawOnChartArea: false,
        },
      },
      y1: {
        position: "right" as const,
        beginAtZero: true,
        max: problemType === "Regression" ? undefined : 1,
        title: { display: false, text: metricLabel },
        ticks: { font: { size: 8 } },
        grid: {
          drawOnChartArea: false,
        },
      },
      ...(problemType === "Regression" && {
        y2: {
          position: "right" as const,
          beginAtZero: true,
          title: { display: false, text: valMetricLabel },
          ticks: { font: { size: 8 } },
          grid: {
            drawOnChartArea: false,
          },
        },
      }),
      x: {
        title: { display: true, text: "Epoch" },
        ticks: { font: { size: 8 } },
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{ flex: 1, minHeight: 0 }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
};

export default LossChart;