// LossChart.tsx
import { useState } from "react";
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

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend);

const LossChart = ({ lossHistory, problemType }: { lossHistory: { loss: number; metric: number; val_loss?: number; val_metric?: number }[]; problemType: string }) => {
  const [visibleMetrics, setVisibleMetrics] = useState({
    loss: true,
    metric: true,
    val_loss: true,
    val_metric: true,
  });

  if (lossHistory.length === 0) {
    return (
      <div style={{ marginTop: 20, textAlign: "center" }}>
        <h3>Training Metrics</h3>
        <p>No training data available yet. Please train the model.</p>
      </div>
    );
  }

  const metricLabel = problemType === "Regression" ? 'MAE' : 'Accuracy';
  const valMetricLabel = problemType === "Regression" ? 'RMSE' : 'Val Accuracy';

  const data = {
    labels: lossHistory.map((_, i) => `Epoch ${i + 1}`),
    datasets: [
      visibleMetrics.loss && {
        label: "Loss",
        data: lossHistory.map((h) => h.loss),
        borderColor: "#e74c3c",
        backgroundColor: "rgba(231, 76, 60, 0.2)",
        yAxisID: "y",
        fill: false,
        tension: 0.4,
      },
      visibleMetrics.metric && {
        label: metricLabel,
        data: lossHistory.map((h) => h.metric || 0),
        borderColor: "#2ecc71",
        backgroundColor: "rgba(46, 204, 113, 0.2)",
        yAxisID: "y1",
        fill: false,
        tension: 0.4,
      },
      visibleMetrics.val_loss && {
        label: "Validation Loss",
        data: lossHistory.map((h) => h.val_loss ?? null),
        borderColor: "#f1c40f",
        backgroundColor: "rgba(241, 196, 15, 0.2)",
        yAxisID: "y",
        fill: false,
        tension: 0.4,
      },
      visibleMetrics.val_metric && {
        label: valMetricLabel,
        data: lossHistory.map((h) => h.val_metric ?? null),
        borderColor: "#9b59b6",
        backgroundColor: "rgba(155, 89, 182, 0.2)",
        yAxisID: problemType === "Regression" ? "y2" : "y1",
        fill: false,
        tension: 0.4,
      },
    ].filter(Boolean) as any[],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        title: { display: true, text: "Loss" },
      },
      y1: {
        position: "right" as const,
        beginAtZero: true,
        max: problemType === "Regression" ? undefined : 1,
        title: { display: true, text: metricLabel },
        grid: {
          drawOnChartArea: false, // Avoid overlapping grids
        },
      },
      ...(problemType === "Regression" && {
        y2: {
          position: "right" as const,
          beginAtZero: true,
          title: { display: true, text: valMetricLabel },
          grid: {
            drawOnChartArea: false,
          },
        },
      }),
      x: {
        title: { display: true, text: "Epochs" },
      },
    },
  };

  return (
    <div style={{ marginTop: 20 }}>
      <h3 style={{ textAlign: "center" }}>Training Metrics</h3>
      <div style={{ marginBottom: 10 }}>
        <label style={{ marginRight: 10 }}>
          <input
            type="checkbox"
            checked={visibleMetrics.loss}
            onChange={() => setVisibleMetrics({ ...visibleMetrics, loss: !visibleMetrics.loss })}
          />
          Loss
        </label>
        <label style={{ marginRight: 10 }}>
          <input
            type="checkbox"
            checked={visibleMetrics.metric}
            onChange={() => setVisibleMetrics({ ...visibleMetrics, metric: !visibleMetrics.metric })}
          />
          {metricLabel}
        </label>
        <label style={{ marginRight: 10 }}>
          <input
            type="checkbox"
            checked={visibleMetrics.val_loss}
            onChange={() => setVisibleMetrics({ ...visibleMetrics, val_loss: !visibleMetrics.val_loss })}
          />
          Validation Loss
        </label>
        <label>
          <input
            type="checkbox"
            checked={visibleMetrics.val_metric}
            onChange={() => setVisibleMetrics({ ...visibleMetrics, val_metric: !visibleMetrics.val_metric })}
          />
          {valMetricLabel}
        </label>
      </div>
      <Line data={data} options={options} />
    </div>
  );
};

export default LossChart;