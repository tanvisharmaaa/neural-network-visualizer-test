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

const LossChart = ({ lossHistory }: { lossHistory: { loss: number; accuracy: number; val_loss?: number }[] }) => {
  console.log("LossChart received lossHistory:", lossHistory);

  const [visibleMetrics, setVisibleMetrics] = useState({
    loss: true,
    accuracy: true,
    val_loss: true,
  });

  if (lossHistory.length === 0) {
    return (
      <div style={{ marginTop: 20, textAlign: "center" }}>
        <h3>Training Metrics</h3>
        <p>No training data available yet. Please train the model.</p>
      </div>
    );
  }

  const data = {
    labels: lossHistory.map((_, i) => `Epoch ${i + 1}`),
    datasets: [
      visibleMetrics.loss && {
        label: "Loss",
        data: lossHistory.map((h) => h.loss),
        borderColor: "#e74c3c",
        yAxisID: "y",
        fill: false,
        tension: 0.4,
      },
      visibleMetrics.accuracy && {
        label: "Accuracy",
        data: lossHistory.map((h) => h.accuracy),
        borderColor: "#2ecc71",
        yAxisID: "y1",
        fill: false,
        tension: 0.4,
      },
      visibleMetrics.val_loss && {
        label: "Validation Loss",
        data: lossHistory.map((h) => h.val_loss ?? null),
        borderColor: "#f1c40f",
        yAxisID: "y",
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
        max: 1,
        title: { display: true, text: "Accuracy" },
      },
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
            checked={visibleMetrics.accuracy}
            onChange={() => setVisibleMetrics({ ...visibleMetrics, accuracy: !visibleMetrics.accuracy })}
          />
          Accuracy
        </label>
        <label>
          <input
            type="checkbox"
            checked={visibleMetrics.val_loss}
            onChange={() => setVisibleMetrics({ ...visibleMetrics, val_loss: !visibleMetrics.val_loss })}
          />
          Validation Loss
        </label>
      </div>
      <Line data={data} options={options} />
    </div>
  );
};

export default LossChart;