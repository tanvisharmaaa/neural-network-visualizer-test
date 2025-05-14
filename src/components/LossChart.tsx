// LossChart.tsx
import React from "react";
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

const LossChart = ({ lossHistory }: { lossHistory: number[] }) => {
  const data = {
    labels: lossHistory.map((_, i) => `Epoch ${i + 1}`),
    datasets: [
      {
        label: "Loss",
        data: lossHistory,
        borderColor: "#e74c3c",
        fill: false,
        tension: 0.4,
      },
    ],
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
      x: {
        title: { display: true, text: "Epochs" },
      },
    },
  };

  return (
    <div style={{ marginTop: 20 }}>
      <h3 style={{ textAlign: "center" }}>Loss Curve</h3>
      <Line data={data} options={options} />
    </div>
  );
};

export default LossChart;
