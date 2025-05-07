// ✅ NetworkGraph.tsx
import React, { useEffect, useState } from "react";

interface Pulse {
  from: { x: number; y: number };
  to: { x: number; y: number };
  progress: number;
  direction: "forward" | "backward";
}

interface Props {
  inputNeurons: number;
  hiddenLayers: number[];
  outputNeurons: number;
  weights: number[][][];
  biases: number[][];
  activationFunction: string;
  pulses: Pulse[];
  neuronEquations: Map<string, string>;
  neuronValues: Map<string, number>;
}

const NetworkGraph: React.FC<Props> = ({
  inputNeurons,
  hiddenLayers,
  outputNeurons,
  weights,
  biases,
  activationFunction,
  pulses,
  neuronEquations,
  neuronValues,
}) => {
  const layers = [inputNeurons, ...hiddenLayers, outputNeurons];
  const neuronRadius = 20;
  const hSpacing = 140;
  const vSpacing = 80;

  const [justUpdatedNeurons, setJustUpdatedNeurons] = useState<
    { layer: number; index: number }[]
  >([]);

  const positions: { x: number; y: number }[][] = layers.map((count, layerIdx) => {
    const totalHeight = (count - 1) * vSpacing;
    return Array.from({ length: count }, (_, neuronIdx) => ({
      x: layerIdx * hSpacing + 70,
      y: neuronIdx * vSpacing - totalHeight / 2 + 200,
    }));
  });

  useEffect(() => {
    const updated: { layer: number; index: number }[] = [];

    pulses.forEach((pulse) => {
      if (pulse.progress >= 1) {
        positions.forEach((layer, layerIdx) => {
          layer.forEach((pos, idx) => {
            if (Math.abs(pulse.to.x - pos.x) < 1 && Math.abs(pulse.to.y - pos.y) < 1) {
              const alreadyUpdated = justUpdatedNeurons.some(
                (n) => n.layer === layerIdx && n.index === idx
              );
              if (!alreadyUpdated) {
                updated.push({ layer: layerIdx, index: idx });
              }
            }
          });
        });
      }
    });

    if (updated.length > 0) {
      setJustUpdatedNeurons((prev) => [...prev, ...updated]);
      setTimeout(() => {
        setJustUpdatedNeurons((prev) =>
          prev.filter((n) => !updated.some((u) => u.layer === n.layer && u.index === n.index))
        );
      }, 500);
    }
  }, [pulses]);

  const isTrained = weights.length > 0 && biases.length > 0;

  return (
    <div style={{ position: "relative", paddingBottom: "100px" }}>
      <svg width="100%" height="600">
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          <linearGradient id="pulseForward" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ffb347" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#ffcc80" stopOpacity="0.8" />
          </linearGradient>

          <linearGradient id="pulseBackward" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ff6b81" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#ffa3b1" stopOpacity="0.8" />
          </linearGradient>
        </defs>

        {positions.slice(0, -1).map((fromLayer, layerIdx) => {
          const toLayer = positions[layerIdx + 1];
          return fromLayer.map((from, fromIdx) =>
            toLayer.map((to, toIdx) => (
              <line
                key={`conn-${layerIdx}-${fromIdx}-${toIdx}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke="#aac4f6"
                strokeWidth={2}
                opacity={0.6}
              />
            ))
          );
        })}

        {pulses.map((pulse, idx) => {
          const { from, to, progress, direction } = pulse;
          const dx = to.x - from.x;
          const dy = to.y - from.y;
          const length = Math.sqrt(dx * dx + dy * dy);
          const directionFactor = direction === "forward" ? 1 : -1;
          const offset = directionFactor * (neuronRadius / length) * 0.5;
          const adjustedProgress = Math.min(Math.max(progress + offset, 0), 1);
          const cx = from.x + dx * adjustedProgress;
          const cy = from.y + dy * adjustedProgress;

          return (
            <circle
              key={`pulse-${idx}`}
              cx={cx}
              cy={cy}
              r={5}
              fill={direction === "forward" ? "url(#pulseForward)" : "url(#pulseBackward)"}
              opacity={0.9}
            />
          );
        })}

        {positions.map((layer, layerIdx) =>
          layer.map((pos, neuronIdx) => {
            const isUpdated = justUpdatedNeurons.some(
              (n) => n.layer === layerIdx && n.index === neuronIdx
            );
            const key = `${layerIdx}-${neuronIdx}`;
            const value = neuronValues.get(key);
            const equation = neuronEquations.get(key);

            return (
              <g key={`n-${layerIdx}-${neuronIdx}`}>
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={neuronRadius}
                  fill="#ffffff"
                  stroke={isUpdated ? "#2ecc71" : "#6ba4ff"}
                  strokeWidth={3}
                  filter={isUpdated ? "url(#glow)" : "none"}
                />

                {isTrained && value !== undefined && !isNaN(value) && (
                  <text
                    x={pos.x}
                    y={pos.y}
                    fill="#444"
                    fontSize="10"
                    textAnchor="middle"
                    dominantBaseline="middle"
                  >
                    {value.toFixed(2)}
                  </text>
                )}

                {isTrained && equation && (
                  <text
                    x={pos.x}
                    y={pos.y - 30}
                    fill="#555"
                    fontSize="10"
                    textAnchor="middle"
                  >
                    {equation}
                  </text>
                )}
              </g>
            );
          })
        )}
      </svg>

      {isTrained && (
        <div style={{
          textAlign: "center",
          marginTop: "20px",
          fontSize: "16px",
          color: "#555",
          background: "#f8f8f8",
          padding: "10px",
          borderRadius: "10px",
          boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
          maxWidth: "600px",
          marginLeft: "auto",
          marginRight: "auto",
        }}>
          <div><b>Equation:</b> z = Σ(wx) + b</div>
          <div><b>Activation:</b> {activationFunction}(z)</div>
        </div>
      )}
    </div>
  );
};

export default NetworkGraph;
