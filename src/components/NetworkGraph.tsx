import React, { useEffect, useRef, useState } from "react";

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
  neuronGradients: Map<string, number>;
  showWeights: boolean;
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
  neuronGradients,
  showWeights,
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

          <filter id="errorGlowHigh">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          <filter id="errorGlowMed">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
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

        {/* Connection Lines */}
        {positions.slice(0, -1).map((fromLayer, layerIdx) => {
          const toLayer = positions[layerIdx + 1];
          return fromLayer.map((from, fromIdx) =>
            toLayer.map((to, toIdx) => {
              const weight =
                weights[layerIdx]?.[fromIdx]?.[toIdx] ?? null;
              return (
                <g key={`line-${layerIdx}-${fromIdx}-${toIdx}`}>
                  <line
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke="#aac4f6"
                    strokeWidth={2}
                    opacity={0.6}
                  />
                  {showWeights && weight !== null && (
                    <text
                      x={(from.x + to.x) / 2}
                      y={(from.y + to.y) / 2 - 6}
                      fontSize="10"
                      fill="#444"
                      textAnchor="middle"
                    >
                      w={weight.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })
          );
        })}

        {/* Pulse Animation */}
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
              r={6}
              fill={direction === "forward" ? "url(#pulseForward)" : "url(#pulseBackward)"}
              opacity={0.9}
              style={{ filter: direction === "backward" ? "url(#glow)" : "none" }}
            />
          );
        })}

        {/* Neurons */}
        {positions.map((layer, layerIdx) =>
          layer.map((pos, neuronIdx) => {
            const key = `${layerIdx}-${neuronIdx}`;
            const isUpdated = justUpdatedNeurons.some(
              (n) => n.layer === layerIdx && n.index === neuronIdx
            );
            const value = neuronValues.get(key);
            const equation = neuronEquations.get(key);
            const gradient = neuronGradients.get(key) || 0;

            let glowFilter = "none";
            if (gradient > 0.6) glowFilter = "url(#errorGlowHigh)";
            else if (gradient > 0.3) glowFilter = "url(#errorGlowMed)";
            else if (isUpdated) glowFilter = "url(#glow)";

            return (
              <g key={`n-${layerIdx}-${neuronIdx}`}>
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={neuronRadius}
                  fill="#ffffff"
                  stroke="#6ba4ff"
                  strokeWidth={2.5}
                  filter={glowFilter}
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
                    y={pos.y - 28}
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
    </div>
  );
};

export default NetworkGraph;
