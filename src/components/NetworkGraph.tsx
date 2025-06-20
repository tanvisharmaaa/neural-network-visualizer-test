import React, { useEffect } from "react";

interface Props {
  inputNeurons: number;
  hiddenLayers: number[];
  outputNeurons: number;
  weights: number[][][];
  biases: number[][];
  activationFunction: string;
  pulses: any[];
  neuronEquations: Map<string, string>;
  neuronValues: Map<string, number>;
  neuronGradients: Map<string, number>;
  showWeights: boolean;
  lineThicknessMode: "auto" | "fixed";
  zoomLevel: number;
  epochDisplay: number;
  problemType: string;
  currentInputs: number[];
  outputs: number[];
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
  lineThicknessMode,
  zoomLevel,
  epochDisplay,
  problemType,
  currentInputs,
  outputs,
}) => {
  useEffect(() => {
    console.log("NetworkGraph rerendered with weights:", weights.map(w => w.map(row => row.map(val => val.toFixed(2)))));
    console.log("Neuron equations:", Array.from(neuronEquations.entries()));
    console.log("Layer sizes:", [inputNeurons, ...hiddenLayers, outputNeurons]);
    console.log("Pulses:", pulses);
    console.log("Current inputs:", currentInputs);
    console.log("Outputs:", outputs);
  }, [weights, neuronEquations, pulses, currentInputs, outputs]);

  useEffect(() => {
    console.log("NetworkGraph rerendered with epochDisplay:", epochDisplay);
  }, [epochDisplay]);

  useEffect(() => {
    console.log("NetworkGraph mounted");
    return () => {
      console.log("NetworkGraph unmounted");
    };
  }, []);

  const getNeuronPositions = () => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const positions: { x: number; y: number }[] = [];
    let neuronIndex = 0;
    layerSizes.forEach((count, layerIdx) => {
      const totalHeight = (count - 1) * 100;
      for (let neuronIdx = 0; neuronIdx < count; neuronIdx++) {
        positions[neuronIndex] = {
          x: layerIdx * 150 + 50,
          y: neuronIdx * 100 - totalHeight / 2 + 250,
        };
        neuronIndex++;
      }
    });
    return positions;
  };

  const getNeuronIndex = (layerIdx: number, neuronIdx: number) => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    let index = 0;
    for (let i = 0; i < layerIdx; i++) {
      index += layerSizes[i];
    }
    return index + neuronIdx;
  };

  const getGlowIntensity = (layerIdx: number, neuronIdx: number) => {
    if (layerIdx === 0) return 0;
    const activation = neuronValues.get(`${layerIdx}-${neuronIdx}`) ?? 0;
    const intensity = Math.min(Math.abs(activation) * 5, 15);
    return intensity;
  };

  const getNeuronEquation = (layerIdx: number, neuronIdx: number) => {
    if (layerIdx === 0) return "Input Neuron";
    const incomingWeights = weights[layerIdx - 1]?.map(row => row[neuronIdx] ?? 0) || [];
    const bias = biases[layerIdx]?.[neuronIdx] ?? 0;
    const terms = incomingWeights.map((w, i) => {
      const inputLabel = layerIdx === 1 ? `x${i + 1}` : `h${i + 1}`;
      return `${w.toFixed(2)}·${inputLabel}`;
    }).join(" + ");
    return `z = ${terms} + ${bias.toFixed(2)}`;
  };

  const truncateEquation = (equation: string, maxLength: number = 20) => {
    if (equation.length > maxLength) {
      return equation.substring(0, maxLength - 3) + "...";
    }
    return equation;
  };

  const calculateTotalParameters = () => {
    let totalParams = 0;
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    let paramBreakdown = [];
    for (let layerIdx = 0; layerIdx < layerSizes.length - 1; layerIdx++) {
      const fromSize = layerSizes[layerIdx];
      const toSize = layerSizes[layerIdx + 1];
      const weights = fromSize * toSize;
      const biases = toSize;
      totalParams += weights + biases;
      paramBreakdown.push(`Layer ${layerIdx}→${layerIdx + 1}: ${weights} weights + ${biases} biases`);
    }
    console.log(`Parameter breakdown: ${paramBreakdown.join(", ")} | Total: ${totalParams}`);
    return totalParams;
  };

  const positions = getNeuronPositions();
  const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
  const totalParams = calculateTotalParameters();

  const outputLayerIdx = layerSizes.length - 1;
  const outputEquations = Array(outputNeurons).fill(0).map((_, neuronIdx) => {
    const key = `${outputLayerIdx}-${neuronIdx}`;
    return neuronEquations.get(key) || getNeuronEquation(outputLayerIdx, neuronIdx);
  });

  const getBezierPoint = (t: number, start: { x: number; y: number }, control: { x: number; y: number }, end: { x: number; y: number }) => {
    const u = 1 - t;
    const tt = t * t;
    const uu = u * u;
    const ut2 = 2 * u * t;

    const x = uu * start.x + ut2 * control.x + tt * end.x;
    const y = uu * start.y + ut2 * control.y + tt * end.y;

    return { x, y };
  };

  const svgWidth = layerSizes.length * 150 + 100;
  const svgHeight = Math.max(...layerSizes) * 100 + 300;

  return (
    <svg width="100%" height="700" viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ backgroundColor: "#e6f3ff" }}>
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <g transform={`scale(${zoomLevel})`}>
        {layerSizes.slice(0, -1).map((fromSize, layerIdx) => {
          const startIndex = layerIdx === 0 ? 0 : layerSizes.slice(0, layerIdx).reduce((a, b) => a + b, 0);
          return Array(fromSize).fill(0).map((_, fromIdx) =>
            Array(layerSizes[layerIdx + 1]).fill(0).map((_, toIdx) => {
              const fromGlobalIdx = startIndex + fromIdx;
              const toGlobalIdx = layerIdx + 1 < layerSizes.length
                ? startIndex + fromSize + toIdx
                : getNeuronIndex(layerIdx + 1, toIdx);
              const from = positions[fromGlobalIdx];
              const to = positions[toGlobalIdx];
              const midX = (from.x + to.x) / 2;
              const midY = (from.y + to.y) / 2;

              const weight = weights[layerIdx]?.[fromIdx]?.[toIdx] ?? 0;
              const strokeColor = "#888888";
              const strokeWidth = lineThicknessMode === "auto" ? 1 + Math.abs(weight) * 2 : 2;

              const staggerY = toIdx * 10;
              const offsetY = 10;

              return (
                <g key={`${layerIdx}-${fromIdx}-${toIdx}`}>
                  <line
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    opacity="0.8"
                  />
                  {showWeights && neuronValues.size > 0 && weight !== 0 && (
                    <text
                      x={midX}
                      y={midY + offsetY + staggerY}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize="10"
                      fill="#666"
                      style={{ backgroundColor: "rgba(255, 255, 255, 0.9)" }}
                    >
                      w={weight.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })
          );
        })}

{positions.map((pos, i) => {
  let layerIdx = 0;
  let neuronIdx = i;
  for (let l = 0; l < layerSizes.length; l++) {
    if (neuronIdx < layerSizes[l]) {
      layerIdx = l;
      break;
    }
    neuronIdx -= layerSizes[l];
  }
  const strokeColor = layerIdx === 0 ? "#ADD8E6" : layerIdx <= hiddenLayers.length ? "#90EE90" : "#FFA07A";
  const glowIntensity = getGlowIntensity(layerIdx, neuronIdx);
  const glowFilter = glowIntensity > 0 ? `url(#glow) drop-shadow(0 0 ${glowIntensity}px ${strokeColor})` : "none";

  const equationKey = `${layerIdx}-${neuronIdx}`;
  const fullEquation = neuronEquations.get(equationKey) || getNeuronEquation(layerIdx, neuronIdx);
  const displayedEquation = truncateEquation(fullEquation);

  const isOutputLayer = layerIdx === layerSizes.length - 1;
  const outputValue = isOutputLayer && outputs.length > 0 ? outputs[neuronIdx]?.toFixed(2) : null;

  return (
    <g key={i}>
      <circle
        cx={pos.x}
        cy={pos.y}
        r="20"
        fill="white"
        stroke={strokeColor}
        strokeWidth="2"
        style={{ filter: glowFilter }}
      />
      <title>{fullEquation !== "Input Neuron" ? fullEquation : ""}</title>
      <text
        x={pos.x}
        y={pos.y}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize="12"
      >
        {layerIdx === 0
          ? `x${neuronIdx + 1}`
          : layerIdx <= hiddenLayers.length
          ? `h${i - inputNeurons + 1}`
          : `o${i - inputNeurons - hiddenLayers.reduce((a, b) => a + b, 0) + 1}`}
      </text>
      {layerIdx === 0 && currentInputs.length > neuronIdx && (
        <text
          x={pos.x}
          y={pos.y + 25}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="10"
          fill="#666"
        >
          {currentInputs[neuronIdx].toFixed(2)}
        </text>
      )}
      {neuronValues.size > 0 && !isOutputLayer && (
        <text
          x={pos.x}
          y={pos.y + 25}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="10"
          fill="#666"
        >
          {neuronValues.get(`${layerIdx}-${neuronIdx}`)?.toFixed(2) || ""}
        </text>
      )}
      {isOutputLayer && outputs.length > 0 && outputValue && (
        <text
          x={pos.x + 50}
          y={pos.y}
          textAnchor="start"
          dominantBaseline="middle"
          fontSize="10"
          fill="#333"
        >
          {`Output: ${outputValue}`}
        </text>
      )}
      {isOutputLayer && outputs.length > 0 && displayedEquation && (
        <text
          x={pos.x + 50}
          y={pos.y + 15}
          textAnchor="start"
          dominantBaseline="middle"
          fontSize="9"
          fill="#333"
        >
          {`Eq: ${displayedEquation}`}
        </text>
      )}
      {neuronValues.size > 0 && !isOutputLayer && displayedEquation && (
        <text
          x={pos.x}
          y={pos.y - 40}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="9"
          fill="#333"
        >
          {displayedEquation}
        </text>
      )}
    </g>
  );
})}

        {pulses.map((pulse, i) => {
          const t = pulse.progress;
          const to = pulse.to || positions.find(p => Math.hypot(p.x - pulse.from.x, p.y - pulse.from.y) < 50);
          if (!to) return null;
          const point = getBezierPoint(t, pulse.from, pulse.control, to);
          const fillColor = pulse.direction === "forward" ? "green" : "red";
          return (
            <circle
              key={i}
              cx={point.x}
              cy={point.y}
              r="5"
              fill={fillColor}
              stroke="#000"
              strokeWidth="1"
            />
          );
        })}
      </g>

      <text
        x={50}
        y={svgHeight - 50}
        textAnchor="start"
        dominantBaseline="middle"
        fontSize="12"
        fill="#333"
      >
        {`Epoch: ${epochDisplay} | Total Parameters: ${totalParams} | Activation: ${activationFunction} | Problem Type: ${problemType}`}
      </text>
    </svg>
  );
};

export default NetworkGraph;