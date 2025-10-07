
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
  showWeights: boolean;
  lineThicknessMode: "auto" | "fixed";
  zoomLevel: number;
  epochDisplay: number;
  problemType: string;
  currentInputs: number[];
  outputs: number[];
  hasDataset: boolean;
  isTrained: boolean;
  displayedWeights: Map<string, number>;
  displayedConnections?: any;
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
  showWeights,
  lineThicknessMode,
  zoomLevel,
  epochDisplay,
  problemType,
  currentInputs,
  outputs,
  hasDataset,
  isTrained,
  displayedWeights,
}) => {
  useEffect(() => {
   
  }, [weights, displayedWeights, neuronValues]);

  const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];

  const getNeuronPositions = () => {
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
    let index = 0;
    for (let i = 0; i < layerIdx; i++) index += layerSizes[i];
    return index + neuronIdx;
  };

  const layerMaxAvgAbsIncoming: number[] = (() => {
    const maxes: number[] = Array(layerSizes.length).fill(1);
    for (let L = 1; L < layerSizes.length; L++) {
      const prevSize = layerSizes[L - 1];
      const thisSize = layerSizes[L];
      const W = weights[L - 1] || [];
      let layerMax = 0;
      for (let j = 0; j < thisSize; j++) {
        let sumAbs = 0;
        for (let i = 0; i < prevSize; i++) {
          const w = W[i]?.[j] ?? 0;
          sumAbs += Math.abs(w);
        }
        const avgAbs = prevSize > 0 ? sumAbs / prevSize : 0;
        layerMax = Math.max(layerMax, avgAbs);
      }
      maxes[L] = layerMax || 1;
    }
    return maxes;
  })();

  const getGlowForNeuron = (layerIdx: number, neuronIdx: number) => {
    const a = Math.abs(neuronValues.get(`${layerIdx}-${neuronIdx}`) ?? 0);

    if (layerIdx === 0) {
      
      return Math.max(0, Math.min(1, Math.abs(a) * 0.5)); 
    }

    const prevSize = layerSizes[layerIdx - 1];
    let sumAbs = 0;
    for (let i = 0; i < prevSize; i++) {
      const w = weights[layerIdx - 1]?.[i]?.[neuronIdx] ?? 0;
      sumAbs += Math.abs(w);
    }
    const avgAbs = prevSize > 0 ? sumAbs / prevSize : 0;

    const relW = avgAbs / (layerMaxAvgAbsIncoming[layerIdx] || 1);

    const importance = a * relW;

    return Math.max(0, Math.min(1, importance * 3)); 
  };

  const getNeuronEquation = (layerIdx: number, neuronIdx: number) => {
    if (!hasDataset || layerIdx === 0 || !isTrained) return "";
    const incomingWeights =
      weights[layerIdx - 1]?.map((row) => row[neuronIdx] ?? 0) || [];
    const bias = biases[layerIdx - 1]?.[neuronIdx] ?? 0;
    const terms = incomingWeights
      .map((w, i) => {
        const inputLabel = layerIdx === 1 ? `x${i + 1}` : `h${i + 1}`;
        return `${w.toFixed(2)}·${inputLabel}`;
      })
      .join(" + ");
    return `z = ${terms} + ${bias.toFixed(2)}`;
  };

  const getOutputEquations = () => {
    if (!hasDataset || !isTrained) return [];
    const outputLayerIdx = layerSizes.length - 1;
    const equations: string[] = [];
    const incomingWeightsAll =
      weights[outputLayerIdx - 1]?.map((row) => row) || [];
    const biasesOutput = biases[outputLayerIdx - 1] ?? [];
    const activation =
      problemType === "Regression"
        ? ""
        : problemType === "Classification" && outputNeurons > 1
        ? "softmax"
        : "sigmoid";

    for (let neuronIdx = 0; neuronIdx < outputNeurons; neuronIdx++) {
      const incomingWeights = incomingWeightsAll.map(
        (row) => row[neuronIdx] ?? 0
      );
      const bias = biasesOutput[neuronIdx] ?? 0;
      const inputLabels = incomingWeights.map((_, i) => `h${i + 1}`);
      const terms = incomingWeights
        .map((w, i) => `${w.toFixed(2)}·${inputLabels[i]}`)
        .join(" + ");
      const zExpr = `${terms} + ${bias.toFixed(2)}`;
      equations.push(
        `o${neuronIdx + 1} = ${activation ? `${activation}(${zExpr})` : zExpr}`
      );
    }
    return equations;
  };

  const truncateEquation = (equation: string, maxLength: number = 200) =>
    equation.length > maxLength
      ? equation.substring(0, maxLength - 3) + "..."
      : equation;

  const calculateTotalParameters = () => {
    let totalParams = 0;
    let paramBreakdown: string[] = [];
    for (let layerIdx = 0; layerIdx < layerSizes.length - 1; layerIdx++) {
      const fromSize = layerSizes[layerIdx];
      const toSize = layerSizes[layerIdx + 1];
      const weightsCount = fromSize * toSize;
      const biasesCount = toSize;
      totalParams += weightsCount + biasesCount;
      paramBreakdown.push(
        `Layer ${layerIdx}→${
          layerIdx + 1
        }: ${weightsCount} weights + ${biasesCount} biases`
      );
    }
    
    return totalParams;
  };

  const positions = getNeuronPositions();
  const totalParams = calculateTotalParameters();
  const outputEquations = getOutputEquations();

  const svgWidth = layerSizes.length * 150 + 100;
  const svgHeight = Math.max(...layerSizes) * 100 + 400;

  const getBezierPoint = (
    t: number,
    start: { x: number; y: number },
    control: { x: number; y: number },
    end: { x: number; y: number }
  ) => {
    const u = 1 - t;
    const tt = t * t;
    const uu = u * u;
    const ut2 = 2 * u * t;
    const x = uu * start.x + ut2 * control.x + tt * end.x;
    const y = uu * start.y + ut2 * control.y + tt * end.y;
    return { x, y };
  };

  return (
    <svg
      width="100%"
      height="800"
      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
      style={{ backgroundColor: "#ffffff" }}
    >
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
        {/* Edges */}
        {layerSizes.slice(0, -1).map((fromSize, layerIdx) => {
          const startIndex =
            layerIdx === 0
              ? 0
              : layerSizes.slice(0, layerIdx).reduce((a, b) => a + b, 0);
          return Array(fromSize)
            .fill(0)
            .map((_, fromIdx) =>
              Array(layerSizes[layerIdx + 1])
                .fill(0)
                .map((_, toIdx) => {
                  const fromGlobalIdx = startIndex + fromIdx;
                  const toGlobalIdx =
                    layerIdx + 1 < layerSizes.length
                      ? startIndex + fromSize + toIdx
                      : getNeuronIndex(layerIdx + 1, toIdx);
                  const from = positions[fromGlobalIdx];
                  const to = positions[toGlobalIdx];
                  const midX = (from.x + to.x) / 2;
                  const midY = (from.y + to.y) / 2;

                  const pulse = pulses.find(
                    (p) =>
                      p.layerIdx === layerIdx &&
                      p.fromIdx === fromIdx &&
                      p.toIdx === toIdx
                  );
                  const connectionKey = `${layerIdx}-${fromIdx}-${toIdx}`;
                  const weight = displayedWeights.get(connectionKey);
                  const isActive = !!pulse;
                  const strokeColor =
                    isActive && pulse?.direction === "backward"
                      ? "#ff0000"
                      : "#888888";
                  const strokeWidth =
                    lineThicknessMode === "auto" && isTrained
                      ? 1 + Math.abs(weight || 0) * 2
                      : 2;

                  const staggerY = toIdx * 10;
                  const offsetY = 10;

                  return (
                    <g key={connectionKey}>
                      <line
                        x1={from.x}
                        y1={from.y}
                        x2={to.x}
                        y2={to.y}
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        opacity={isActive ? 1 : 0.8}
                      />
                      {showWeights &&
                        hasDataset &&
                        isTrained &&
                        weight !== undefined && (
                          <text
                            x={midX}
                            y={midY + offsetY + staggerY}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fontSize="10"
                            fill="#666"
                            style={{
                              backgroundColor: "rgba(255, 255, 255, 0.9)",
                            }}
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

          const strokeColor =
            layerIdx === 0
              ? "#ADD8E6"
              : layerIdx <= hiddenLayers.length
              ? "#90EE90"
              : "#FFA07A";

          const importance = getGlowForNeuron(layerIdx, neuronIdx); // 0..1
          const glowPx = 14 * importance; // radius
          const glowAlpha = importance; // opacity 0..1
          const glowColor = `rgba(255, 165, 0, ${glowAlpha})`; // orange glow

          const style = {
            filter:
              importance > 0.01
                ? `drop-shadow(0 0 ${glowPx}px ${glowColor})`
                : "none",
          };

          const equationKey = `${layerIdx}-${neuronIdx}`;
          const fullEquation = isTrained
            ? neuronEquations.get(equationKey) ||
              getNeuronEquation(layerIdx, neuronIdx)
            : "";

          const isOutputLayer = layerIdx === layerSizes.length - 1;
          const outputValue =
            isOutputLayer && outputs.length > 0
              ? outputs[neuronIdx]?.toFixed(2)
              : null;

          return (
            <g key={i}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r="20"
                fill="white"
                stroke={strokeColor}
                strokeWidth="2"
                style={style}
              />
              <title>{fullEquation !== "" ? fullEquation : ""}</title>{" "}
              {/* Full equation in tooltip */}
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
                  : `o${
                      i -
                      inputNeurons -
                      hiddenLayers.reduce((a, b) => a + b, 0) +
                      1
                    }`}
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
                  {neuronValues.get(`${layerIdx}-${neuronIdx}`)?.toFixed(2) ||
                    ""}
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
            </g>
          );
        })}

        {/* Pulses */}

        {pulses.map((pulse, i) => {
          const t = pulse.progress;
          const positionsAll = getNeuronPositions();
          const to =
            pulse.to ||
            positionsAll.find(
              (p) => Math.hypot(p.x - pulse.from.x, p.y - pulse.from.y) < 50
            );
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
        y={svgHeight - 100}
        textAnchor="start"
        dominantBaseline="middle"
        fontSize="12"
        fill="#333"
      >
        {`Epoch: ${epochDisplay} | Total Parameters: ${totalParams} | Activation: ${activationFunction} | Problem Type: ${problemType}`}
      </text>

      {hasDataset &&
        outputEquations.length > 0 &&
        isTrained &&
        outputEquations.map((equation, index) => (
          <g key={index}>
            <text
              x={50}
              y={svgHeight - 60 + index * 20}
              textAnchor="start"
              dominantBaseline="middle"
              fontSize="14"
              fill="#333"
            >
              {truncateEquation(equation)}
            </text>
            <title>{equation}</title>{" "}
            {/* Full equation in tooltip for truncated ones */}
          </g>
        ))}
    </svg>
  );
};

export default NetworkGraph;
