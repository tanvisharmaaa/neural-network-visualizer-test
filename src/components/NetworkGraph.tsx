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
  featureLabels?: string[];
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
  featureLabels,
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
          y: neuronIdx * 100 - totalHeight / 2 + 50, // Restored +50 offset to maintain alignment
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

  // Calculate actual bounds from neuron positions - minimal side padding
  // Account for neuron radius (20) to ensure neurons aren't cut off, but minimize whitespace
  const paddingX = 25; 
  const minX = Math.min(...positions.map(p => p.x)) - paddingX;
  const maxX = Math.max(...positions.map(p => p.x)) + paddingX;
  const minY = Math.min(...positions.map(p => p.y));
  const maxY = Math.max(...positions.map(p => p.y));
  
  
  const contentWidth = maxX - minX;
  
  
  const svgWidth = contentWidth;

  
  const graphTopY = minY;
  const graphBottomY = maxY;
  const centerXForText = (minX + maxX) / 2; 
  const footerFontSize = 8; 
  const eqFontSize = 8; 
  const footerY = graphTopY - 1; 
  const eqStartY = graphBottomY + 1; 
  const eqLineGap = 12 
  

  const totalMinY = Math.min(minY, footerY) - 3; 
  const equationsHeight = outputEquations.length > 0 
    ? eqStartY + (outputEquations.length - 1) * eqLineGap + eqFontSize 
    : graphBottomY; 
  const totalMaxY = Math.max(maxY, equationsHeight) + 3;

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

  
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2; 

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={`${minX} ${totalMinY} ${svgWidth} ${totalMaxY - totalMinY}`}
      style={{ backgroundColor: "#ffffff", display: "block", margin: "0 auto" }}
      preserveAspectRatio="xMidYMin meet"
    >
      <g transform={`translate(${centerX}, ${centerY}) scale(${zoomLevel}) translate(${-centerX}, ${-centerY})`}>
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

                  const staggerY = toIdx * 15;
                  const offsetY = 15;
                  const offsetX = (toIdx % 2 === 0 ? 1 : -1) * 8; 

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
                          <g>
                            <rect
                              x={midX + offsetX - 25}
                              y={midY + offsetY + staggerY - 8}
                              width="50"
                              height="16"
                              fill="rgba(255, 255, 255, 0.9)"
                              stroke="none"
                              rx="3"
                            />
                            <text
                              x={midX + offsetX}
                              y={midY + offsetY + staggerY}
                              textAnchor="middle"
                              dominantBaseline="middle"
                              fontSize="10"
                              fill="#666"
                            >
                              w={weight.toFixed(2)}
                            </text>
                          </g>
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


          const style = {
            filter: "none",
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
                fontSize="10"
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
              {layerIdx === 0 && featureLabels && featureLabels[neuronIdx] && (
                <text
                  x={pos.x - 40}
                  y={pos.y}
                  textAnchor="end"
                  dominantBaseline="middle"
                  fontSize="10"
                  fill="#666"
                  style={{ fontWeight: "500" }}
                >
                  {featureLabels[neuronIdx]}
                </text>
              )}
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

      {/* Footer text centered at top */}
      <text
        x={centerXForText}
        y={footerY}
        textAnchor="middle"
        dominantBaseline="hanging"
        fontSize={footerFontSize}
        fill="#333"
        fontFamily="Arial, sans-serif"
      >
        {`Epoch: ${epochDisplay} | Params: ${totalParams} | ${activationFunction} | ${problemType}`}
      </text>

      {/* Output equations centered, close to graph bottom */}
      {hasDataset &&
        outputEquations.length > 0 &&
        isTrained &&
        outputEquations.map((equation, index) => (
          <g key={index}>
            <text
              x={centerXForText}
              y={eqStartY + index * eqLineGap}
              textAnchor="middle"
              dominantBaseline="hanging"
              fontSize={eqFontSize}
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
