import React from "react"

interface Props {
  inputNeurons: number
  hiddenLayers: number[]
  outputNeurons: number
  weights: number[][][]
  biases: number[][]
}

const NetworkGraph: React.FC<Props> = ({ inputNeurons, hiddenLayers, outputNeurons, weights, biases }) => {
  const layers = [inputNeurons, ...hiddenLayers, outputNeurons]
  const neuronRadius = 20
  const hSpacing = 120
  const vSpacing = 70

  const positions: { x: number; y: number }[][] = layers.map((count, layerIdx) => {
    const totalHeight = (count - 1) * vSpacing
    return Array.from({ length: count }, (_, neuronIdx) => ({
      x: layerIdx * hSpacing + 50,
      y: neuronIdx * vSpacing - totalHeight / 2 + 200,
    }))
  })

  return (
    <svg width="100%" height="500" style={{ backgroundColor: "#1e1e2f" }}>
      {/* Draw connections with weights */}
      {weights.map((layerWeights, layerIdx) =>
        layerWeights.map((fromWeights, fromIdx) =>
          fromWeights.map((weight, toIdx) => {
            const from = positions[layerIdx][fromIdx]
            const to = positions[layerIdx + 1][toIdx]
            const color = weight >= 0 ? "#3498db" : "#e74c3c"
            const opacity = Math.min(Math.abs(weight) * 5, 1)

            const midX = (from.x + to.x) / 2
            const midY = (from.y + to.y) / 2

            return (
              <g key={`${layerIdx}-${fromIdx}-${toIdx}`}>
                <line
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={color}
                  strokeWidth={2}
                  opacity={opacity}
                />
                <text
                  x={midX}
                  y={midY}
                  fill="#ccc"
                  fontSize="10"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {weight.toFixed(2)}
                </text>
              </g>
            )
          })
        )
      )}

      {/* Draw neurons + biases */}
      {positions.map((layer, layerIdx) =>
        layer.map((pos, neuronIdx) => {
          const biasLayerIdx = layerIdx - 1
          const biasVal = biases?.[biasLayerIdx]?.[neuronIdx]

          return (
            <g key={`n-${layerIdx}-${neuronIdx}`}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={neuronRadius}
                fill="#fff"
                stroke="#3498db"
                strokeWidth={2}
              />
              {layerIdx > 0 && biasVal !== undefined && (
                <text
                  x={pos.x}
                  y={pos.y}
                  fill="#000"
                  fontSize="10"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {biasVal.toFixed(2)}
                </text>
              )}
            </g>
          )
        })
      )}
    </svg>
  )
}

export default NetworkGraph
