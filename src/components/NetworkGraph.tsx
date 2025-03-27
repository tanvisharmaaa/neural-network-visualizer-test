import React from "react"

interface Props {
  inputNeurons: number
  hiddenLayers: number[]
  outputNeurons: number
  weights: number[][][]
  biases: number[][]
}

const NetworkGraph: React.FC<Props> = ({
  inputNeurons,
  hiddenLayers,
  outputNeurons,
  weights,
  biases,
}) => {
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

  const isTrained = weights.length > 0 && biases.length > 0

  return (
    <svg width="100%" height="500" style={{ backgroundColor: "#1e1e2f" }}>
      <defs>
        <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#00f" stopOpacity="0" />
          <stop offset="50%" stopColor="#00f" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#00f" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Input lines */}
      {positions[0]?.map((neuron, i) => (
        <line
          key={`input-line-${i}`}
          x1={neuron.x - 120}
          y1={neuron.y}
          x2={neuron.x - 10}
          y2={neuron.y}
          stroke="yellow"
          strokeWidth={2}
          strokeDasharray="4"
          opacity={0.7}
        />
      ))}

      {/* Output lines */}
      {positions[positions.length - 1]?.map((neuron, i) => (
        <line
          key={`output-line-${i}`}
          x1={neuron.x + 10}
          y1={neuron.y}
          x2={neuron.x + 120}
          y2={neuron.y}
          stroke="yellow"
          strokeWidth={2}
          strokeDasharray="4"
          opacity={0.7}
        />
      ))}

      {/* Flow lines between neurons */}
      {positions.slice(0, -1).map((fromLayer, i) => {
        const toLayer = positions[i + 1]
        return fromLayer.map((from, fromIdx) =>
          toLayer.map((to, toIdx) => (
            <line
              key={`flow-${i}-${fromIdx}-${toIdx}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke="url(#flowGradient)"
              strokeWidth={1}
              strokeDasharray="5,5"
              opacity={0.3}
            />
          ))
        )
      })}

      {/* Trained weights */}
      {isTrained && weights.map((layerWeights, layerIdx) =>
        layerWeights.map((fromWeights, fromIdx) =>
          fromWeights.map((weight, toIdx) => {
            const from = positions[layerIdx]?.[fromIdx]
            const to = positions[layerIdx + 1]?.[toIdx]
            if (!from || !to) return null

            const isZero = weight === 0
            const opacity = isZero ? 0.3 : Math.min(Math.abs(weight) * 5, 1)
            const color = isZero ? "#aaa" : weight > 0 ? "#3498db" : "#e74c3c"
            


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

      {/* Neurons and biases */}
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
              {isTrained && layerIdx > 0 && biasVal !== undefined && (
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
