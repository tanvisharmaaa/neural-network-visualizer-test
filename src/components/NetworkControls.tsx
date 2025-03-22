import React from "react"

interface Props {
  hiddenLayers: number[]
  setHiddenLayers: (layers: number[]) => void
  inputNeurons: number
  setInputNeurons: (value: number) => void
  outputNeurons: number
  setOutputNeurons: (value: number) => void
  activationFunction: string
  setActivationFunction: (fn: string) => void
  problemType: string
  setProblemType: (type: string) => void
}

const activationOptions = ["sigmoid", "relu", "tanh", "linear"]
const problemTypes = ["Classification", "Regression"]

const NetworkControls: React.FC<Props> = ({
  hiddenLayers,
  setHiddenLayers,
  inputNeurons,
  setInputNeurons,
  outputNeurons,
  setOutputNeurons,
  activationFunction,
  setActivationFunction,
  problemType,
  setProblemType,
}) => {
  const addLayer = () => {
    setHiddenLayers([...hiddenLayers, 3])
  }

  const updateNeuronCount = (index: number, delta: number) => {
    const updated = [...hiddenLayers]
    updated[index] = Math.max(1, updated[index] + delta)
    setHiddenLayers(updated)
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div>
        <label>Activation Function</label>
        <select
          value={activationFunction}
          onChange={(e) => setActivationFunction(e.target.value)}
          style={styles.input}
        >
          {activationOptions.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      </div>

      <div>
        <label>Problem Type</label>
        <select
          value={problemType}
          onChange={(e) => setProblemType(e.target.value)}
          style={styles.input}
        >
          {problemTypes.map((type) => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </div>

      <div>
        <label>Input Neurons</label>
        <input
          type="number"
          value={inputNeurons}
          min={1}
          max={10}
          onChange={(e) => setInputNeurons(+e.target.value)}
          style={styles.input}
        />
      </div>

      <div>
        <label>Output Neurons</label>
        <input
          type="number"
          value={outputNeurons}
          min={1}
          max={5}
          onChange={(e) => setOutputNeurons(+e.target.value)}
          style={styles.input}
        />
      </div>

      <div>
        <label>Hidden Layers</label>
        <button onClick={addLayer} style={styles.addBtn}>+ Add Layer</button>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", marginTop: "8px" }}>
          {hiddenLayers.map((neurons, index) => (
            <div key={index} style={styles.layerCard}>
              <strong>Layer {index + 1}</strong>
              <div>{neurons} neurons</div>
              <div>
                <button onClick={() => updateNeuronCount(index, -1)} style={styles.layerBtn}>–</button>
                <button onClick={() => updateNeuronCount(index, 1)} style={styles.layerBtn}>+</button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

const styles: { [key: string]: React.CSSProperties } = {
  input: {
    width: "100%",
    padding: "6px 10px",
    borderRadius: 4,
    border: "1px solid #ccc",
    backgroundColor: "#121225",
    color: "white",
    marginTop: 4,
  },
  addBtn: {
    marginLeft: "10px",
    padding: "4px 10px",
    borderRadius: 4,
    cursor: "pointer",
  },
  layerCard: {
    backgroundColor: "#2c2c3e",
    padding: "10px",
    borderRadius: 6,
    minWidth: 100,
    textAlign: "center",
  },
  layerBtn: {
    margin: "0 4px",
    padding: "2px 6px",
    fontWeight: "bold",
    cursor: "pointer",
  },
}

export default NetworkControls
