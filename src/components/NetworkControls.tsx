// NetworkControls.tsx
import React from "react";

interface Props {
  hiddenLayers: number[];
  setHiddenLayers: (layers: number[]) => void;
  inputNeurons: number;
  setInputNeurons: (val: number) => void;
  outputNeurons: number;
  setOutputNeurons: (val: number) => void;
  activationFunction: string;
  setActivationFunction: (val: string) => void;
  problemType: string;
  setProblemType: (val: string) => void;
  showWeights: boolean;
  setShowWeights: (val: boolean) => void;
}

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
  showWeights,
  setShowWeights,
}) => {
  const updateLayer = (index: number, value: number) => {
    const updated = [...hiddenLayers];
    updated[index] = Math.max(1, updated[index] + value);
    setHiddenLayers(updated);
  };

  const addLayer = () => {
    setHiddenLayers([...hiddenLayers, 2]);
  };

  return (
    <div className="panel">
      <div>
        <label>Activation Function</label>
        <select value={activationFunction} onChange={(e) => setActivationFunction(e.target.value)}>
          <option value="sigmoid">sigmoid</option>
          <option value="relu">relu</option>
          <option value="tanh">tanh</option>
        </select>
      </div>

      <div>
        <label>Problem Type</label>
        <select value={problemType} onChange={(e) => setProblemType(e.target.value)}>
          <option value="Classification">Classification</option>
          <option value="Regression">Regression</option>
        </select>
      </div>

      <div>
        <label>Input Neurons</label>
        <input
          type="number"
          value={inputNeurons}
          onChange={(e) => setInputNeurons(Math.max(1, parseInt(e.target.value)))}
        />
      </div>

      <div>
        <label>Output Neurons</label>
        <input
          type="number"
          value={outputNeurons}
          onChange={(e) => setOutputNeurons(Math.max(1, parseInt(e.target.value)))}
        />
      </div>

      <div>
        <label>Epochs</label>
        <input
          type="number"
          defaultValue={5}
          min={1}
          onChange={(e) => {
            const val = Math.max(1, parseInt(e.target.value));
            localStorage.setItem("numEpochs", val.toString());
          }}
        />
      </div>

      <div className="mt-2">
        <label>Hidden Layers</label>
        <button onClick={addLayer} style={{ marginLeft: 10 }}>+ Add Layer</button>
      </div>

      {hiddenLayers.map((neurons, idx) => (
        <div key={idx}>
          <span>Layer {idx + 1}: {neurons} neurons </span>
          <button onClick={() => updateLayer(idx, -1)}>-</button>
          <button onClick={() => updateLayer(idx, 1)}>+</button>
        </div>
      ))}

      <div style={{ marginTop: 10 }}>
        <label>
          <input
            type="checkbox"
            checked={showWeights}
            onChange={(e) => setShowWeights(e.target.checked)}
          />
          Show Weights on Edges
        </label>
      </div>
    </div>
  );
};

export default NetworkControls;
