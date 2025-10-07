import React, { useState, useEffect } from "react";
import DatasetSelector from "./DatasetSelector";

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
  onPredict: (inputs: number[]) => void;
  onDatasetUpload: (data: {
    inputs: number[][];
    outputs: number[][];
    needsOutputNormalization?: boolean;
  }) => void;
  lineThicknessMode: "auto" | "fixed";
  setLineThicknessMode: (mode: "auto" | "fixed") => void;
  zoomLevel: number;
  setZoomLevel: (level: number) => void;
  onPlay: (epochs: number) => void;
  onPause: () => void;
  isTraining: boolean;
  isPaused: boolean;
  animationSpeed: number;
  setAnimationSpeed: (speed: number) => void;
  learningRate: number;
  setLearningRate: (rate: number) => void;
  hasDataset: boolean;
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
  onPredict,
  onDatasetUpload,
  lineThicknessMode,
  setLineThicknessMode,
  zoomLevel,
  setZoomLevel,
  onPlay,
  onPause,
  isTraining,
  isPaused,
  animationSpeed,
  setAnimationSpeed,
  learningRate,
  setLearningRate,
  hasDataset,
}) => {
  const [testInputs, setTestInputs] = useState<number[]>(
    Array(inputNeurons).fill(0)
  );
  const [epochs, setEpochs] = useState<number>(1);


  useEffect(() => {
    setTestInputs((prev) => {
      const newInputs = Array(inputNeurons).fill(0);
      for (let i = 0; i < Math.min(prev.length, inputNeurons); i++) {
        if (!isNaN(prev[i])) {
          newInputs[i] = prev[i];
        }
      }
      return newInputs;
    });
  }, [inputNeurons]);

  const handleEpochChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setEpochs(!isNaN(val) && val >= 1 ? val : 1);
  };

  const updateLayer = (index: number, value: number) => {
    const updated = [...hiddenLayers];
    updated[index] = Math.max(1, updated[index] + value);
    setHiddenLayers(updated);
  };

  const addLayer = () => {
    setHiddenLayers([...hiddenLayers, 4]);
  };

  const handlePredict = () => {
    if (testInputs.length !== inputNeurons) {
      alert(`Please provide exactly ${inputNeurons} input values.`);
      return;
    }
    if (testInputs.some((val) => isNaN(val))) {
      alert("All test inputs must be valid numbers.");
      return;
    }
    onPredict(testInputs);
  };



  const handleZoomIn = () => setZoomLevel(Math.min(zoomLevel + 0.1, 2));
  const handleZoomOut = () => setZoomLevel(Math.max(zoomLevel - 0.1, 0.5));


  return (
    <div style={{ padding: "20px", border: "1px solid #e0e0e0", borderRadius: "8px", backgroundColor: "#fafafa", fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif", boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}>
      <DatasetSelector
        onDatasetLoad={onDatasetUpload}
        onInputNeuronsChange={setInputNeurons}
        onOutputNeuronsChange={setOutputNeurons}
        onProblemTypeChange={setProblemType}
        problemType={problemType}
      />
      
      {hasDataset && (
        <>
          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>Activation Function</label>
        <select
          value={activationFunction}
          onChange={(e) => setActivationFunction(e.target.value)}
          style={{ 
            width: "100%", 
            padding: "10px 12px", 
            border: "1px solid #e0e0e0", 
            borderRadius: "4px", 
            fontSize: "14px",
            fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
            backgroundColor: "#fff",
            color: "#424242"
          }}
        >
          <option value="sigmoid">sigmoid</option>
          <option value="relu">relu</option>
          <option value="tanh">tanh</option>
        </select>
      </div>

      <div style={{ marginBottom: "15px" }}>
        <label style={{ display: "block", marginBottom: "8px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>Problem Type</label>
        <select
          value={problemType}
          onChange={(e) => setProblemType(e.target.value)}
          style={{ 
            width: "100%", 
            padding: "10px 12px", 
            border: "1px solid #e0e0e0", 
            borderRadius: "4px", 
            fontSize: "14px",
            fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
            backgroundColor: "#fff",
            color: "#424242"
          }}
        >
          <option value="Classification">Classification</option>
          <option value="Regression">Regression</option>
        </select>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Input Neurons</label>
        <input
          type="number"
          value={inputNeurons}
          onChange={(e) => {
            const val = parseInt(e.target.value);
            if (!isNaN(val) && val >= 1) setInputNeurons(val);
          }}
          min="1"
          readOnly={hasDataset}
        />
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Output Neurons</label>
        <input
          type="number"
          value={outputNeurons}
          onChange={(e) => {
            const val = parseInt(e.target.value);
            if (!isNaN(val) && val >= 1) setOutputNeurons(val);
          }}
          min="1"
        />
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Learning Rate</label>
        <input
          type="number"
          step="0.01"
          min="0.001"
          max="1"
          value={learningRate}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            if (!isNaN(val) && val >= 0.001 && val <= 1) setLearningRate(val);
          }}
        />
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Epochs</label>
        <input
          type="number"
          value={epochs}
          min={1}
          onChange={handleEpochChange}
        />
        <span style={{ marginLeft: "10px" }}>
          (Training for {epochs} epoch{epochs !== 1 ? "s" : ""})
        </span>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Test Inputs</label>
        {testInputs.map((val, idx) => (
          <input
            key={idx}
            type="number"
            step="0.1"
            value={val}
            onChange={(e) => {
              const newInputs = [...testInputs];
              const val = parseFloat(e.target.value);
              newInputs[idx] = isNaN(val) ? 0 : val;
              setTestInputs(newInputs);
            }}
            style={{ width: "60px", marginRight: "5px" }}
          />
        ))}
        <button onClick={handlePredict}>Predict</button>
      </div>


      <div style={{ marginBottom: 10 }}>
        <label>Hidden Layers</label>
        <button onClick={addLayer} style={{ marginLeft: 10 }}>
          + Add Layer
        </button>
      </div>

      {hiddenLayers.map((neurons, idx) => (
        <div key={idx} style={{ marginBottom: 5 }}>
          <span>
            Layer {idx + 1}: {neurons} neurons{" "}
          </span>
          <button onClick={() => updateLayer(idx, -1)}>-</button>
          <button onClick={() => updateLayer(idx, 1)}>+</button>
        </div>
      ))}

      <div style={{ marginBottom: "15px" }}>
        <label style={{ display: "flex", alignItems: "center", cursor: "pointer", fontSize: "14px", color: "#424242" }}>
          <input
            type="checkbox"
            checked={showWeights}
            onChange={(e) => setShowWeights(e.target.checked)}
            style={{ marginRight: "8px" }}
          />
          Show Weights on Edges
        </label>
      </div>

      <div style={{ marginBottom: "15px" }}>
        <label style={{ display: "block", marginBottom: "8px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>Line Thickness</label>
        <select
          value={lineThicknessMode}
          onChange={(e) =>
            setLineThicknessMode(e.target.value as "auto" | "fixed")
          }
          style={{ 
            width: "100%", 
            padding: "10px 12px", 
            border: "1px solid #e0e0e0", 
            borderRadius: "4px", 
            fontSize: "14px",
            fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
            backgroundColor: "#fff",
            color: "#424242"
          }}
        >
          <option value="auto">Weight-based</option>
          <option value="fixed">Fixed</option>
        </select>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Zoom</label>
        <button onClick={handleZoomIn} style={{ marginLeft: 10 }}>
          +
        </button>
        <button onClick={handleZoomOut} style={{ marginLeft: 5 }}>
          -
        </button>
        <span style={{ marginLeft: 10 }}>{(zoomLevel * 100).toFixed(0)}%</span>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Animation Speed</label>
        <input
          type="range"
          min="0.1"
          max="5"
          step="0.1"
          value={animationSpeed}
          onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
        />
        <span>{animationSpeed.toFixed(1)}x</span>
      </div>

      <div>
        <button
          onClick={() => onPlay(epochs)}
          disabled={isTraining && !isPaused}
        >
          Play
        </button>
        <button onClick={onPause} disabled={!isTraining || isPaused}>
          Pause
        </button>
      </div>
        </>
      )}
    </div>
  );
};

export default NetworkControls;