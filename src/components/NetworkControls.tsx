
import React, { useState, useEffect } from "react";
import Papa from "papaparse";

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
  onDatasetUpload: (data: { inputs: number[][]; outputs: number[][] }) => void;
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
}) => {
  const [testInputs, setTestInputs] = useState<number[]>(Array(inputNeurons).fill(0));
  const [epochs, setEpochs] = useState<number>(1);

  useEffect(() => {
    setTestInputs(Array(inputNeurons).fill(0));
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
    if (testInputs.some(isNaN)) {
      alert("All test inputs must be valid numbers.");
      return;
    }
    onPredict(testInputs);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        if (fileExtension === 'json') {
          const data = JSON.parse((event.target as FileReader).result as string);
          if (data.inputs && data.outputs && Array.isArray(data.inputs) && Array.isArray(data.outputs)) {
            setInputNeurons(data.inputs[0].length);
            setOutputNeurons(data.outputs[0].length);
            onDatasetUpload(data);
          } else {
            alert("Invalid JSON dataset format.");
          }
        } else if (fileExtension === 'csv') {
          Papa.parse((event.target as FileReader).result as string, {
            complete: (result) => {
              const data = result.data as any[];
              if (data.length < 1) {
                alert("CSV file is empty or invalid.");
                return;
              }

              const headers = Object.keys(data[0] || {});
              if (headers.length < 2) {
                alert("CSV must have at least 2 columns (1 input and 1 output).");
                return;
              }

              const rows = data.filter(row => row && Object.keys(row).length === headers.length);
              if (rows.length === 0) {
                alert("CSV file contains no valid rows.");
                return;
              }

              const inputs: number[][] = [];
              let outputs: number[][] = [];

              let assumedOutputNeurons = 1;
              let isBinary = false;
              const lastColValues = rows.map(row => row[headers[headers.length - 1]]?.toString().toLowerCase());
              const uniqueValues = [...new Set(lastColValues)];

              if (problemType === "Classification" && uniqueValues.length > 0) {
                isBinary = uniqueValues.length <= 2 && uniqueValues.every(val =>
                  ['0', '1', 'm', 'b', 'true', 'false'].includes(val) || !isNaN(parseFloat(val))
                );
                assumedOutputNeurons = isBinary ? 1 : uniqueValues.length;
              } else if (problemType === "Regression") {
                assumedOutputNeurons = 1;
              }

              const inputCols = headers.slice(0, headers.length - assumedOutputNeurons);
              const outputCols = headers.slice(headers.length - assumedOutputNeurons);

              const rawOutputs: number[][] = [];
              rows.forEach(row => {
                const input = inputCols.map(col => parseFloat(row[col]) || 0);
                const output = outputCols.map(col => {
                  const val = row[col]?.toString().toLowerCase();
                  if (problemType === "Classification") {
                    if (assumedOutputNeurons === 1) {
                      return val === 'm' || val === '1' || val === 'true' ? 1 : 0;
                    } else {
                      return uniqueValues.indexOf(val);
                    }
                  }
                  const numVal = parseFloat(val);
                  return isNaN(numVal) ? 0 : numVal;
                });
                if (input.every(val => !isNaN(val)) && output.every(val => !isNaN(val))) {
                  inputs.push(input);
                  rawOutputs.push(output);
                }
              });

              if (problemType === "Classification" && assumedOutputNeurons > 1) {
                outputs = rawOutputs.map(row => {
                  const labelIndex = row[0];
                  const oneHot = Array(assumedOutputNeurons).fill(0);
                  oneHot[labelIndex] = 1;
                  return oneHot;
                });
              } else {
                outputs = rawOutputs;
              }

              if (inputs.length > 0 && outputs.length > 0) {
                setInputNeurons(inputs[0].length);
                setOutputNeurons(outputs[0].length);
                onDatasetUpload({ inputs, outputs });
              } else {
                alert("Parsed dataset has invalid input or output data.");
              }
            },
            header: true,
            skipEmptyLines: true,
          });
        } else {
          alert("Unsupported file type. Use JSON or CSV.");
        }
      } catch (error) {
        alert("Error parsing dataset: " + error);
      }
    };

    reader.readAsText(file);
  };

  const handleZoomIn = () => setZoomLevel(Math.min(zoomLevel + 0.1, 2));
  const handleZoomOut = () => setZoomLevel(Math.max(zoomLevel - 0.1, 0.5));

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
        <input type="number" value={inputNeurons} readOnly />
      </div>

      <div>
        <label>Output Neurons</label>
        <input
          type="number"
          value={outputNeurons}
          onChange={(e) => setOutputNeurons(Math.max(1, parseInt(e.target.value)))}
          min="1"
        />
      </div>

      <div>
        <label>Learning Rate</label>
        <input
          type="number"
          step="0.01"
          min="0.001"
          max="1"
          defaultValue="0.1"
          onChange={(e) => localStorage.setItem("learningRate", Math.max(0.001, parseFloat(e.target.value)).toString())}
        />
      </div>

      <div>
        <label>Epochs</label>
        <input type="number" value={epochs} min={1} onChange={handleEpochChange} />
        <span style={{ marginLeft: "10px" }}>
          (Training for {epochs} epoch{epochs !== 1 ? "s" : ""})
        </span>
      </div>

      <div>
        <label>Test Inputs</label>
        {testInputs.map((val, idx) => (
          <input
            key={idx}
            type="number"
            step="0.1"
            value={val}
            onChange={(e) => {
              const newInputs = [...testInputs];
              newInputs[idx] = parseFloat(e.target.value);
              setTestInputs(newInputs);
            }}
            style={{ width: "60px", marginRight: "5px" }}
          />
        ))}
        <button onClick={handlePredict}>Predict</button>
      </div>

      <div>
        <label>Upload Dataset (JSON or CSV)</label>
        <input type="file" accept=".json,.csv" onChange={handleFileUpload} />
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

      <div style={{ marginTop: 10 }}>
        <label>Line Thickness</label>
        <select value={lineThicknessMode} onChange={(e) => setLineThicknessMode(e.target.value as "auto" | "fixed")}>
          <option value="auto">Weight-based</option>
          <option value="fixed">Fixed</option>
        </select>
      </div>

      <div style={{ marginTop: 10 }}>
        <label>Zoom</label>
        <button onClick={handleZoomIn} style={{ marginLeft: 10 }}>+</button>
        <button onClick={handleZoomOut} style={{ marginLeft: 5 }}>-</button>
        <span style={{ marginLeft: 10 }}>{(zoomLevel * 100).toFixed(0)}%</span>
      </div>

      <div style={{ marginTop: 10 }}>
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

      <div style={{ marginTop: "20px" }}>
        <button onClick={() => onPlay(epochs)} disabled={isTraining && !isPaused}>
          Play
        </button>
        <button onClick={onPause} disabled={!isTraining || isPaused}>
          Pause
        </button>
      </div>
    </div>
  );
};

export default NetworkControls;