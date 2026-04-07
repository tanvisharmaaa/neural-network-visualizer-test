import React, { useState, useEffect } from "react";

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
  lineThicknessMode: "auto" | "fixed";
  setLineThicknessMode: (mode: "auto" | "fixed") => void;
  zoomLevel: number;
  setZoomLevel: (level: number) => void;
  animationSpeed: number;
  setAnimationSpeed: (speed: number) => void;
  learningRate: number;
  setLearningRate: (rate: number) => void;
  hasDataset: boolean;
  epochs: number;
  setEpochs: (epochs: number) => void;
}

const selectStyle: React.CSSProperties = {
  width: "100%",
  padding: "10px 12px",
  border: "1px solid #e0e0e0",
  borderRadius: "4px",
  fontSize: "14px",
  fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
  backgroundColor: "#fff",
  color: "#424242",
};

const labelBlock: React.CSSProperties = {
  display: "block",
  marginBottom: "8px",
  fontWeight: "500",
  fontSize: "14px",
  color: "#424242",
};

const columnTitle: React.CSSProperties = {
  fontSize: "12px",
  fontWeight: "600",
  color: "#757575",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  marginBottom: "4px",
};

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
  lineThicknessMode,
  setLineThicknessMode,
  zoomLevel,
  setZoomLevel,
  animationSpeed,
  setAnimationSpeed,
  learningRate,
  setLearningRate,
  hasDataset,
  epochs,
  setEpochs,
}) => {
  const [testInputs, setTestInputs] = useState<number[]>(
    Array(inputNeurons).fill(0)
  );

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
  const handleZoomOut = () => setZoomLevel(Math.max(zoomLevel - 0.1, 0.1));

  return (
    <div>
      {hasDataset && (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
              gap: "24px 28px",
              alignItems: "start",
            }}
          >
            {/* Column 1: training & size */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "15px",
                minWidth: 0,
              }}
            >
              <div style={columnTitle}>Training</div>
              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Input Neurons</label>
                <input
                  type="number"
                  value={inputNeurons}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val) && val >= 1) setInputNeurons(val);
                  }}
                  min="1"
                  readOnly={hasDataset}
                  style={{
                    width: "100%",
                    padding: "8px 10px",
                    border: "1px solid #e0e0e0",
                    borderRadius: "4px",
                    fontSize: "14px",
                    boxSizing: "border-box",
                  }}
                />
              </div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Output Neurons</label>
                <input
                  type="number"
                  value={outputNeurons}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val) && val >= 1) setOutputNeurons(val);
                  }}
                  min="1"
                  style={{
                    width: "100%",
                    padding: "8px 10px",
                    border: "1px solid #e0e0e0",
                    borderRadius: "4px",
                    fontSize: "14px",
                    boxSizing: "border-box",
                  }}
                />
              </div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Learning Rate</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.001"
                  max="1"
                  value={learningRate}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val) && val >= 0.001 && val <= 1)
                      setLearningRate(val);
                  }}
                  style={{
                    width: "100%",
                    padding: "8px 10px",
                    border: "1px solid #e0e0e0",
                    borderRadius: "4px",
                    fontSize: "14px",
                    boxSizing: "border-box",
                  }}
                />
              </div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Epochs</label>
                <input
                  type="number"
                  value={epochs}
                  min={1}
                  onChange={handleEpochChange}
                  style={{
                    width: "100%",
                    padding: "8px 10px",
                    border: "1px solid #e0e0e0",
                    borderRadius: "4px",
                    fontSize: "14px",
                    boxSizing: "border-box",
                  }}
                />
                <span
                  style={{
                    display: "block",
                    marginTop: "6px",
                    fontSize: "12px",
                    color: "#757575",
                  }}
                >
                  Training for {epochs} epoch{epochs !== 1 ? "s" : ""}
                </span>
              </div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Zoom</label>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    flexWrap: "wrap",
                  }}
                >
                  <button type="button" onClick={handleZoomIn}>
                    +
                  </button>
                  <button type="button" onClick={handleZoomOut}>
                    -
                  </button>
                  <span style={{ fontSize: "14px", color: "#424242" }}>
                    {(zoomLevel * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Animation Speed</label>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "12px",
                    flexWrap: "wrap",
                  }}
                >
                  <input
                    type="range"
                    min="0.1"
                    max="5"
                    step="0.1"
                    value={animationSpeed}
                    onChange={(e) =>
                      setAnimationSpeed(parseFloat(e.target.value))
                    }
                    style={{ flex: "1 1 120px", minWidth: "100px" }}
                  />
                  <span style={{ fontSize: "14px", color: "#424242" }}>
                    {animationSpeed.toFixed(1)}x
                  </span>
                </div>
              </div>
            </div>

            {/* Column 2: inference, architecture, display */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "15px",
                minWidth: 0,
              }}
            >
              <div style={columnTitle}>Network & view</div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Activation Function</label>
                <select
                  value={activationFunction}
                  onChange={(e) => setActivationFunction(e.target.value)}
                  style={selectStyle}
                >
                  <option value="sigmoid">sigmoid</option>
                  <option value="relu">relu</option>
                  <option value="tanh">tanh</option>
                </select>
              </div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Problem Type</label>
                <select
                  value={problemType}
                  onChange={(e) => setProblemType(e.target.value)}
                  style={selectStyle}
                >
                  <option value="Classification">Classification</option>
                  <option value="Regression">Regression</option>
                </select>
              </div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Test Inputs</label>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "8px",
                    alignItems: "center",
                  }}
                >
                  {testInputs.map((val, idx) => (
                    <input
                      key={idx}
                      type="number"
                      step="0.1"
                      value={val}
                      onChange={(e) => {
                        const newInputs = [...testInputs];
                        const v = parseFloat(e.target.value);
                        newInputs[idx] = isNaN(v) ? 0 : v;
                        setTestInputs(newInputs);
                      }}
                      style={{
                        width: "72px",
                        padding: "6px 8px",
                        border: "1px solid #e0e0e0",
                        borderRadius: "4px",
                        fontSize: "14px",
                        boxSizing: "border-box",
                      }}
                    />
                  ))}
                  <button type="button" onClick={handlePredict}>
                    Predict
                  </button>
                </div>
              </div>

              <div style={{ marginBottom: 0 }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                    gap: "8px",
                    marginBottom: "8px",
                  }}
                >
                  <label style={{ ...labelBlock, marginBottom: 0 }}>
                    Hidden Layers
                  </label>
                  <button type="button" onClick={addLayer}>
                    + Add Layer
                  </button>
                </div>
                {hiddenLayers.map((neurons, idx) => (
                  <div
                    key={idx}
                    style={{
                      marginBottom: 6,
                      display: "flex",
                      alignItems: "center",
                      flexWrap: "wrap",
                      gap: "8px",
                    }}
                  >
                    <span style={{ fontSize: "14px", color: "#424242" }}>
                      Layer {idx + 1}: {neurons} neurons
                    </span>
                    <button type="button" onClick={() => updateLayer(idx, -1)}>
                      -
                    </button>
                    <button type="button" onClick={() => updateLayer(idx, 1)}>
                      +
                    </button>
                  </div>
                ))}
              </div>

              <div style={{ marginBottom: 0 }}>
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    cursor: "pointer",
                    fontSize: "14px",
                    color: "#424242",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={showWeights}
                    onChange={(e) => setShowWeights(e.target.checked)}
                    style={{ marginRight: "8px" }}
                  />
                  Show Weights on Edges
                </label>
              </div>

              <div style={{ marginBottom: 0 }}>
                <label style={labelBlock}>Line Thickness</label>
                <select
                  value={lineThicknessMode}
                  onChange={(e) =>
                    setLineThicknessMode(e.target.value as "auto" | "fixed")
                  }
                  style={selectStyle}
                >
                  <option value="auto">Weight-based</option>
                  <option value="fixed">Fixed</option>
                </select>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default NetworkControls;
