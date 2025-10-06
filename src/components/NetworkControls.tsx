
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
  onDatasetUpload: (data: { inputs: number[][]; outputs: number[][]; needsOutputNormalization?: boolean }) => void;
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
  const [testInputs, setTestInputs] = useState<number[]>(Array(inputNeurons).fill(0));
  const [epochs, setEpochs] = useState<number>(1);
  const [labelColumn, setLabelColumn] = useState<string>("");
  const [outputColumns, setOutputColumns] = useState<string[]>([]);
  const [inputColumns, setInputColumns] = useState<string[]>([]);
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [fileContent, setFileContent] = useState<string | null>(null);


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


  const classifyColumns = (rows: any[], columns: string[]) => {
    const columnTypes = new Map<string, 'numeric' | 'categorical'>();
    const columnMeans = new Map<string, number>();
    const columnUniques = new Map<string, Set<string>>();

    columns.forEach((col) => {
      const values = rows.map((row) => row[col]?.toString().trim() || '');
      const nonMissingValues = values.filter((val) => val && val.toUpperCase() !== 'NA');
      const isNumeric = nonMissingValues.length > 0 && nonMissingValues.every((val) => !isNaN(parseFloat(val)));

      columnTypes.set(col, isNumeric ? 'numeric' : 'categorical');

      if (isNumeric) {
        const numValues = nonMissingValues.map((val) => parseFloat(val));
        const mean = numValues.length > 0 ? numValues.reduce((a, b) => a + b, 0) / numValues.length : 0;
        columnMeans.set(col, mean);
      } else {
        const uniques = new Set(nonMissingValues.map((val) => val.toLowerCase()));
        columnUniques.set(col, uniques);
        if (uniques.size > 10) {
          console.warn(`Column ${col} has too many unique values (${uniques.size}). Consider preprocessing.`);
        }
      }
    });

    return { columnTypes, columnMeans, columnUniques };
  };


  const processRow = (row: any, inputCols: string[], outputCols: string[], problemType: string, columnTypes: Map<string, 'numeric' | 'categorical'>, columnMeans: Map<string, number>, columnUniques: Map<string, Set<string>>, labelToIndex: Map<string, number>) => {
    const inputValues: number[] = [];
    for (const col of inputCols) {
      let val = row[col]?.toString().trim() || '';
      if (val.toUpperCase() === 'NA' || val === '') {
        if (columnTypes.get(col) === 'numeric') {
          val = columnMeans.get(col)!.toString(); 
        } else {
          val = 'missing'; 
        }
      }
      if (columnTypes.get(col) === 'numeric') {
        const numVal = parseFloat(val);
        if (isNaN(numVal)) return null;
        inputValues.push(numVal);
      } else {
        const uniques = Array.from(columnUniques.get(col)!);
        const oneHot = uniques.map((u) => (val.toLowerCase() === u ? 1 : 0));
        inputValues.push(...oneHot);
      }
    }

    const outputValues: number[] = [];
    for (const col of outputCols) {
      let val = row[col]?.toString().trim() || '';
      if (val.toUpperCase() === 'NA' || val === '') {
        if (problemType === "Regression") {
          val = columnMeans.get(col)!.toString();
        } else {
          return null; 
        }
      }
      if (problemType === "Regression") {
        const numVal = parseFloat(val);
        if (isNaN(numVal)) return null;
        outputValues.push(numVal);
      } else {
        const label = val.toLowerCase();
        if (!label || label.toUpperCase() === 'NA') return null;
        const index = labelToIndex.get(label);
        if (index === undefined) return null;
        outputValues.push(index);
      }
    }

    if (inputValues.every((val) => !isNaN(val)) && outputValues.every((val) => !isNaN(val))) {
      return { input: inputValues, output: outputValues };
    }
    return null;
  };

  const processCSV = () => {
    if (!fileContent) {
      alert("No file content available to process.");
      return;
    }

    Papa.parse(fileContent, {
      complete: (result) => {
        const data = result.data as any[];
        if (data.length < 1) {
          alert("CSV file is empty or invalid.");
          return;
        }

        const headers = Object.keys(data[0] || {}).filter((h) => h && h !== '');
        if (headers.length < 2) {
          alert("Invalid headers: CSV must have at least 2 unique, non-empty column names.");
          return;
        }

        const uniqueHeaders = [...new Set(headers)];
        if (uniqueHeaders.length !== headers.length) {
          alert("Invalid headers: CSV contains duplicate column names.");
          return;
        }

        const rows = data.filter((row) => row && Object.keys(row).length === headers.length);
        if (rows.length === 0) {
          alert("CSV file contains no valid rows matching header count.");
          return;
        }

        let inputCols: string[] = inputColumns;
        let outputCols: string[] = [];
        let assumedOutputNeurons = outputNeurons;

        if (problemType === "Classification") {
          if (!labelColumn || !headers.includes(labelColumn)) {
            alert("Please select a valid label column for classification.");
            return;
          }
          outputCols = [labelColumn];
          const uniqueValues = [...new Set(
            rows.map((row) => row[labelColumn]?.toString().trim().toLowerCase() || "")
              .filter((val) => val && val.toUpperCase() !== 'NA')
          )];
          console.log("Unique labels:", uniqueValues); 
          if (uniqueValues.length === 0) {
            alert("No valid labels found in the selected label column.");
            return;
          }
          if (uniqueValues.length > 10) {
            alert(`The selected label column has too many unique values (${uniqueValues.length}), which may indicate it's not a categorical label. Please select a different column.`);
            return;
          }
          const isBinary = uniqueValues.length <= 2;
          assumedOutputNeurons = isBinary ? 1 : uniqueValues.length;
          if (assumedOutputNeurons !== outputNeurons) {
            alert(`Classification requires ${assumedOutputNeurons} output neurons for ${uniqueValues.length} unique labels. Adjusting automatically.`);
            setOutputNeurons(assumedOutputNeurons);
          }
        } else {
          if (outputColumns.length === 0) {
            alert("Please select at least one output column for regression.");
            return;
          }
          outputCols = outputColumns;
          assumedOutputNeurons = outputColumns.length;
          if (assumedOutputNeurons !== outputNeurons) {
            alert(`Regression requires ${assumedOutputNeurons} output neurons for selected columns. Adjusting automatically.`);
            setOutputNeurons(assumedOutputNeurons);
          }
        }

        if (inputCols.length === 0) {
          alert("Please select at least one input column.");
          return;
        }

        
        const classificationResult = classifyColumns(rows, [...inputCols, ...outputCols]);
        if (!classificationResult) return;
        const { columnTypes, columnMeans, columnUniques } = classificationResult;

        const inputs: number[][] = [];
        let rawOutputs: number[][] = [];
        const labelToIndex = new Map<string, number>();
        if (problemType === "Classification") {
          const uniqueValues = [...new Set(
            rows.map((row) => row[labelColumn]?.toString().trim().toLowerCase() || "")
              .filter((val) => val && val.toUpperCase() !== 'NA')
          )];
          uniqueValues.sort();
          uniqueValues.forEach((val, idx) => labelToIndex.set(val, idx));
        }

        rows.forEach((row) => {
          const processed = processRow(row, inputCols, outputCols, problemType, columnTypes, columnMeans, columnUniques, labelToIndex);
          if (processed) {
            inputs.push(processed.input);
            rawOutputs.push(processed.output);
          }
        });

        let outputs: number[][] = rawOutputs;
        if (problemType === "Classification" && assumedOutputNeurons > 1) {
          outputs = rawOutputs.map((row) => {
            const labelIndex = row[0];
            const oneHot = Array(assumedOutputNeurons).fill(0);
            oneHot[labelIndex] = 1;
            return oneHot;
          });
        }

        if (inputs.length > 0 && outputs.length > 0) {
          console.log(`Input neurons: ${inputs[0].length} from columns: ${inputCols.join(', ')}`);
          setInputNeurons(inputs[0].length);
          onDatasetUpload({ inputs, outputs, needsOutputNormalization: problemType === "Regression" });
        } else {
          alert("Parsed dataset has no valid rows after filtering invalid or missing data. Check if selected columns have valid data.");
        }
      },
      header: true,
      skipEmptyLines: 'greedy',
      dynamicTyping: false,
    });
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        if (fileExtension === 'json') {
          const data = JSON.parse(event.target?.result as string);
          if (!validateJSONDataset(data)) {
            alert("Invalid JSON dataset: Must contain 'inputs' and 'outputs' arrays with consistent numeric values.");
            return;
          }
          setInputNeurons(data.inputs[0].length);
          setOutputNeurons(data.outputs[0].length);
          setAvailableColumns([]);
          setLabelColumn("");
          setOutputColumns([]);
          setInputColumns([]);
          setFileContent(null);
          onDatasetUpload({ ...data, needsOutputNormalization: problemType === "Regression" });
        } else if (fileExtension === 'csv') {
          const content = event.target?.result as string;
          setFileContent(content);
          Papa.parse(content, {
            complete: (result) => {
              const data = result.data as any[];
              if (data.length < 1) {
                alert("CSV file is empty or invalid.");
                return;
              }

              const headers = Object.keys(data[0] || {}).filter((h) => h && h !== '');
              if (headers.length < 2) {
                alert("Invalid headers: CSV must have at least 2 unique, non-empty column names.");
                return;
              }

              const uniqueHeaders = [...new Set(headers)];
              if (uniqueHeaders.length !== headers.length) {
                alert("Invalid headers: CSV contains duplicate column names.");
                return;
              }

              setAvailableColumns(headers);
              if (problemType === "Classification") {
                setLabelColumn(headers[headers.length - 1]);
                setInputColumns(headers.slice(0, -1));
              } else {
                setOutputColumns([headers[headers.length - 1]]);
                setInputColumns(headers.slice(0, -1));
              }
            },
            header: true,
            skipEmptyLines: 'greedy',
            dynamicTyping: false,
          });
        } else {
          alert("Unsupported file type. Use JSON or CSV.");
        }
      } catch (error) {
        alert(`Error parsing dataset: ${error}`);
      }
    };

    reader.readAsText(file);
  };

  
  const validateJSONDataset = (data: any): data is { inputs: number[][]; outputs: number[][] } => {
    if (!data.inputs || !data.outputs || !Array.isArray(data.inputs) || !Array.isArray(data.outputs)) {
      return false;
    }
    if (data.inputs.length === 0 || data.outputs.length === 0 || data.inputs.length !== data.outputs.length) {
      return false;
    }
    const inputLength = data.inputs[0].length;
    const outputLength = data.outputs[0].length;
    return (
      data.inputs.every((row: any) => Array.isArray(row) && row.length === inputLength && row.every((val: any) => typeof val === 'number' && !isNaN(val))) &&
      data.outputs.every((row: any) => Array.isArray(row) && row.length === outputLength && row.every((val: any) => typeof val === 'number' && !isNaN(val)))
    );
  };

  const handleZoomIn = () => setZoomLevel(Math.min(zoomLevel + 0.1, 2));
  const handleZoomOut = () => setZoomLevel(Math.max(zoomLevel - 0.1, 0.5));

  const handleLoadDataset = () => {
    processCSV();
  };

  return (
    <div style={{ padding: 10, border: "1px solid #ccc", borderRadius: 4 }}>
      <div style={{ marginBottom: 10 }}>
        <label>Activation Function</label>
        <select value={activationFunction} onChange={(e) => setActivationFunction(e.target.value)}>
          <option value="sigmoid">sigmoid</option>
          <option value="relu">relu</option>
          <option value="tanh">tanh</option>
        </select>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Problem Type</label>
        <select value={problemType} onChange={(e) => setProblemType(e.target.value)}>
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
        <label>Upload Dataset (JSON or CSV)</label>
        <input type="file" accept=".json,.csv" onChange={handleFileUpload} />
      </div>

      {availableColumns.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <label>Select Input Columns</label>
          <select
            multiple
            value={inputColumns}
            onChange={(e) => {
              const selected = Array.from(e.target.selectedOptions).map((option) => option.value);
              setInputColumns(selected);
            }}
            style={{ width: "200px", height: "100px" }}
          >
            {availableColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      )}

      {availableColumns.length > 0 && problemType === "Classification" && (
        <div style={{ marginBottom: 10 }}>
          <label>Select Label Column</label>
          <select
            value={labelColumn}
            onChange={(e) => setLabelColumn(e.target.value)}
          >
            <option value="">Select a column</option>
            {availableColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      )}

      {availableColumns.length > 0 && problemType === "Regression" && (
        <div style={{ marginBottom: 10 }}>
          <label>Select Output Columns</label>
          <select
            multiple
            value={outputColumns}
            onChange={(e) => {
              const selected = Array.from(e.target.selectedOptions).map((option) => option.value);
              setOutputColumns(selected);
            }}
            style={{ width: "200px", height: "100px" }}
          >
            {availableColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      )}

      {availableColumns.length > 0 && (
        <button onClick={handleLoadDataset} style={{ marginBottom: 10 }}>
          Load Dataset
        </button>
      )}

      <div style={{ marginBottom: 10 }}>
        <label>Hidden Layers</label>
        <button onClick={addLayer} style={{ marginLeft: 10 }}>+ Add Layer</button>
      </div>

      {hiddenLayers.map((neurons, idx) => (
        <div key={idx} style={{ marginBottom: 5 }}>
          <span>Layer {idx + 1}: {neurons} neurons </span>
          <button onClick={() => updateLayer(idx, -1)}>-</button>
          <button onClick={() => updateLayer(idx, 1)}>+</button>
        </div>
      ))}

      <div style={{ marginBottom: 10 }}>
        <label>
          <input
            type="checkbox"
            checked={showWeights}
            onChange={(e) => setShowWeights(e.target.checked)}
          />
          Show Weights on Edges
        </label>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Line Thickness</label>
        <select value={lineThicknessMode} onChange={(e) => setLineThicknessMode(e.target.value as "auto" | "fixed")}>
          <option value="auto">Weight-based</option>
          <option value="fixed">Fixed</option>
        </select>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Zoom</label>
        <button onClick={handleZoomIn} style={{ marginLeft: 10 }}>+</button>
        <button onClick={handleZoomOut} style={{ marginLeft: 5 }}>-</button>
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
