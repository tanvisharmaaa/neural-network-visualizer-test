import React, { useState } from "react";
import Papa from "papaparse";

interface DatasetSelectorProps {
  onDatasetLoad: (data: {
    inputs: number[][];
    outputs: number[][];
    needsOutputNormalization?: boolean;
  }) => void;
  onInputNeuronsChange: (neurons: number) => void;
  onOutputNeuronsChange: (neurons: number) => void;
  onProblemTypeChange: (problemType: string) => void;
  problemType: string;
}

interface SampleDataset {
  name: string;
  description: string;
  filename: string;
  inputColumns: string[];
  outputColumn: string;
  problemType: "Classification" | "Regression";
}

const DatasetSelector: React.FC<DatasetSelectorProps> = ({
  onDatasetLoad,
  onInputNeuronsChange,
  onOutputNeuronsChange,
  onProblemTypeChange,
  problemType,
}) => {
  const [selectedOption, setSelectedOption] = useState<"sample" | "upload">("sample");
  const [selectedSample, setSelectedSample] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [labelColumn, setLabelColumn] = useState<string>("");
  const [outputColumns, setOutputColumns] = useState<string[]>([]);
  const [inputColumns, setInputColumns] = useState<string[]>([]);
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [fileContent, setFileContent] = useState<string | null>(null);

  const sampleDatasets: SampleDataset[] = [
    {
      name: "Iris Dataset",
      description: "Classic classification dataset with 4 features and 3 species",
      filename: "Iris.csv",
      inputColumns: ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
      outputColumn: "Species",
      problemType: "Classification",
    },
    {
      name: "Diabetes Dataset",
      description: "Regression dataset with 6 normalized features for diabetes progression",
      filename: "diabetes-simplified.json",
      inputColumns: ["age", "sex", "bmi", "bp", "s1", "s2"],
      outputColumn: "disease_progression",
      problemType: "Regression",
    },
  ];

  const loadSampleDataset = async (dataset: SampleDataset) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}datasets/${dataset.filename}`);
      if (!response.ok) {
        throw new Error(`Failed to load ${dataset.filename}`);
      }
      
      // Handle JSON datasets
      if (dataset.filename.endsWith('.json')) {
        const jsonData = await response.json();
        if (!validateJSONDataset(jsonData)) {
          alert("Invalid JSON dataset format.");
          setIsLoading(false);
          return;
        }
        
        onInputNeuronsChange(jsonData.inputs[0].length);
        onOutputNeuronsChange(jsonData.outputs[0].length);
        onProblemTypeChange(dataset.problemType);
        onDatasetLoad({
          inputs: jsonData.inputs,
          outputs: jsonData.outputs,
          needsOutputNormalization: dataset.problemType === "Regression",
        });
        setIsLoading(false);
        return;
      }
      
      // Handle CSV datasets
      const csvText = await response.text();
      
      Papa.parse(csvText, {
        complete: (result) => {
          try {
            const data = result.data as any[];
            if (data.length < 1) {
              alert("Sample dataset is empty or invalid.");
              setIsLoading(false);
              return;
            }

            const headers = Object.keys(data[0] || {}).filter((h) => h && h !== "");
            const rows = data.filter(
              (row) => row && Object.keys(row).length === headers.length
            );

            if (rows.length === 0) {
              alert("Sample dataset contains no valid rows.");
              setIsLoading(false);
              return;
            }

            // Process the data based on the dataset configuration
            const inputs: number[][] = [];
            const outputs: number[][] = [];

            if (dataset.problemType === "Classification") {
              // Get unique labels
              const uniqueLabels = [
                ...new Set(
                  rows
                    .map((row) => row[dataset.outputColumn]?.toString().trim().toLowerCase() || "")
                    .filter((val) => val && val.toUpperCase() !== "NA")
                ),
              ].sort();

              const labelToIndex = new Map<string, number>();
              uniqueLabels.forEach((label, idx) => labelToIndex.set(label, idx));

              const outputNeurons = uniqueLabels.length;
              onOutputNeuronsChange(outputNeurons);

              rows.forEach((row) => {
                const inputValues: number[] = [];
                const outputValue = row[dataset.outputColumn]?.toString().trim().toLowerCase() || "";

                // Process input columns
                dataset.inputColumns.forEach((col) => {
                  const val = parseFloat(row[col]?.toString().trim() || "0");
                  if (!isNaN(val)) {
                    inputValues.push(val);
                  }
                });

                // Process output
                if (inputValues.length === dataset.inputColumns.length && outputValue) {
                  const labelIndex = labelToIndex.get(outputValue);
                  if (labelIndex !== undefined) {
                    inputs.push(inputValues);
                    
                    if (outputNeurons === 1) {
                      // Binary classification
                      outputs.push([labelIndex]);
                    } else {
                      // Multi-class classification - one-hot encoding
                      const oneHot = Array(outputNeurons).fill(0);
                      oneHot[labelIndex] = 1;
                      outputs.push(oneHot);
                    }
                  }
                }
              });
            } else {
              // Regression
              rows.forEach((row) => {
                const inputValues: number[] = [];
                const outputValue = parseFloat(row[dataset.outputColumn]?.toString().trim() || "0");

                // Process input columns
                dataset.inputColumns.forEach((col) => {
                  const val = parseFloat(row[col]?.toString().trim() || "0");
                  if (!isNaN(val)) {
                    inputValues.push(val);
                  }
                });

                if (inputValues.length === dataset.inputColumns.length && !isNaN(outputValue)) {
                  inputs.push(inputValues);
                  outputs.push([outputValue]);
                }
              });
            }

            if (inputs.length > 0 && outputs.length > 0) {
              onInputNeuronsChange(inputs[0].length);
              onProblemTypeChange(dataset.problemType);
              onDatasetLoad({
                inputs,
                outputs,
                needsOutputNormalization: dataset.problemType === "Regression",
              });
            } else {
              alert("Failed to process sample dataset. No valid data found.");
            }
          } catch (error) {
            console.error("Error processing sample dataset:", error);
            alert(`Error processing sample dataset: ${error}`);
          } finally {
            setIsLoading(false);
          }
        },
        header: true,
        skipEmptyLines: "greedy",
        dynamicTyping: false,
      });
    } catch (error) {
      console.error("Error loading sample dataset:", error);
      alert(`Error loading sample dataset: ${error}`);
      setIsLoading(false);
    }
  };

  const handleSampleSelect = (datasetName: string) => {
    setSelectedSample(datasetName);
    const dataset = sampleDatasets.find(d => d.name === datasetName);
    if (dataset) {
      loadSampleDataset(dataset);
    }
  };

  // File upload functionality (moved from NetworkControls)
  const classifyColumns = (rows: any[], columns: string[]) => {
    const columnTypes = new Map<string, "numeric" | "categorical">();
    const columnMeans = new Map<string, number>();
    const columnUniques = new Map<string, Set<string>>();

    columns.forEach((col) => {
      const values = rows.map((row) => row[col]?.toString().trim() || "");
      const nonMissingValues = values.filter(
        (val) => val && val.toUpperCase() !== "NA"
      );
      const isNumeric =
        nonMissingValues.length > 0 &&
        nonMissingValues.every((val) => !isNaN(parseFloat(val)));

      columnTypes.set(col, isNumeric ? "numeric" : "categorical");

      if (isNumeric) {
        const numValues = nonMissingValues.map((val) => parseFloat(val));
        const mean =
          numValues.length > 0
            ? numValues.reduce((a, b) => a + b, 0) / numValues.length
            : 0;
        columnMeans.set(col, mean);
      } else {
        const uniques = new Set(
          nonMissingValues.map((val) => val.toLowerCase())
        );
        columnUniques.set(col, uniques);
        if (uniques.size > 10) {
          console.warn(
            `Column ${col} has too many unique values (${uniques.size}). Consider preprocessing.`
          );
        }
      }
    });

    return { columnTypes, columnMeans, columnUniques };
  };

  const processRow = (
    row: any,
    inputCols: string[],
    outputCols: string[],
    problemType: string,
    columnTypes: Map<string, "numeric" | "categorical">,
    columnMeans: Map<string, number>,
    columnUniques: Map<string, Set<string>>,
    labelToIndex: Map<string, number>
  ) => {
    const inputValues: number[] = [];
    for (const col of inputCols) {
      let val = row[col]?.toString().trim() || "";
      if (val.toUpperCase() === "NA" || val === "") {
        if (columnTypes.get(col) === "numeric") {
          val = columnMeans.get(col)!.toString();
        } else {
          val = "missing";
        }
      }
      if (columnTypes.get(col) === "numeric") {
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
      let val = row[col]?.toString().trim() || "";
      if (val.toUpperCase() === "NA" || val === "") {
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
        if (!label || label.toUpperCase() === "NA") return null;
        const index = labelToIndex.get(label);
        if (index === undefined) return null;
        outputValues.push(index);
      }
    }

    if (
      inputValues.every((val) => !isNaN(val)) &&
      outputValues.every((val) => !isNaN(val))
    ) {
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

        const headers = Object.keys(data[0] || {}).filter((h) => h && h !== "");
        if (headers.length < 2) {
          alert(
            "Invalid headers: CSV must have at least 2 unique, non-empty column names."
          );
          return;
        }

        const uniqueHeaders = [...new Set(headers)];
        if (uniqueHeaders.length !== headers.length) {
          alert("Invalid headers: CSV contains duplicate column names.");
          return;
        }

        const rows = data.filter(
          (row) => row && Object.keys(row).length === headers.length
        );
        if (rows.length === 0) {
          alert("CSV file contains no valid rows matching header count.");
          return;
        }

        let inputCols: string[] = inputColumns;
        let outputCols: string[] = [];
        let assumedOutputNeurons = 1;

        if (problemType === "Classification") {
          if (!labelColumn || !headers.includes(labelColumn)) {
            alert("Please select a valid label column for classification.");
            return;
          }
          outputCols = [labelColumn];
          const uniqueValues = [
            ...new Set(
              rows
                .map(
                  (row) =>
                    row[labelColumn]?.toString().trim().toLowerCase() || ""
                )
                .filter((val) => val && val.toUpperCase() !== "NA")
            ),
          ];
          if (uniqueValues.length === 0) {
            alert("No valid labels found in the selected label column.");
            return;
          }
          if (uniqueValues.length > 10) {
            alert(
              `The selected label column has too many unique values (${uniqueValues.length}), which may indicate it's not a categorical label. Please select a different column.`
            );
            return;
          }
          const isBinary = uniqueValues.length <= 2;
          assumedOutputNeurons = isBinary ? 1 : uniqueValues.length;
          onOutputNeuronsChange(assumedOutputNeurons);
        } else {
          if (outputColumns.length === 0) {
            alert("Please select at least one output column for regression.");
            return;
          }
          outputCols = outputColumns;
          assumedOutputNeurons = outputColumns.length;
          onOutputNeuronsChange(assumedOutputNeurons);
        }

        if (inputCols.length === 0) {
          alert("Please select at least one input column.");
          return;
        }

        const classificationResult = classifyColumns(rows, [
          ...inputCols,
          ...outputCols,
        ]);
        if (!classificationResult) return;
        const { columnTypes, columnMeans, columnUniques } =
          classificationResult;

        const inputs: number[][] = [];
        let rawOutputs: number[][] = [];
        const labelToIndex = new Map<string, number>();
        if (problemType === "Classification") {
          const uniqueValues = [
            ...new Set(
              rows
                .map(
                  (row) =>
                    row[labelColumn]?.toString().trim().toLowerCase() || ""
                )
                .filter((val) => val && val.toUpperCase() !== "NA")
            ),
          ];
          uniqueValues.sort();
          uniqueValues.forEach((val, idx) => labelToIndex.set(val, idx));
        }

        rows.forEach((row) => {
          const processed = processRow(
            row,
            inputCols,
            outputCols,
            problemType,
            columnTypes,
            columnMeans,
            columnUniques,
            labelToIndex
          );
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
          onInputNeuronsChange(inputs[0].length);
          onDatasetLoad({
            inputs,
            outputs,
            needsOutputNormalization: problemType === "Regression",
          });
        } else {
          alert(
            "Parsed dataset has no valid rows after filtering invalid or missing data. Check if selected columns have valid data."
          );
        }
      },
      header: true,
      skipEmptyLines: "greedy",
      dynamicTyping: false,
    });
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileExtension = file.name.split(".").pop()?.toLowerCase();
    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        if (fileExtension === "json") {
          const data = JSON.parse(event.target?.result as string);
          if (!validateJSONDataset(data)) {
            alert(
              "Invalid JSON dataset: Must contain 'inputs' and 'outputs' arrays with consistent numeric values."
            );
            return;
          }
          onInputNeuronsChange(data.inputs[0].length);
          onOutputNeuronsChange(data.outputs[0].length);
          setAvailableColumns([]);
          setLabelColumn("");
          setOutputColumns([]);
          setInputColumns([]);
          setFileContent(null);
          onDatasetLoad({
            ...data,
            needsOutputNormalization: problemType === "Regression",
          });
        } else if (fileExtension === "csv") {
          const content = event.target?.result as string;
          setFileContent(content);
          Papa.parse(content, {
            complete: (result) => {
              const data = result.data as any[];
              if (data.length < 1) {
                alert("CSV file is empty or invalid.");
                return;
              }

              const headers = Object.keys(data[0] || {}).filter(
                (h) => h && h !== ""
              );
              if (headers.length < 2) {
                alert(
                  "Invalid headers: CSV must have at least 2 unique, non-empty column names."
                );
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
            skipEmptyLines: "greedy",
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

  const validateJSONDataset = (
    data: any
  ): data is { inputs: number[][]; outputs: number[][] } => {
    if (
      !data.inputs ||
      !data.outputs ||
      !Array.isArray(data.inputs) ||
      !Array.isArray(data.outputs)
    ) {
      return false;
    }
    if (
      data.inputs.length === 0 ||
      data.outputs.length === 0 ||
      data.inputs.length !== data.outputs.length
    ) {
      return false;
    }
    const inputLength = data.inputs[0].length;
    const outputLength = data.outputs[0].length;
    return (
      data.inputs.every(
        (row: any) =>
          Array.isArray(row) &&
          row.length === inputLength &&
          row.every((val: any) => typeof val === "number" && !isNaN(val))
      ) &&
      data.outputs.every(
        (row: any) =>
          Array.isArray(row) &&
          row.length === outputLength &&
          row.every((val: any) => typeof val === "number" && !isNaN(val))
      )
    );
  };

  return (
    <div style={{ marginBottom: "20px", padding: "20px", border: "1px solid #e0e0e0", borderRadius: "8px", backgroundColor: "#fafafa", fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif", boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}>
      <h3 style={{ margin: "0 0 20px 0", color: "#212121", fontWeight: "500", fontSize: "20px", letterSpacing: "-0.02em" }}>Dataset Selection</h3>
      
      <div style={{ marginBottom: "15px" }}>
        <label style={{ display: "block", marginBottom: "12px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>
          Choose Dataset Source:
        </label>
        <div style={{ display: "flex", gap: "15px" }}>
          <label style={{ display: "flex", alignItems: "center", cursor: "pointer", fontSize: "14px", color: "#424242" }}>
            <input
              type="radio"
              name="datasetSource"
              value="sample"
              checked={selectedOption === "sample"}
              onChange={(e) => setSelectedOption(e.target.value as "sample" | "upload")}
              style={{ marginRight: "8px" }}
            />
            Sample Dataset
          </label>
          <label style={{ display: "flex", alignItems: "center", cursor: "pointer", fontSize: "14px", color: "#424242" }}>
            <input
              type="radio"
              name="datasetSource"
              value="upload"
              checked={selectedOption === "upload"}
              onChange={(e) => setSelectedOption(e.target.value as "sample" | "upload")}
              style={{ marginRight: "8px" }}
            />
            Upload Your Own
          </label>
        </div>
      </div>

      {selectedOption === "sample" && (
        <div>
          <label style={{ display: "block", marginBottom: "12px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>
            Select Sample Dataset:
          </label>
          <select
            value={selectedSample}
            onChange={(e) => handleSampleSelect(e.target.value)}
            disabled={isLoading}
            style={{
              width: "100%",
              padding: "10px 12px",
              borderRadius: "4px",
              border: "1px solid #e0e0e0",
              fontSize: "14px",
              fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
              backgroundColor: "#fff",
              color: "#424242"
            }}
          >
            <option value="">Choose a sample dataset...</option>
            {sampleDatasets.map((dataset) => (
              <option key={dataset.name} value={dataset.name}>
                {dataset.name} - {dataset.description}
              </option>
            ))}
          </select>
          
          {selectedSample && (
            <div style={{ marginTop: "10px", padding: "10px", backgroundColor: "#e8f4fd", borderRadius: "4px" }}>
              <strong>Selected:</strong> {selectedSample}
              <br />
              <strong>Type:</strong> {sampleDatasets.find(d => d.name === selectedSample)?.problemType}
              <br />
              <strong>Features:</strong> {sampleDatasets.find(d => d.name === selectedSample)?.inputColumns.join(", ")}
              <br />
              <strong>Target:</strong> {sampleDatasets.find(d => d.name === selectedSample)?.outputColumn}
            </div>
          )}
          
          {isLoading && (
            <div style={{ marginTop: "10px", color: "#666", fontStyle: "italic" }}>
              Loading dataset...
            </div>
          )}
        </div>
      )}

      {selectedOption === "upload" && (
        <div style={{ padding: "15px", backgroundColor: "#fff", borderRadius: "4px", border: "1px solid #ddd" }}>
          <div style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>
              Upload Your Own Dataset (JSON or CSV)
            </label>
            <input 
              type="file" 
              accept=".json,.csv" 
              onChange={handleFileUpload}
              style={{ 
                marginBottom: "10px", 
                width: "100%",
                padding: "8px",
                border: "1px solid #e0e0e0",
                borderRadius: "4px",
                fontSize: "14px",
                fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif"
              }}
            />
            <p style={{ margin: "0", fontSize: "12px", color: "#666" }}>
              Upload a CSV file with headers or a JSON file with "inputs" and "outputs" arrays.
            </p>
          </div>

          {availableColumns.length > 0 && (
            <div style={{ marginBottom: "15px" }}>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>
                Problem Type
              </label>
              <select
                value={problemType}
                onChange={(e) => onProblemTypeChange(e.target.value)}
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
          )}

          {availableColumns.length > 0 && (
            <div style={{ marginBottom: "15px" }}>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>
                Select Input Columns
              </label>
              <select
                multiple
                value={inputColumns}
                onChange={(e) => {
                  const selected = Array.from(e.target.selectedOptions).map(
                    (option) => option.value
                  );
                  setInputColumns(selected);
                }}
                style={{ 
                  width: "100%", 
                  height: "100px", 
                  marginBottom: "10px",
                  padding: "8px",
                  border: "1px solid #e0e0e0", 
                  borderRadius: "4px", 
                  fontSize: "14px",
                  fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
                  backgroundColor: "#fff",
                  color: "#424242"
                }}
              >
                {availableColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          )}

          {availableColumns.length > 0 && problemType === "Classification" && (
            <div style={{ marginBottom: "15px" }}>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>
                Select Label Column
              </label>
              <select
                value={labelColumn}
                onChange={(e) => setLabelColumn(e.target.value)}
                style={{ 
                  width: "100%", 
                  marginBottom: "10px",
                  padding: "10px 12px", 
                  border: "1px solid #e0e0e0", 
                  borderRadius: "4px", 
                  fontSize: "14px",
                  fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
                  backgroundColor: "#fff",
                  color: "#424242"
                }}
              >
                <option value="">Select a column</option>
                {availableColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          )}

          {availableColumns.length > 0 && problemType === "Regression" && (
            <div style={{ marginBottom: "15px" }}>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: "500", fontSize: "14px", color: "#424242" }}>
                Select Output Columns
              </label>
              <select
                multiple
                value={outputColumns}
                onChange={(e) => {
                  const selected = Array.from(e.target.selectedOptions).map(
                    (option) => option.value
                  );
                  setOutputColumns(selected);
                }}
                style={{ 
                  width: "100%", 
                  height: "100px", 
                  marginBottom: "10px",
                  padding: "8px",
                  border: "1px solid #e0e0e0", 
                  borderRadius: "4px", 
                  fontSize: "14px",
                  fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
                  backgroundColor: "#fff",
                  color: "#424242"
                }}
              >
                {availableColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          )}

          {availableColumns.length > 0 && (
            <button 
              onClick={processCSV} 
              style={{ 
                padding: "12px 24px", 
                backgroundColor: "#1976d2", 
                color: "white", 
                border: "none", 
                borderRadius: "4px", 
                cursor: "pointer",
                fontSize: "14px",
                fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif",
                fontWeight: "500",
                transition: "background-color 0.2s ease"
              }}
              onMouseOver={(e) => (e.target as HTMLButtonElement).style.backgroundColor = "#1565c0"}
              onMouseOut={(e) => (e.target as HTMLButtonElement).style.backgroundColor = "#1976d2"}
            >
              Load Dataset
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default DatasetSelector;