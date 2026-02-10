import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import NetworkControls from "./NetworkControls";
import NetworkGraph from "./NetworkGraph";
import LossChart from "./LossChart";
import DatasetSelector from "./DatasetSelector";
interface DebugWeightsDisplayProps {
  debugWeights: string;
  trainingPhase: string;
}
const DebugWeightsDisplay: React.FC<DebugWeightsDisplayProps> = ({
  debugWeights,
  trainingPhase,
}) => {
  let weightsData: number[][][] = [];
  try {
    const weightsStr = debugWeights.split(": ")[1];
    if (weightsStr) {
      weightsData = JSON.parse(weightsStr);
    }
  } catch (error) {
    console.error("Error parsing debug weights:", error);
  }
  return (
    <div className="mt-4 p-4 bg-gray-100 rounded-lg shadow">
      {trainingPhase && (
        <h4 className="text-lg font-semibold text-gray-800 mb-2">
          Training Phase: {trainingPhase}
        </h4>
      )}
      {weightsData.length > 0 ? (
        <div>
          <h4 className="text-lg font-semibold text-gray-800 mb-2">
            Debug Weights
          </h4>
          <div className="overflow-x-auto">
            {weightsData.map((layer, layerIdx) => (
              <div key={layerIdx} className="mb-4">
                <h5 className="text-md font-medium text-gray-700">
                  Layer {layerIdx} → {layerIdx + 1}
                </h5>
                <table className="table-auto border-collapse border border-gray-300 w-full">
                  <thead>
                    <tr className="bg-gray-200">
                      <th className="border border-gray-300 px-2 py-1 text-sm">
                        From
                      </th>
                      {layer[0]?.map((_, toIdx) => (
                        <th
                          key={toIdx}
                          className="border border-gray-300 px-2 py-1 text-sm"
                        >
                          To {toIdx + 1}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {layer.map((row, fromIdx) => (
                      <tr key={fromIdx}>
                        <td className="border border-gray-300 px-2 py-1 text-sm font-medium">
                          Neuron {fromIdx + 1}
                        </td>
                        {row.map((weight, toIdx) => (
                          <td
                            key={toIdx}
                            className="border border-gray-300 px-2 py-1 text-sm text-center"
                          >
                            {typeof weight === "number" && !isNaN(weight)
                              ? weight.toFixed(2)
                              : "N/A"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-gray-600">No weights data available.</p>
      )}
    </div>
  );
};
const NeuralNetworkVisualizer = React.memo(() => {
  const [inputNeurons, setInputNeurons] = useState(3);
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([2]);
  const [outputNeurons, setOutputNeurons] = useState(1);
  const [weights, setWeights] = useState<number[][][]>([]);
  const [biases, setBiases] = useState<number[][]>([]);
  const [activationFunction, setActivationFunction] = useState("sigmoid");
  const [problemType, setProblemType] = useState("Classification");
  const [loss, setLoss] = useState<string | null>(null);
  const [lossHistory, setLossHistory] = useState<
    { loss: number; metric: number; val_loss?: number; val_metric?: number }[]
  >([]);
  const [outputs, setOutputs] = useState<number[]>([]);
  const [currentInputs, setCurrentInputs] = useState<number[]>([]);
  const [pulses, setPulses] = useState<any[]>([]);
  const [displayedConnections, setDisplayedConnections] = useState<Set<string>>(
    new Set()
  );
  const [isTraining, setIsTraining] = useState(true);
  const [isTrained, setIsTrained] = useState(true);
  const [neuronEquations, setNeuronEquations] = useState<Map<string, string>>(
    new Map()
  );
  const [neuronValues, setNeuronValues] = useState<Map<string, number>>(
    new Map()
  );
  const [epochDisplay, setEpochDisplay] = useState(0);
  const [showWeights, setShowWeights] = useState(true);
  const [lineThicknessMode, setLineThicknessMode] = useState<"auto" | "fixed">(
    "auto"
  );
  const [zoomLevel, setZoomLevel] = useState(0.21); // Zoomed out by 30% (0.3 * 0.7 = 0.21)
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [dataset, setDataset] = useState<{
    inputs: number[][];
    outputs: number[][];
  }>({ inputs: [], outputs: [] });
  const [featureLabels, setFeatureLabels] = useState<string[]>([]);
  const [debugWeights, setDebugWeights] = useState<string>("");
  const [trainingPhase, setTrainingPhase] = useState<string>("");
  const [displayedWeights, setDisplayedWeights] = useState<Map<string, number>>(
    new Map()
  );
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [isPaused, setIsPaused] = useState(true);
  const [learningRate, setLearningRate] = useState(0.1);
  const [epochs, setEpochs] = useState(1);
  const [inputMin, setInputMin] = useState<number[]>([]);
  const [inputMax, setInputMax] = useState<number[]>([]);
  const [outputMin, setOutputMin] = useState<number[]>([]);
  const [outputMax, setOutputMax] = useState<number[]>([]);
  const graphPanelRef = useRef<HTMLDivElement>(null);
  const trainingInitiated = useRef(false);
  const stopTrainingRef = useRef(false);
  const cancelAnimationRef = useRef(false);
  const pauseResolveRef = useRef<(() => void) | null>(null);
  const animationSpeedRef = useRef(animationSpeed);
  const isPausedRef = useRef(isPaused);
  const trainingStateRef = useRef({
    loss: null as string | null,
    lossHistory: [] as {
      loss: number;
      metric: number;
      val_loss?: number;
      val_metric?: number;
    }[],
    outputs: [] as number[],
  });
  const sessionId = useRef(Date.now().toString());
  useEffect(() => {
    animationSpeedRef.current = animationSpeed;
  }, [animationSpeed]);
  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);
  const initializeZeroWeights = (
    inputNeurons: number,
    hiddenLayers: number[],
    outputNeurons: number
  ) => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const weights: number[][][] = [];
    for (let layerIdx = 0; layerIdx < layerSizes.length - 1; layerIdx++) {
      const fromSize = layerSizes[layerIdx];
      const toSize = layerSizes[layerIdx + 1];
      const layerWeights: number[][] = [];
      for (let fromIdx = 0; fromIdx < fromSize; fromIdx++) {
        const neuronWeights: number[] = Array(toSize).fill(0);
        layerWeights.push(neuronWeights);
      }
      weights.push(layerWeights);
    }
    return weights;
  };
  const initializeZeroBiases = (
    hiddenLayers: number[],
    outputNeurons: number
  ) => {
    const layerSizes = [...hiddenLayers, outputNeurons];
    const biases: number[][] = [];
    for (let layerIdx = 0; layerIdx < layerSizes.length; layerIdx++) {
      const size = layerSizes[layerIdx];
      const layerBiases: number[] = Array(size).fill(0);
      biases.push(layerBiases);
    }
    return biases;
  };
  const getNeuronPositions = () => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const positions: { x: number; y: number }[] = [];
    let neuronIndex = 0;
    layerSizes.forEach((count, layerIdx) => {
      const totalHeight = (count - 1) * 100;
      for (let neuronIdx = 0; neuronIdx < count; neuronIdx++) {
        positions[neuronIndex] = {
          x: layerIdx * 150 + 50,
          y: neuronIdx * 100 - totalHeight / 2 + 50, // Start much closer to top
        };
        neuronIndex++;
      }
    });
    return positions;
  };
  const getNeuronIndex = (layerIdx: number, neuronIdx: number) => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    let index = 0;
    for (let i = 0; i < layerIdx; i++) {
      index += layerSizes[i];
    }
    return index + neuronIdx;
  };
  const shuffleData = (
    inputs: number[][],
    outputs: number[][]
  ): { inputs: number[][]; outputs: number[][] } => {
    const indices = Array.from({ length: inputs.length }, (_, i) => i);
    indices.sort(() => Math.random() - 0.5);
    return {
      inputs: indices.map((i) => inputs[i]),
      outputs: indices.map((i) => outputs[i]),
    };
  };
  const normalize = (
    data: number[][],
    min: number[],
    max: number[]
  ): number[][] => {
    return data.map((row) =>
      row.map((val, idx) => {
        const range = max[idx] - min[idx];
        return range === 0 ? 0 : (val - min[idx]) / range;
      })
    );
  };
  const denormalize = (
    data: number[][],
    min: number[],
    max: number[]
  ): number[][] => {
    return data.map((row) =>
      row.map((val, idx) => {
        const range = max[idx] - min[idx];
        return range === 0 ? val : val * range + min[idx];
      })
    );
  };
  const computeMinMax = (
    data: number[][]
  ): { min: number[]; max: number[] } => {
    if (data.length === 0) return { min: [], max: [] };
    const numFeatures = data[0].length;
    const min = Array(numFeatures).fill(Infinity);
    const max = Array(numFeatures).fill(-Infinity);
    data.forEach((row) => {
      row.forEach((val, idx) => {
        if (!isNaN(val)) {
          min[idx] = Math.min(min[idx], val);
          max[idx] = Math.max(max[idx], val);
        }
      });
    });
    return {
      min: min.map((val) => (isFinite(val) ? val : 0)),
      max: max.map((val) => (isFinite(val) ? val : 1)),
    };
  };
  const applyActivation = (z: number, activation: string): number => {
    switch (activation) {
      case "sigmoid":
        return 1 / (1 + Math.exp(-z));
      case "relu":
        return Math.max(0, z);
      case "tanh":
        return Math.tanh(z);
      case "linear":
        return z;
      default:
        return z;
    }
  };
  const computeActivations = async (model: tf.Sequential, inputs: number[]) => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const activations: number[][] = [inputs];
    const neuronValuesMap = new Map<string, number>();
    inputs.forEach((value, idx) => {
      neuronValuesMap.set(`0-${idx}`, value);
    });
    const modelWeights = model.getWeights();
    const parsedWeights = (await Promise.all(
      modelWeights.filter((_, i) => i % 2 === 0).map((w) => w.array())
    )) as number[][][];
    const parsedBiases = (await Promise.all(
      modelWeights.filter((_, i) => i % 2 !== 0).map((b) => b.array())
    )) as number[][];
    for (let layerIdx = 0; layerIdx < layerSizes.length - 1; layerIdx++) {
      const layerWeights = parsedWeights[layerIdx] as number[][];
      const layerBiases = parsedBiases[layerIdx] as number[];
      const prevActivations = activations[layerIdx];
      const currentZs: number[] = [];
      const currentActivations: number[] = [];
      for (let toIdx = 0; toIdx < layerSizes[layerIdx + 1]; toIdx++) {
        let z = layerBiases[toIdx] || 0;
        for (let fromIdx = 0; fromIdx < layerSizes[layerIdx]; fromIdx++) {
          z +=
            (layerWeights[fromIdx][toIdx] || 0) *
            (prevActivations[fromIdx] || 0);
        }
        currentZs.push(z);
      }
      let layerActivation =
        layerIdx === layerSizes.length - 2
          ? problemType === "Regression"
            ? "linear"
            : outputNeurons > 1
            ? "softmax"
            : "sigmoid"
          : activationFunction;
      if (layerActivation === "softmax") {
        const maxZ = Math.max(...currentZs);
        const exps = currentZs.map((z) => Math.exp(z - maxZ));
        const sumExps = exps.reduce((a, b) => a + b, 0) || 1;
        exps.forEach((exp, idx) => {
          const activation = exp / sumExps;
          currentActivations.push(activation);
          neuronValuesMap.set(`${layerIdx + 1}-${idx}`, activation);
        });
      } else {
        currentZs.forEach((z, idx) => {
          const activation = applyActivation(z, layerActivation);
          currentActivations.push(activation);
          neuronValuesMap.set(`${layerIdx + 1}-${idx}`, activation);
        });
      }
      activations.push(currentActivations);
    }
    return { activations, neuronValuesMap, parsedWeights, parsedBiases };
  };
  const trainModel = async (numEpochs: number) => {
    console.log("trainModel called with numEpochs:", numEpochs);
    console.log("Dataset inputs length:", dataset.inputs.length);
    console.log("Dataset outputs length:", dataset.outputs.length);
    if (dataset.inputs.length === 0 || dataset.outputs.length === 0) {
      alert("Please upload a valid dataset first.");
      return;
    }
    if (dataset.inputs[0].length !== inputNeurons) {
      alert(
        `Dataset inputs have shape [${dataset.inputs.length}, ${dataset.inputs[0].length}], but model expects [null, ${inputNeurons}]. Please reload the dataset.`
      );
      return;
    }
    if (
      dataset.inputs.some((row) => row.some((val) => isNaN(val))) ||
      dataset.outputs.some((row) => row.some((val) => isNaN(val)))
    ) {
      alert("Dataset contains invalid (NaN) values. Please check your data.");
      return;
    }
    if (isTraining && !isPaused) {
      console.log("Training already in progress, skipping...");
      return;
    }
    console.log(
      `Starting trainModel for ${numEpochs} epochs, sessionId: ${sessionId.current}`
    );
    trainingInitiated.current = true;
    setIsTraining(true);
    setIsTrained(true);
    setIsPaused(false);
    stopTrainingRef.current = false;
    cancelAnimationRef.current = false;
    trainingStateRef.current = { loss: null, lossHistory: [], outputs: [] };
    let { inputs, outputs } = shuffleData(dataset.inputs, dataset.outputs);
    const { min: inMin, max: inMax } = computeMinMax(inputs);
    setInputMin(inMin);
    setInputMax(inMax);
    inputs = normalize(inputs, inMin, inMax);
    if (problemType === "Regression") {
      const { min, max } = computeMinMax(outputs);
      setOutputMin(min);
      setOutputMax(max);
      outputs = normalize(outputs, min, max);
    }
    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(outputs);
    if (xs.shape[0] === 0 || ys.shape[0] === 0 || xs.shape[1] === 0) {
      alert(
        "Empty or invalid dataset shape after preprocessing. Please check your selected columns and data."
      );
      xs.dispose();
      ys.dispose();
      return;
    }
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [inputNeurons],
        units: hiddenLayers[0] || outputNeurons,
        activation: activationFunction as any,
        kernelInitializer: "glorotNormal",
      })
    );
    hiddenLayers
      .slice(1)
      .forEach((units) =>
        model.add(
          tf.layers.dense({
            units,
            activation: activationFunction as any,
            kernelInitializer: "glorotNormal",
          })
        )
      );
    const outputActivation =
      problemType === "Regression"
        ? undefined
        : outputNeurons > 1
        ? "softmax"
        : "sigmoid";
    model.add(
      tf.layers.dense({ units: outputNeurons, activation: outputActivation })
    );
    const optimizer = tf.train.adam(Math.max(learningRate, 0.001));
    const lossFunction =
      problemType === "Regression"
        ? "meanSquaredError"
        : outputNeurons > 1
        ? "categoricalCrossentropy"
        : "binaryCrossentropy";
    const metrics = problemType === "Regression" ? [] : ["accuracy"];
    model.compile({ optimizer, loss: lossFunction, metrics });
    setModel(model);
    setLossHistory([]);
    setEpochDisplay(0);
    const validationSplit = dataset.inputs.length >= 10 ? 0.2 : 0;
    if (validationSplit === 0) {
      console.warn(
        "Dataset too small for validation split. No validation metrics will be available."
      );
    }
    let prevLoss = Infinity;
    const patience = 5;
    let wait = 0;
    for (let epoch = 1; epoch <= numEpochs; epoch++) {
      if (stopTrainingRef.current) {
        console.log("Training stopped by user.");
        break;
      }
      const preTrainWeights = (await Promise.all(
        model
          .getWeights()
          .filter((_, i) => i % 2 === 0)
          .map((w) => w.array())
      )) as number[][][];
      const preTrainBiases = (await Promise.all(
        model
          .getWeights()
          .filter((_, i) => i % 2 !== 0)
          .map((b) => b.array())
      )) as number[][];
      const history = await model.fit(xs, ys, {
        epochs: 1,
        validationSplit,
        verbose: 0,
        batchSize: 32,
        shuffle: true,
      });
      const lossValue =
        typeof history.history.loss?.[0] === "number"
          ? history.history.loss[0]
          : (history.history.loss?.[0] as any)?.dataSync?.()?.[0] ?? 0;
      console.log(`Epoch ${epoch} loss: ${lossValue.toFixed(4)}`);
      const valLossRaw = history.history.val_loss?.[0] ?? undefined;
      const valLoss =
        typeof valLossRaw === "number"
          ? valLossRaw
          : (valLossRaw as any)?.dataSync?.()?.[0] ?? undefined;
      let metricValue: number = 0;
      let valMetric: number | undefined = undefined;
      if (problemType === "Regression") {
        metricValue = Math.sqrt(lossValue);
        if (valLoss !== undefined) {
          valMetric = Math.sqrt(valLoss);
        }
      } else {
        const allKeys = Object.keys(history.history);
        const metricKeys = allKeys.filter(
          (k) => k !== "loss" && !k.startsWith("val_")
        );
        const metricKey = metricKeys.length > 0 ? metricKeys[0] : "acc";
        metricValue = (history.history[metricKey]?.[0] as number) ?? 0;
        const valMetricKeys = allKeys.filter(
          (k) => k.startsWith("val_") && k !== "val_loss"
        );
        const valMetricKey =
          valMetricKeys.length > 0 ? valMetricKeys[0] : "val_acc";
        valMetric = (history.history[valMetricKey]?.[0] as number) ?? undefined;
      }
      setLoss(lossValue.toFixed(4));
      setLossHistory((prev) => [
        ...prev,
        {
          loss: lossValue,
          metric: metricValue,
          val_loss: valLoss,
          val_metric: valMetric,
        },
      ]);
      setEpochDisplay(epoch);
      const postTrainWeights = (await Promise.all(
        model
          .getWeights()
          .filter((_, i) => i % 2 === 0)
          .map((w) => w.array())
      )) as number[][][];
      const postTrainBiases = (await Promise.all(
        model
          .getWeights()
          .filter((_, i) => i % 2 !== 0)
          .map((b) => b.array())
      )) as number[][];

      let weightDiffSum = 0;
      preTrainWeights.forEach((layer, layerIdx) => {
        layer.forEach((row, fromIdx) => {
          row.forEach((weight, toIdx) => {
            const postWeight = postTrainWeights[layerIdx][fromIdx][toIdx];
            weightDiffSum += Math.abs(weight - postWeight);
          });
        });
      });
      console.log(
        `Epoch ${epoch} weight change sum: ${weightDiffSum.toFixed(6)}`
      );
      const sampleIndex = Math.floor(Math.random() * inputs.length);
      const sampleInput = inputs[sampleIndex];
      const { neuronValuesMap } = await computeActivations(model, sampleInput);
      await playForward({
        model,
        neuronValuesMap,
        parsedWeights: preTrainWeights,
        parsedBiases: preTrainBiases,
      });
      await new Promise((resolve) =>
        setTimeout(resolve, 300 / animationSpeedRef.current)
      );
      await playBackward({
        model,
        neuronValuesMap,
        preTrainWeights: preTrainWeights,
        preTrainBiases: preTrainBiases,
        postTrainWeights: postTrainWeights,
        postTrainBiases: postTrainBiases,
      });
      if (valLoss !== undefined && valLoss > prevLoss) {
        wait++;
        if (wait >= patience) {
          console.log("Early stopping triggered.");
          break;
        }
      } else {
        wait = 0;
      }
      prevLoss = valLoss ?? lossValue;
    }
    xs.dispose();
    ys.dispose();
    setIsTraining(false);
    setIsPaused(false);
  };
  const playForward = async (epochLog: {
    model: tf.Sequential;
    neuronValuesMap: Map<string, number>;
    parsedWeights: number[][][];
    parsedBiases: number[][];
  }) => {
    const { neuronValuesMap: valueMap, parsedWeights, parsedBiases } = epochLog;
    const positions = getNeuronPositions();
    const equationMap = new Map<string, string>();
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const newDisplayedConnections = new Set<string>();
    const newDisplayedWeights = new Map<string, number>();
    let currentWeights: number[][][] = JSON.parse(
      JSON.stringify(parsedWeights as number[][][])
    );
    let currentBiases: number[][] = JSON.parse(
      JSON.stringify(parsedBiases as number[][])
    );
    const layers = parsedWeights.map((layerWeights, layerIdx) => {
      const steps = (layerWeights as number[][]).flatMap(
        (neuronWeights, fromIdx) =>
          neuronWeights.map((weight, toIdx) => ({
            fromIdx,
            toIdx,
            weight,
            bias: parsedBiases[layerIdx]?.[toIdx] ?? 0,
          }))
      );
      return { from: layerIdx, to: layerIdx + 1, steps };
    });
    for (const layer of layers) {
      if (
        layer.to > layerSizes.length - 1 ||
        layer.from >= layerSizes.length - 1
      ) {
        console.error(
          `Invalid layer indices: from=${layer.from}, to=${layer.to}`
        );
        continue;
      }
      const pulses = layer.steps.map((step: any) => {
        const fromIndex = getNeuronIndex(layer.from, step.fromIdx);
        const toIndex = getNeuronIndex(layer.to, step.toIdx);
        const from = positions[fromIndex];
        const to = positions[toIndex];
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        const controlX = midX - 20;
        const controlY = midY;
        const connectionKey = `${layer.from}-${step.fromIdx}-${step.toIdx}`;
        newDisplayedConnections.add(connectionKey);
        return {
          from,
          to,
          control: { x: controlX, y: controlY },
          progress: 0,
          direction: "forward" as const,
          fromIdx: step.fromIdx,
          toIdx: step.toIdx,
          layerIdx: layer.from,
          weight: step.weight,
          connectionKey,
        };
      });
      let progress = 0;
      while (progress < 1) {
        if (cancelAnimationRef.current) {
          console.log("Forward animation cancelled");
          setPulses([]);
          setDisplayedConnections(new Set());
          setDisplayedWeights(new Map());
          return;
        }
        if (isPausedRef.current) {
          await new Promise<void>((resolve) => {
            pauseResolveRef.current = resolve;
          });
          pauseResolveRef.current = null;
        }
        await new Promise((resolve) =>
          setTimeout(resolve, 16 / animationSpeedRef.current)
        );
        progress += 0.025 * animationSpeedRef.current;
        const updatedPulses = pulses.map((p) => ({
          ...p,
          progress: Math.min(1, progress),
        }));
        setPulses(updatedPulses);
        setDisplayedConnections(new Set(newDisplayedConnections));
        if (progress >= 1) {
          for (const step of layer.steps) {
            currentWeights[layer.from][step.fromIdx][step.toIdx] = step.weight;
            currentBiases[layer.to - 1][step.toIdx] = step.bias;
            newDisplayedWeights.set(
              `${layer.from}-${step.fromIdx}-${step.toIdx}`,
              step.weight
            );
            const equationKey = `${layer.to}-${step.toIdx}`;
            const weightsToNeuron = currentWeights[layer.from].map(
              (row) => (row[step.toIdx] as number) ?? 0
            );
            const bias = step.bias ?? 0;
            const terms = weightsToNeuron
              .map((w, i) => {
                const inputLabel = layer.from === 0 ? `x${i + 1}` : `h${i + 1}`;
                return `${(w as number).toFixed(2)}·${inputLabel}`;
              })
              .join(" + ");
            const equation = `z = ${terms} + ${bias.toFixed(2)}`;
            equationMap.set(equationKey, equation);
          }
          setWeights([...currentWeights]);
          setBiases([...currentBiases]);
          setDisplayedWeights(new Map(newDisplayedWeights));
          setNeuronEquations(new Map(equationMap));
          setNeuronValues(new Map(valueMap));
        }
      }
      setPulses([]);
      setDisplayedConnections(new Set(newDisplayedConnections));
      await new Promise((resolve) =>
        setTimeout(resolve, 150 / animationSpeedRef.current)
      );
    }
  };
  const playBackward = async (epochLog: {
    model: tf.Sequential;
    neuronValuesMap: Map<string, number>;
    preTrainWeights: number[][][];
    preTrainBiases: number[][];
    postTrainWeights: number[][][];
    postTrainBiases: number[][];
  }) => {
    const {
      neuronValuesMap: valueMap,
      preTrainWeights,
      postTrainWeights,
      postTrainBiases,
    } = epochLog;
    const positions = getNeuronPositions();
    const gradientMap = new Map<string, number>();
    const equationMap = new Map<string, string>();
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const newDisplayedConnections = new Set<string>();
    const newDisplayedWeights = new Map<string, number>();
    let currentWeights: number[][][] = JSON.parse(
      JSON.stringify(preTrainWeights as number[][][])
    );
    let currentBiases: number[][] = JSON.parse(
      JSON.stringify(postTrainBiases as number[][])
    );
    const layers = postTrainWeights.map((layerWeights, layerIdx) => {
      const steps = (layerWeights as number[][]).flatMap(
        (neuronWeights, fromIdx) =>
          neuronWeights.map((weight, toIdx) => ({
            fromIdx,
            toIdx,
            weight,
            bias: postTrainBiases[layerIdx]?.[toIdx] ?? 0,
          }))
      );
      return { from: layerIdx, to: layerIdx + 1, steps };
    });
    for (const layer of [...layers].reverse()) {
      if (
        layer.to > layerSizes.length - 1 ||
        layer.from >= layerSizes.length - 1
      ) {
        console.error(
          `Invalid layer indices: from=${layer.from}, to=${layer.to}`
        );
        continue;
      }
      const pulses = layer.steps.map((step: any) => {
        const fromIndex = getNeuronIndex(layer.to, step.toIdx);
        const toIndex = getNeuronIndex(layer.from, step.fromIdx);
        const from = positions[fromIndex];
        const to = positions[toIndex];
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        const controlX = midX + 20;
        const controlY = midY;
        const connectionKey = `${layer.from}-${step.fromIdx}-${step.toIdx}`;
        newDisplayedConnections.add(connectionKey);
        return {
          from,
          to,
          control: { x: controlX, y: controlY },
          progress: 0,
          direction: "backward" as const,
          fromIdx: step.fromIdx,
          toIdx: step.toIdx,
          layerIdx: layer.from,
          weight: step.weight,
          connectionKey,
        };
      });
      let progress = 0;
      while (progress < 1) {
        if (cancelAnimationRef.current) {
          console.log("Backward animation cancelled");
          setPulses([]);
          setDisplayedConnections(new Set());
          setDisplayedWeights(new Map());
          return;
        }
        if (isPausedRef.current) {
          await new Promise<void>((resolve) => {
            pauseResolveRef.current = resolve;
          });
          pauseResolveRef.current = null;
        }
        await new Promise((resolve) =>
          setTimeout(resolve, 16 / animationSpeedRef.current)
        );
        progress += 0.025 * animationSpeedRef.current;
        const updatedPulses = pulses.map((p) => ({
          ...p,
          progress: Math.min(1, progress),
        }));

        setPulses(updatedPulses);
        setDisplayedConnections(new Set(newDisplayedConnections));
        if (progress >= 1) {
          for (const step of layer.steps) {
            const prevWeight =
              currentWeights[layer.from][step.fromIdx][step.toIdx];
            currentWeights[layer.from][step.fromIdx][step.toIdx] = step.weight;
            currentBiases[layer.to - 1][step.toIdx] = step.bias;
            newDisplayedWeights.set(
              `${layer.from}-${step.fromIdx}-${step.toIdx}`,
              step.weight
            );
            const gradientKey = `${layer.from}-${step.fromIdx}`;
            const prevGradient = gradientMap.get(gradientKey) || 0;
            gradientMap.set(
              gradientKey,
              prevGradient + Math.abs(step.weight - prevWeight)
            );
            const equationKey = `${layer.to}-${step.toIdx}`;
            const weightsToNeuron = currentWeights[layer.from].map(
              (row) => (row[step.toIdx] as number) ?? 0
            );
            const bias = step.bias ?? 0;
            const terms = weightsToNeuron
              .map((w, i) => {
                const inputLabel = layer.from === 0 ? `x${i + 1}` : `h${i + 1}`;
                return `${(w as number).toFixed(2)}·${inputLabel}`;
              })
              .join(" + ");
            const equation = `z = ${terms} + ${bias.toFixed(2)}`;
            equationMap.set(equationKey, equation);
          }
          setWeights([...currentWeights]);
          setBiases([...currentBiases]);
          setDisplayedWeights(new Map(newDisplayedWeights));
          setNeuronEquations(new Map(equationMap));
          setNeuronValues(new Map(valueMap));
        }
      }
      setPulses([]);
      setDisplayedConnections(new Set(newDisplayedConnections));
      await new Promise((resolve) =>
        setTimeout(resolve, 150 / animationSpeedRef.current)
      );
    }
  };
  const predict = async (rawInputs: number[]) => {
    if (!model) {
      alert("Please train the model first.");
      return;
    }
    if (rawInputs.length !== inputNeurons) {
      alert(
        `Expected ${inputNeurons} inputs, but received ${rawInputs.length}.`
      );
      return;
    }
    if (rawInputs.some((val) => isNaN(val))) {
      alert("All inputs must be valid numbers.");
      return;
    }
    const normalizedInputs = normalize([rawInputs], inputMin, inputMax)[0];
    setCurrentInputs(normalizedInputs.map((val) => parseFloat(val.toFixed(3))));
    const xs = tf.tensor2d([normalizedInputs]);
    const pred = model.predict(xs) as tf.Tensor;
    let predArray = (await pred.array()) as number[][];
    if (problemType === "Regression") {
      predArray = denormalize(predArray, outputMin, outputMax);
    }
    setOutputs(
      predArray.flatMap((v: any) =>
        v.map((val: number) => parseFloat(val.toFixed(3)))
      )
    );
    xs.dispose();
    const modelWeights = model.getWeights();
    const parsedWeights = (await Promise.all(
      modelWeights.filter((_, i) => i % 2 === 0).map((w) => w.array())
    )) as number[][][];
    const parsedBiases = (await Promise.all(
      modelWeights.filter((_, i) => i % 2 !== 0).map((b) => b.array())
    )) as number[][];
    setDebugWeights("Prediction Weights: " + JSON.stringify(parsedWeights));
    setTrainingPhase("Prediction");
    const newDisplayedWeights = new Map<string, number>();
    parsedWeights.forEach((layerWeights, layerIdx) => {
      layerWeights.forEach((neuronWeights, fromIdx) => {
        neuronWeights.forEach((weight, toIdx) => {
          newDisplayedWeights.set(`${layerIdx}-${fromIdx}-${toIdx}`, weight);
        });
      });
    });
    cancelAnimationRef.current = false;
    setDisplayedWeights(newDisplayedWeights);
    const { neuronValuesMap } = await computeActivations(
      model,
      normalizedInputs
    );
    await playForward({ model, neuronValuesMap, parsedWeights, parsedBiases });
    setTrainingPhase("");
  };
  const onDatasetUpload = (data: {
    inputs: number[][];
    outputs: number[][];
    needsOutputNormalization?: boolean;
    featureLabels?: string[];
  }) => {
    if (
      data.inputs.length > 0 &&
      data.outputs.length > 0 &&
      data.inputs.length === data.outputs.length &&
      data.inputs.every((row) => row.length === data.inputs[0].length) &&
      data.outputs.every((row) => row.length === data.outputs[0].length)
    ) {
      setDataset({ inputs: data.inputs, outputs: data.outputs });
      setFeatureLabels(data.featureLabels || []);
      console.log("Dataset uploaded successfully:", { inputs: data.inputs, outputs: data.outputs, featureLabels: data.featureLabels });
      // Reset training state when new dataset is loaded
      setIsTraining(false);
      setIsPaused(false);
      setIsTrained(false);
    }
  };
  useEffect(() => {
    if (dataset.inputs.length > 0) {
      const newInputNeurons = dataset.inputs[0].length;
      const newOutputNeurons = dataset.outputs[0].length;
      setInputNeurons(newInputNeurons);
      setOutputNeurons(newOutputNeurons);
      setWeights(
        initializeZeroWeights(newInputNeurons, hiddenLayers, newOutputNeurons)
      );
      setBiases(initializeZeroBiases(hiddenLayers, newOutputNeurons));
      setLossHistory([]);
      setNeuronEquations(new Map());
      setNeuronValues(new Map());
      setOutputs([]);
      setCurrentInputs([]);
      setPulses([]);
      setDisplayedConnections(new Set());
      setDisplayedWeights(new Map());
      setModel(null);
      setIsTrained(false);
      setDebugWeights("");
      setTrainingPhase("");
      setInputMin([]);
      setInputMax([]);
      setOutputMin([]);
      setOutputMax([]);
    } else if (
      inputNeurons !== 3 ||
      outputNeurons !== 1 ||
      hiddenLayers.toString() !== "2"
    ) {
      setInputNeurons(3);
      setOutputNeurons(1);
      setHiddenLayers([2]);
      setWeights(initializeZeroWeights(3, [2], 1));
      setBiases(initializeZeroBiases([2], 1));
      setFeatureLabels([]);
    }
  }, [dataset, hiddenLayers, inputNeurons, outputNeurons]);
  const handlePlay = async (epochs: number) => {
    console.log("handlePlay called with epochs:", epochs);
    console.log("Current dataset:", dataset);
    console.log("isPaused:", isPaused);
    console.log("isTraining:", isTraining);
    if (isPaused) {
      setIsPaused(false);
      if (pauseResolveRef.current) {
        pauseResolveRef.current();
        pauseResolveRef.current = null;
      }
    } else {
      await trainModel(epochs);
      console.log("handlePlay completed");
    }
  };
  const handlePause = () => {
    if (isTraining && !isPaused) {
      setIsPaused(true);
    }
  };
  const handleStop = () => {
    stopTrainingRef.current = true;
    cancelAnimationRef.current = true;
    setIsPaused(false);
    if (pauseResolveRef.current) {
      pauseResolveRef.current();
      pauseResolveRef.current = null;
    }
    setPulses([]);
    setDisplayedConnections(new Set());
    setDisplayedWeights(new Map());
    setCurrentInputs([]);
    setDebugWeights("");
    setTrainingPhase("");
    console.log("Stop training triggered, clearing visualizations");
  };
  return (
    <div
      style={{
        padding: "20px",
        backgroundColor: "#ffffff",
        minHeight: "100vh",
        fontFamily: "Roboto, Helvetica Neue, Arial, sans-serif"
      }}
    >
      <h1 style={{ textAlign: "center", fontSize: "32px", color: "#212121", fontWeight: "400", marginBottom: "30px", letterSpacing: "-0.02em" }}>
        NeuroViz
      </h1>
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: "20px", marginBottom: "20px" }}>
        {/* Left: Dataset Selector */}
        <div className="panel" style={{ padding: "20px", border: "1px solid #e0e0e0", borderRadius: "8px", backgroundColor: "#fafafa", fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif", boxShadow: "0 1px 3px rgba(0,0,0,0.1)", display: "flex", flexDirection: "column", gap: "20px" }}>
          <DatasetSelector
            onDatasetLoad={onDatasetUpload}
            onInputNeuronsChange={setInputNeurons}
            onOutputNeuronsChange={setOutputNeurons}
            onProblemTypeChange={setProblemType}
            problemType={problemType}
          />
          
          {/* Information Panel */}
          {dataset.inputs.length > 0 && (
            <div style={{ padding: "15px", backgroundColor: "#ffffff", border: "1px solid #e0e0e0", borderRadius: "8px" }}>
              <h3 style={{ margin: "0 0 15px 0", color: "#212121", fontWeight: "500", fontSize: "16px", letterSpacing: "-0.02em" }}>Network Information</h3>
              
              <div style={{ display: "flex", flexDirection: "column", gap: "12px", fontSize: "13px" }}>
                {/* Dataset Info */}
                <div>
                  <div style={{ fontWeight: "600", color: "#424242", marginBottom: "4px" }}>Dataset</div>
                  <div style={{ color: "#666", paddingLeft: "8px" }}>
                    <div>Samples: {dataset.inputs.length}</div>
                    <div>Features: {inputNeurons}</div>
                    <div>Type: {problemType}</div>
                  </div>
                </div>
                
                {/* Network Architecture */}
                <div>
                  <div style={{ fontWeight: "600", color: "#424242", marginBottom: "4px" }}>Architecture</div>
                  <div style={{ color: "#666", paddingLeft: "8px" }}>
                    <div>Input: {inputNeurons}</div>
                    <div>Hidden: [{hiddenLayers.join(", ")}]</div>
                    <div>Output: {outputNeurons}</div>
                    <div>Activation: {activationFunction}</div>
                  </div>
                </div>
                
                {/* Training Status */}
                <div>
                  <div style={{ fontWeight: "600", color: "#424242", marginBottom: "4px" }}>Training Status</div>
                  <div style={{ color: "#666", paddingLeft: "8px" }}>
                    <div>Status: {isTraining ? (isPaused ? "Paused" : "Training") : isTrained ? "Trained" : "Not Started"}</div>
                    <div>Epoch: {epochDisplay} / {epochs}</div>
                    {loss && <div>Current Loss: {loss}</div>}
                    {lossHistory.length > 0 && (
                      <div>Best Loss: {Math.min(...lossHistory.map(h => h.loss)).toFixed(4)}</div>
                    )}
                    {lossHistory.length > 0 && problemType === "Classification" && (
                      <div>Best Accuracy: {(Math.max(...lossHistory.map(h => h.metric)) * 100).toFixed(2)}%</div>
                    )}
                  </div>
                </div>
                
                {/* Hyperparameters */}
                <div>
                  <div style={{ fontWeight: "600", color: "#424242", marginBottom: "4px" }}>Hyperparameters</div>
                  <div style={{ color: "#666", paddingLeft: "8px" }}>
                    <div>Learning Rate: {learningRate}</div>
                    <div>Animation Speed: {animationSpeed}x</div>
                  </div>
                </div>
                
                {/* Model Stats */}
                {weights.length > 0 && (
                  <div>
                    <div style={{ fontWeight: "600", color: "#424242", marginBottom: "4px" }}>Model Statistics</div>
                    <div style={{ color: "#666", paddingLeft: "8px" }}>
                      {(() => {
                        let totalParams = 0;
                        weights.forEach((layer, i) => {
                          layer.forEach(neuronWeights => {
                            totalParams += neuronWeights.length;
                          });
                          if (biases[i]) {
                            totalParams += biases[i].length;
                          }
                        });
                        return <div>Total Parameters: {totalParams}</div>;
                      })()}
                      <div>Layers: {weights.length}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Middle: Play/Pause/Stop Controls */}
        <div className="panel" style={{ padding: "20px", border: "1px solid #e0e0e0", borderRadius: "8px", backgroundColor: "#fafafa", fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif", boxShadow: "0 1px 3px rgba(0,0,0,0.1)", display: "flex", flexDirection: "column", gap: "10px", alignItems: "center", justifyContent: "flex-start" }}>
          {dataset.inputs.length > 0 && (
            <>
              <div style={{ marginTop: "auto", display: "flex", gap: "10px", flexDirection: "column", width: "100%" }}>
                <div style={{ marginBottom: "10px" }}>
                  <label style={{ display: "block", marginBottom: "5px", fontSize: "12px", color: "#424242" }}>Epochs</label>
                  <input
                    type="number"
                    value={epochs}
                    min={1}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      setEpochs(!isNaN(val) && val >= 1 ? val : 1);
                    }}
                    style={{ width: "100%", padding: "5px", border: "1px solid #e0e0e0", borderRadius: "4px" }}
                  />
                </div>
                <button
                  onClick={() => {
                    if (isTraining && !isPaused) {
                      handlePause();
                    } else {
                      handlePlay(epochs);
                    }
                  }}
                  disabled={dataset.inputs.length === 0}
                  style={{ 
                    padding: "10px 20px", 
                    backgroundColor: (isTraining && !isPaused) ? "#f57c00" : "#1976d2", 
                    color: "white", 
                    border: "none", 
                    borderRadius: "4px", 
                    cursor: dataset.inputs.length > 0 ? "pointer" : "not-allowed",
                    fontSize: "14px",
                    fontWeight: "500",
                    opacity: dataset.inputs.length > 0 ? 1 : 0.5
                  }}
                >
                  {isTraining && !isPaused ? "Pause" : "Play"}
                </button>
                <button
                  onClick={handleStop}
                  disabled={!isTraining}
                  style={{ 
                    padding: "10px 20px", 
                    backgroundColor: "#d32f2f", 
                    color: "white", 
                    border: "none", 
                    borderRadius: "4px", 
                    cursor: isTraining ? "pointer" : "not-allowed",
                    fontSize: "14px",
                    fontWeight: "500",
                    opacity: isTraining ? 1 : 0.5
                  }}
                >
                  Stop
                </button>
              </div>
              {loss && (
                <p style={{ marginTop: "10px", color: "#2e7d32", fontSize: "14px", fontWeight: "500", textAlign: "center" }}>Loss: {loss}</p>
              )}
              {outputs.length > 0 && (
                <p style={{ marginTop: "10px", color: "#2e7d32", fontSize: "14px", fontWeight: "500", textAlign: "center" }}>
                  Predictions: {outputs.join(", ")}
                </p>
              )}
            </>
          )}
        </div>
        
        {/* Right: Network Edit Options */}
        <div className="panel" style={{ padding: "20px", border: "1px solid #e0e0e0", borderRadius: "8px", backgroundColor: "#fafafa", fontFamily: "'Roboto', 'Helvetica Neue', Arial, sans-serif", boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}>
          {dataset.inputs.length > 0 && (
            <NetworkControls
              hiddenLayers={hiddenLayers}
              setHiddenLayers={setHiddenLayers}
              inputNeurons={inputNeurons}
              setInputNeurons={setInputNeurons}
              outputNeurons={outputNeurons}
              setOutputNeurons={setOutputNeurons}
              activationFunction={activationFunction}
              setActivationFunction={setActivationFunction}
              problemType={problemType}
              setProblemType={setProblemType}
              showWeights={showWeights}
              setShowWeights={setShowWeights}
              onPredict={predict}
              lineThicknessMode={lineThicknessMode}
              setLineThicknessMode={setLineThicknessMode}
              zoomLevel={zoomLevel}
              setZoomLevel={setZoomLevel}
              animationSpeed={animationSpeed}
              setAnimationSpeed={setAnimationSpeed}
              learningRate={learningRate}
              setLearningRate={setLearningRate}
              hasDataset={dataset.inputs.length > 0}
              epochs={epochs}
              setEpochs={setEpochs}
            />
          )}
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "row", gap: "20px", alignItems: "flex-start" }}>
          <div
            className="graph-panel"
            ref={graphPanelRef}
            style={{
              position: "relative",
              overflow: "visible",
              flex: "1 1 70%",
              backgroundColor: "#ffffff",
              border: "1px solid #e0e0e0",
              borderRadius: "8px",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              padding: "0", // No padding
            }}
          >
            {inputNeurons > 0 && outputNeurons > 0 ? (
              <NetworkGraph
                key={`${inputNeurons}-${outputNeurons}`}
                inputNeurons={inputNeurons}
                hiddenLayers={hiddenLayers}
                outputNeurons={outputNeurons}
                weights={weights}
                biases={biases}
                activationFunction={activationFunction}
                pulses={pulses}
                neuronEquations={neuronEquations}
                neuronValues={neuronValues}
                showWeights={showWeights}
                lineThicknessMode={lineThicknessMode}
                zoomLevel={zoomLevel}
                epochDisplay={epochDisplay}
                problemType={problemType}
                currentInputs={currentInputs}
                outputs={outputs}
                hasDataset={
                  dataset.inputs.length > 0 && dataset.outputs.length > 0
                }
                isTrained={isTrained}
                displayedWeights={displayedWeights}
                displayedConnections={displayedConnections}
                featureLabels={featureLabels}
              />
            ) : (
              <div
                style={{ textAlign: "center", marginTop: "40px", color: "#757575", fontSize: "16px", fontWeight: "400" }}
              >
                <p>
                  No dataset loaded. Showing dummy network. Please load a dataset.
                </p>
              </div>
            )}
            {(debugWeights || trainingPhase) && (
              <DebugWeightsDisplay
                debugWeights={debugWeights}
                trainingPhase={trainingPhase}
              />
            )}
          </div>
          
          {/* Loss Chart beside the network */}
          {dataset.inputs.length > 0 && lossHistory.length > 0 && (
            <div style={{
              backgroundColor: "#ffffff",
              border: "1px solid #e0e0e0",
              borderRadius: "8px",
              padding: "15px",
              height: "500px",
              width: "400px",
              flex: "0 0 400px",
              display: "flex",
              flexDirection: "column"
            }}>
              <h3 style={{ 
                margin: "0 0 10px 0", 
                color: "#212121", 
                fontWeight: "500", 
                fontSize: "14px" 
              }}>
                Training Progress
              </h3>
              <div style={{ flex: "1", minHeight: 0 }}>
                <LossChart lossHistory={lossHistory} problemType={problemType} />
              </div>
            </div>
          )}
      </div>
    </div>
  );
});
export default NeuralNetworkVisualizer;
