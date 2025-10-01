// NeuralNetworkVisualizer.tsx (updated for dynamic pause and speed)
import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import NetworkControls from "./NetworkControls";
import NetworkGraph from "./NetworkGraph";
import LossChart from "./LossChart";

interface DebugWeightsDisplayProps {
  debugWeights: string;
  trainingPhase: string;
}

const DebugWeightsDisplay: React.FC<DebugWeightsDisplayProps> = ({ debugWeights, trainingPhase }) => {
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
          <h4 className="text-lg font-semibold text-gray-800 mb-2">Debug Weights</h4>
          <div className="overflow-x-auto">
            {weightsData.map((layer, layerIdx) => (
              <div key={layerIdx} className="mb-4">
                <h5 className="text-md font-medium text-gray-700">
                  Layer {layerIdx} → {layerIdx + 1}
                </h5>
                <table className="table-auto border-collapse border border-gray-300 w-full">
                  <thead>
                    <tr className="bg-gray-200">
                      <th className="border border-gray-300 px-2 py-1 text-sm">From</th>
                      {layer[0]?.map((_, toIdx) => (
                        <th key={toIdx} className="border border-gray-300 px-2 py-1 text-sm">
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
                            {typeof weight === 'number' && !isNaN(weight) ? weight.toFixed(2) : "N/A"}
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
  const [lossHistory, setLossHistory] = useState<{ loss: number; accuracy: number; val_loss?: number }[]>([]);
  const [outputs, setOutputs] = useState<number[]>([]);
  const [currentInputs, setCurrentInputs] = useState<number[]>([]);
  const [pulses, setPulses] = useState<any[]>([]);
  const [displayedConnections, setDisplayedConnections] = useState<Set<string>>(new Set());
  const [isTraining, setIsTraining] = useState(false);
  const [isTrained, setIsTrained] = useState(false);
  const [neuronEquations, setNeuronEquations] = useState<Map<string, string>>(new Map());
  const [neuronValues, setNeuronValues] = useState<Map<string, number>>(new Map());
  const [neuronGradients, setNeuronGradients] = useState<Map<string, number>>(new Map());
  const [epochDisplay, setEpochDisplay] = useState(0);
  const [showBackpropBanner, setShowBackpropBanner] = useState(false);
  const [showWeights, setShowWeights] = useState(true);
  const [lineThicknessMode, setLineThicknessMode] = useState<"auto" | "fixed">("auto");
  const [zoomLevel, setZoomLevel] = useState(1);
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [dataset, setDataset] = useState({ inputs: [], outputs: [] });
  const [debugWeights, setDebugWeights] = useState<string>("");
  const [trainingPhase, setTrainingPhase] = useState<string>("");
  const [displayedWeights, setDisplayedWeights] = useState<Map<string, number>>(new Map());
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [isPaused, setIsPaused] = useState(false);

  const graphPanelRef = useRef<HTMLDivElement>(null);
  const trainingInitiated = useRef(false);
  const stopTrainingRef = useRef(false);
  const cancelAnimationRef = useRef(false);
  const pauseResolveRef = useRef<(() => void) | null>(null);
  const animationSpeedRef = useRef(animationSpeed);
  const isPausedRef = useRef(isPaused);
  const trainingStateRef = useRef({
    loss: null as string | null,
    lossHistory: [] as { loss: number; accuracy: number; val_loss?: number }[],
    outputs: [] as number[],
  });
  const sessionId = useRef(Date.now().toString());

  useEffect(() => {
    animationSpeedRef.current = animationSpeed;
  }, [animationSpeed]);

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  useEffect(() => {
    console.log("lossHistory updated:", lossHistory);
  }, [lossHistory]);

  useEffect(() => {
    if (dataset.inputs.length > 0 && dataset.outputs.length > 0) {
      const newInputNeurons = dataset.inputs[0].length;
      const newOutputNeurons = dataset.outputs[0].length;
      setInputNeurons(newInputNeurons);
      setOutputNeurons(newOutputNeurons);
      setWeights(initializeZeroWeights(newInputNeurons, hiddenLayers, newOutputNeurons));
      setBiases(initializeZeroBiases(hiddenLayers, newOutputNeurons));
      setLossHistory([]);
      setNeuronEquations(new Map());
      setNeuronValues(new Map());
      setNeuronGradients(new Map());
      setOutputs([]);
      setCurrentInputs([]);
      setPulses([]);
      setDisplayedConnections(new Set());
      setDisplayedWeights(new Map());
      setModel(null);
      setIsTrained(false);
      setDebugWeights("");
      setTrainingPhase("");
    }
  }, [dataset, hiddenLayers]);

  const initializeZeroWeights = (inputNeurons: number, hiddenLayers: number[], outputNeurons: number) => {
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

  const initializeZeroBiases = (hiddenLayers: number[], outputNeurons: number) => {
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
          y: neuronIdx * 100 - totalHeight / 2 + 250,
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

  const preprocessData = (inputs: number[][]) => {
    const numFeatures = inputs[0].length;
    const mins = Array(numFeatures).fill(Infinity);
    const maxs = Array(numFeatures).fill(-Infinity);

    inputs.forEach(row => {
      row.forEach((val, idx) => {
        mins[idx] = Math.min(mins[idx], val);
        maxs[idx] = Math.max(maxs[idx], val);
      });
    });

    return inputs.map(row =>
      row.map((val, idx) => {
        const range = maxs[idx] - mins[idx];
        return range === 0 ? 0 : (val - mins[idx]) / range;
      })
    );
  };

  const applyActivation = (z: number, activation: string): number => {
    if (activation === "sigmoid") {
      return 1 / (1 + Math.exp(-z));
    } else if (activation === "relu") {
      return Math.max(0, z);
    } else if (activation === "tanh") {
      return Math.tanh(z);
    }
    return z;
  };

  const computeActivations = async (model: tf.Sequential, inputs: number[][]) => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const activations: number[][] = [inputs[0]];
    const neuronValuesMap = new Map<string, number>();

    inputs[0].forEach((value, idx) => {
      neuronValuesMap.set(`0-${idx}`, value);
    });

    const modelWeights = model.getWeights();
    const parsedWeights = await Promise.all(
      modelWeights.filter((_, i) => i % 2 === 0).map((w) => w.array())
    );
    const parsedBiases = await Promise.all(
      modelWeights.filter((_, i) => i % 2 !== 0).map((b) => b.array())
    );

    for (let layerIdx = 0; layerIdx < layerSizes.length - 1; layerIdx++) {
      const layerWeights = parsedWeights[layerIdx] as number[][];
      const layerBiases = parsedBiases[layerIdx] as number[];
      const prevActivations = activations[layerIdx];
      const currentActivations: number[] = [];

      for (let toIdx = 0; toIdx < layerSizes[layerIdx + 1]; toIdx++) {
        let z = layerBiases[toIdx] || 0;
        for (let fromIdx = 0; fromIdx < layerSizes[layerIdx]; fromIdx++) {
          z += (layerWeights[fromIdx][toIdx] || 0) * (prevActivations[fromIdx] || 0);
        }
        const activation = applyActivation(
          z,
          layerIdx === layerSizes.length - 2
            ? (problemType === "Classification" && outputNeurons > 1 ? "softmax" : "sigmoid")
            : activationFunction
        );
        currentActivations.push(activation);
        neuronValuesMap.set(`${layerIdx + 1}-${toIdx}`, activation);
      }
      activations.push(currentActivations);
    }

    return { activations, neuronValuesMap, parsedWeights, parsedBiases };
  };

  const trainModel = async (numEpochs: number) => {
    console.log(`trainModel called with numEpochs: ${numEpochs}`);
    if (dataset.inputs.length === 0 || dataset.outputs.length === 0) {
      alert("Please upload a valid dataset first.");
      return;
    }

    if (isTraining && !isPaused) {
      console.log("Training already in progress, skipping...");
      return;
    }

    console.log(`Starting trainModel for ${numEpochs} epochs, sessionId: ${sessionId.current}`);
    trainingInitiated.current = true;
    setIsTraining(true);
    setIsTrained(true);
    setIsPaused(false);
    stopTrainingRef.current = false;
    cancelAnimationRef.current = false;
    trainingStateRef.current = { loss: null, lossHistory: [], outputs: [] };

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [inputNeurons], units: hiddenLayers[0], activation: activationFunction }));
    hiddenLayers.slice(1).forEach((units) =>
      model.add(tf.layers.dense({ units, activation: activationFunction }))
    );
    const outputActivation = problemType === "Classification" && outputNeurons > 1 ? "softmax" : "sigmoid";
    model.add(tf.layers.dense({ units: outputNeurons, activation: outputActivation }));

    const learningRate = parseFloat(localStorage.getItem("learningRate") || "0.1");
    const lossFunction = problemType === "Regression" ? "meanSquaredError" : (outputNeurons > 1 ? "categoricalCrossentropy" : "binaryCrossentropy");
    const metrics = problemType === "Regression" ? ["mae"] : ["accuracy"];
    console.log(`Compiling model with learningRate=${learningRate}, loss=${lossFunction}`);
    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: lossFunction,
      metrics,
    });

    const initialModelWeights = model.getWeights();
    const initialParsedWeights = await Promise.all(
      initialModelWeights.filter((_, i) => i % 2 === 0).map((w) => w.array())
    );

    const preprocessedInputs = preprocessData(dataset.inputs);
    const xs = tf.tensor2d(preprocessedInputs);
    const ys = tf.tensor2d(dataset.outputs);

    const numSamples = preprocessedInputs.length;
    const trainSize = Math.floor(numSamples * 0.8);
    const trainXs = xs.slice([0, 0], [trainSize, inputNeurons]);
    const trainYs = ys.slice([0, 0], [trainSize, outputNeurons]);
    const valXs = xs.slice([trainSize, 0], [numSamples - trainSize, inputNeurons]);
    const valYs = ys.slice([trainSize, 0], [numSamples - trainSize, outputNeurons]);

    setModel(model);
    setDebugWeights("Initial Weights: " + JSON.stringify(initialParsedWeights));
    setTrainingPhase("Initialization");

    const newLossHistory: { loss: number; accuracy: number; val_loss?: number }[] = [];
    setLossHistory(newLossHistory);
    setEpochDisplay(0);

    for (let epoch = 1; epoch <= numEpochs; epoch++) {
      if (stopTrainingRef.current) break;

      if (isPausedRef.current) {
        await new Promise<void>((resolve) => {
          pauseResolveRef.current = resolve;
        });
        pauseResolveRef.current = null;
        if (stopTrainingRef.current) break;
      }

      setTrainingPhase(`Epoch ${epoch}: Forward Pass`);
      const randomIndex = Math.floor(Math.random() * trainSize);
      const singleInput = trainXs.slice([randomIndex, 0], [1, inputNeurons]);
      const singleOutput = trainYs.slice([randomIndex, 0], [1, outputNeurons]);

      const preTrainWeights = await Promise.all(model.getWeights().filter((_, i) => i % 2 === 0).map(w => w.array()));
      const preTrainBiases = await Promise.all(model.getWeights().filter((_, i) => i % 2 !== 0).map(b => b.array()));

      const { neuronValuesMap } = await computeActivations(model, await singleInput.array());
      await playForward({ model, neuronValuesMap, parsedWeights: preTrainWeights, parsedBiases: preTrainBiases });

      if (cancelAnimationRef.current || stopTrainingRef.current) break;

      setTrainingPhase(`Epoch ${epoch}: Training`);
      const history = await model.fit(singleInput, singleOutput, { epochs: 1, verbose: 0 });
      const epochLoss = history.history.loss[0] as number;
      const epochAccuracy = (history.history.acc?.[0] || history.history.mae?.[0] || 0) as number;

      const valPred = model.predict(valXs) as tf.Tensor;
      const valLossTensor = tf.losses.meanSquaredError(valYs, valPred);
      const valLoss = await valLossTensor.dataSync()[0];

      newLossHistory.push({ loss: epochLoss, accuracy: epochAccuracy, val_loss: valLoss });
      setLossHistory([...newLossHistory]);
      setLoss(epochLoss.toFixed(4));

      const postTrainWeights = await Promise.all(model.getWeights().filter((_, i) => i % 2 === 0).map(w => w.array()));
      const postTrainBiases = await Promise.all(model.getWeights().filter((_, i) => i % 2 !== 0).map(b => b.array()));

      setDebugWeights(`Epoch ${epoch} Weights: ` + JSON.stringify(postTrainWeights));
      setTrainingPhase(`Epoch ${epoch}: Backward Pass`);

      await playBackward({ model, neuronValuesMap, preTrainWeights, preTrainBiases, postTrainWeights, postTrainBiases });

      if (cancelAnimationRef.current || stopTrainingRef.current) break;

      setEpochDisplay(epoch);
    }

    setTrainingPhase("");
    setDebugWeights("");
    setShowBackpropBanner(false);
    trainingInitiated.current = false;
    setIsTraining(false);
    setIsPaused(false);
    pauseResolveRef.current = null;
    xs.dispose();
    ys.dispose();
    trainXs.dispose();
    trainYs.dispose();
    valXs.dispose();
    valYs.dispose();
  };

  const playForward = async (epochLog: {
    model: tf.Sequential;
    neuronValuesMap: Map<string, number>;
    parsedWeights: number[][][];
    parsedBiases: number[][][];
  }) => {
    const { neuronValuesMap: valueMap, parsedWeights, parsedBiases } = epochLog;
    const positions = getNeuronPositions();
    const equationMap = new Map<string, string>();
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const newDisplayedConnections = new Set<string>();
    const newDisplayedWeights = new Map<string, number>(displayedWeights);

    let currentWeights: number[][][] = JSON.parse(JSON.stringify(parsedWeights));
    let currentBiases: number[][] = JSON.parse(JSON.stringify(parsedBiases));

    const layers = parsedWeights.map((layerWeights, layerIdx) => {
      const steps = (layerWeights as number[][]).flatMap((neuronWeights, fromIdx) =>
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
      if (layer.to > layerSizes.length - 1 || layer.from >= layerSizes.length - 1) {
        console.error(`Invalid layer indices: from=${layer.from}, to=${layer.to}`);
        continue;
      }

      const pulses = layer.steps.map((step: any) => {
        const fromIndex = getNeuronIndex(layer.from, step.fromIdx);
        const toIndex = getNeuronIndex(layer.to, step.toIdx);
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
          setDisplayedWeights(new Map(displayedWeights));
          return;
        }
        if (isPausedRef.current) {
          await new Promise<void>((resolve) => {
            pauseResolveRef.current = resolve;
          });
          pauseResolveRef.current = null;
        }
        await new Promise((resolve) => setTimeout(resolve, 16 / animationSpeedRef.current));
        progress += 0.025 * animationSpeedRef.current;
        const updatedPulses = pulses.map((p) => ({
          ...p,
          progress: Math.min(1, progress),
        }));
        const midY = pulses.length > 0 ? (pulses[0].from.y + pulses[0].to.y) / 2 : 0;
        if (graphPanelRef.current && pulses.length > 0) {
          const scrollTop = midY - graphPanelRef.current.clientHeight / 2;
          graphPanelRef.current.scrollTop = scrollTop;
        }
        setPulses(updatedPulses);
        setDisplayedConnections(new Set(newDisplayedConnections));

        if (progress >= 1) {
          for (const step of layer.steps) {
            currentWeights[layer.from][step.fromIdx][step.toIdx] = step.weight;
            currentBiases[layer.to - 1][step.toIdx] = step.bias;
            newDisplayedWeights.set(`${layer.from}-${step.fromIdx}-${step.toIdx}`, step.weight);

            const key = `${layer.to}-${step.toIdx}`;
            const weightsToNeuron = currentWeights[layer.from].map((row) => row[step.toIdx] ?? 0);
            const bias = step.bias ?? 0;
            const terms = weightsToNeuron.map((w, i) => {
              const inputLabel = layer.from === 0 ? `x${i + 1}` : `h${i + 1}`;
              return `${w.toFixed(2)}·${inputLabel}`;
            }).join(" + ");
            const equation = `z = ${terms} + ${bias.toFixed(2)}`;
            equationMap.set(key, equation);
          }
          setWeights([...currentWeights]);
          setBiases([...currentBiases]);
          setDisplayedWeights(new Map(newDisplayedWeights));
          setNeuronEquations(new Map(equationMap));
          setNeuronValues(new Map(valueMap));
        }
      }

      console.log("Forward pass weights:", currentWeights.map(w => w.map(row => row.map(val => val.toFixed(2)))));
      console.log("Forward pass biases:", currentBiases.map(b => b.map(val => val.toFixed(2))));
      setPulses([]);
      setDisplayedConnections(new Set(newDisplayedConnections));
      await new Promise((resolve) => setTimeout(resolve, 150 / animationSpeedRef.current));
    }
  };

  const playBackward = async (epochLog: {
    model: tf.Sequential;
    neuronValuesMap: Map<string, number>;
    preTrainWeights: number[][][];
    preTrainBiases: number[][][];
    postTrainWeights: number[][][];
    postTrainBiases: number[][][];
  }) => {
    const { neuronValuesMap: valueMap, postTrainWeights, postTrainBiases } = epochLog;
    const positions = getNeuronPositions();
    const gradientMap = new Map<string, number>();
    const equationMap = new Map<string, string>();
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const newDisplayedConnections = new Set<string>();
    const newDisplayedWeights = new Map<string, number>(displayedWeights);

    let currentWeights: number[][][] = JSON.parse(JSON.stringify(postTrainWeights));
    let currentBiases: number[][] = JSON.parse(JSON.stringify(postTrainBiases));

    const layers = postTrainWeights.map((layerWeights, layerIdx) => {
      const steps = (layerWeights as number[][]).flatMap((neuronWeights, fromIdx) =>
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
      if (layer.to > layerSizes.length - 1 || layer.from >= layerSizes.length - 1) {
        console.error(`Invalid layer indices: from=${layer.from}, to=${layer.to}`);
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
          setDisplayedWeights(new Map(displayedWeights));
          return;
        }
        if (isPausedRef.current) {
          await new Promise<void>((resolve) => {
            pauseResolveRef.current = resolve;
          });
          pauseResolveRef.current = null;
        }
        await new Promise((resolve) => setTimeout(resolve, 16 / animationSpeedRef.current));
        progress += 0.025 * animationSpeedRef.current;
        const updatedPulses = pulses.map((p) => ({
          ...p,
          progress: Math.min(1, progress),
        }));
        const midY = pulses.length > 0 ? (pulses[0].from.y + pulses[0].to.y) / 2 : 0;
        if (graphPanelRef.current && pulses.length > 0) {
          const scrollTop = midY - graphPanelRef.current.clientHeight / 2;
          graphPanelRef.current.scrollTop = scrollTop;
        }
        setPulses(updatedPulses);
        setDisplayedConnections(new Set(newDisplayedConnections));

        if (progress >= 1) {
          for (const step of layer.steps) {
            currentWeights[layer.from][step.fromIdx][step.toIdx] = step.weight;
            currentBiases[layer.to - 1][step.toIdx] = step.bias;
            newDisplayedWeights.set(`${layer.from}-${step.fromIdx}-${step.toIdx}`, step.weight);

            const gradientKey = `${layer.from}-${step.fromIdx}`;
            const prev = gradientMap.get(gradientKey) || 0;
            const prevWeight = epochLog.preTrainWeights[layer.from]?.[step.fromIdx]?.[step.toIdx] ?? step.weight;
            gradientMap.set(gradientKey, prev + Math.abs(step.weight - prevWeight));

            const equationKey = `${layer.to}-${step.toIdx}`;
            const weightsToNeuron = currentWeights[layer.from].map((row) => row[step.toIdx] ?? 0);
            const bias = step.bias ?? 0;
            const terms = weightsToNeuron.map((w, i) => {
              const inputLabel = layer.from === 0 ? `x${i + 1}` : `h${i + 1}`;
              return `${w.toFixed(2)}·${inputLabel}`;
            }).join(" + ");
            const equation = `z = ${terms} + ${bias.toFixed(2)}`;
            equationMap.set(equationKey, equation);
          }
          setWeights([...currentWeights]);
          setBiases([...currentBiases]);
          setDisplayedWeights(new Map(newDisplayedWeights));
          setNeuronEquations(new Map(equationMap));
          setNeuronValues(new Map(valueMap));
          setNeuronGradients(new Map(gradientMap));
        }
      }

      console.log(`Backward pass weights for layer ${layer.from} -> ${layer.to}:`, 
        currentWeights[layer.from].map(row => row.map(val => val.toFixed(2))));
      console.log(`Backward pass biases for layer ${layer.to - 1}:`, 
        currentBiases[layer.to - 1].map(val => val.toFixed(2)));
      setPulses([]);
      setDisplayedConnections(new Set(newDisplayedConnections));
      await new Promise((resolve) => setTimeout(resolve, 150 / animationSpeedRef.current));
    }
  };

  const predict = async (inputs: number[]) => {
    if (!model) {
      alert("Please train the model first.");
      return;
    }
    if (inputs.length !== inputNeurons) {
      alert(`Expected ${inputNeurons} inputs, but received ${inputs.length}.`);
      return;
    }
    if (inputs.some((val) => isNaN(val))) {
      alert("All inputs must be valid numbers.");
      return;
    }
    setCurrentInputs(inputs.map(val => parseFloat(val.toFixed(3))));
    const xs = tf.tensor2d([inputs]);
    const pred = model.predict(xs) as tf.Tensor;
    const predArray = await pred.array();
    setOutputs(predArray.flatMap((v: any) => v.map((val: number) => parseFloat(val.toFixed(3)))));
    xs.dispose();

    const modelWeights = model.getWeights();
    const parsedWeights = await Promise.all(
      modelWeights.filter((_, i) => i % 2 === 0).map((w) => w.array())
    );
    const parsedBiases = await Promise.all(
      modelWeights.filter((_, i) => i % 2 !== 0).map((b) => b.array())
    );

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
    const { neuronValuesMap } = await computeActivations(model, [inputs]);
    await playForward({ model, neuronValuesMap, parsedWeights, parsedBiases });
    setTrainingPhase("");
  };

  const onDatasetUpload = (data: { inputs: number[][]; outputs: number[][] }) => {
    if (
      data.inputs.length > 0 &&
      data.outputs.length > 0 &&
      data.inputs.length === data.outputs.length &&
      data.inputs.every((row) => row.length === data.inputs[0].length) &&
      data.outputs.every((row) => row.length === data.outputs[0].length)
    ) {
      setDataset(data);
    }
  };

  const handlePlay = async (epochs: number) => {
    if (isPaused) {
      setIsPaused(false);
      if (pauseResolveRef.current) {
        pauseResolveRef.current();
        pauseResolveRef.current = null;
      }
    } else {
      await trainModel(epochs);
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
    setShowBackpropBanner(false);
    setDisplayedConnections(new Set());
    setDisplayedWeights(new Map());
    setCurrentInputs([]);
    setDebugWeights("");
    setTrainingPhase("");
    console.log("Stop training triggered, clearing visualizations");
  };

  return (
    <div style={{ padding: "20px", backgroundColor: "#ffffff", minHeight: "100vh" }}>
      <h1 style={{ textAlign: "center", fontSize: "24px", color: "#333" }}>
        Neural Network Visualizer
      </h1>
      <div style={{ display: "flex", gap: "20px" }}>
        <div className="panel" style={{ flex: "0 0 280px" }}>
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
            onDatasetUpload={onDatasetUpload}
            lineThicknessMode={lineThicknessMode}
            setLineThicknessMode={setLineThicknessMode}
            zoomLevel={zoomLevel}
            setZoomLevel={setZoomLevel}
            onPlay={handlePlay}
            onPause={handlePause}
            isTraining={isTraining}
            isPaused={isPaused}
            animationSpeed={animationSpeed}
            setAnimationSpeed={setAnimationSpeed}
          />
          <button
            onClick={handleStop}
            disabled={!isTraining}
            style={{ marginTop: "10px" }}
          >
            Stop Training
          </button>
          {loss && <p style={{ marginTop: "10px", color: "#27ae60" }}>Loss: {loss}</p>}
          <LossChart lossHistory={lossHistory} />
          {outputs.length > 0 && (
            <div style={{ marginTop: "10px" }}>
              <p style={{ color: "#27ae60" }}>
                Predictions: {outputs.join(", ")}
              </p>
            </div>
          )}
        </div>
        <div
          className="graph-panel"
          ref={graphPanelRef}
          style={{
            flex: 1,
            position: "relative",
            overflow: "auto",
            backgroundColor: "#ffffff",
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
              displayedWeights={displayedWeights}
              activationFunction={activationFunction}
              pulses={pulses}
              displayedConnections={displayedConnections}
              neuronEquations={neuronEquations}
              neuronValues={neuronValues}
              neuronGradients={neuronGradients}
              showWeights={showWeights}
              lineThicknessMode={lineThicknessMode}
              zoomLevel={zoomLevel}
              epochDisplay={epochDisplay}
              problemType={problemType}
              currentInputs={currentInputs}
              outputs={outputs}
              hasDataset={dataset.inputs.length > 0 && dataset.outputs.length > 0}
              isTrained={isTrained}
            />
          ) : (
            <div style={{ textAlign: "center", marginTop: "20px", color: "#666" }}>
              <p>Error: Unable to render the neural network.</p>
            </div>
          )}
          {(debugWeights || trainingPhase) && (
            <DebugWeightsDisplay debugWeights={debugWeights} trainingPhase={trainingPhase} />
          )}
        </div>
      </div>
    </div>
  );
});

export default NeuralNetworkVisualizer;