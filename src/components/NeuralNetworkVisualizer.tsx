import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import NetworkControls from "./NetworkControls";
import NetworkGraph from "./NetworkGraph";
import LossChart from "./LossChart";

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

  const graphPanelRef = useRef<HTMLDivElement>(null);
  const trainingInitiated = useRef(false);
  const stopTrainingRef = useRef(false);
  const cancelAnimationRef = useRef(false);
  const trainingStateRef = useRef({
    loss: null as string | null,
    lossHistory: [] as { loss: number; accuracy: number; val_loss?: number }[],
    outputs: [] as number[],
  });
  const sessionId = useRef(Date.now().toString());

  useEffect(() => {
    console.log("lossHistory updated:", lossHistory);
  }, [lossHistory]);

  useEffect(() => {
    setWeights(initializeWeights(inputNeurons, hiddenLayers, outputNeurons));
    setBiases(initializeBiases(hiddenLayers, outputNeurons));
  }, []);

  useEffect(() => {
    if (dataset.inputs.length > 0 && dataset.outputs.length > 0) {
      const newInputNeurons = dataset.inputs[0].length;
      const newOutputNeurons = dataset.outputs[0].length;
      setInputNeurons(newInputNeurons);
      setOutputNeurons(newOutputNeurons);
      setWeights(initializeWeights(newInputNeurons, hiddenLayers, newOutputNeurons));
      setBiases(initializeBiases(hiddenLayers, newOutputNeurons));
      setLossHistory([]);
      setNeuronEquations(new Map());
      setNeuronValues(new Map());
      setNeuronGradients(new Map());
      setOutputs([]);
      setCurrentInputs([]);
      setPulses([]);
      setDisplayedConnections(new Set());
      setModel(null);
    }
  }, [dataset, hiddenLayers]);

  const initializeWeights = (inputNeurons: number, hiddenLayers: number[], outputNeurons: number) => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const weights: number[][][] = [];
    for (let layerIdx = 0; layerIdx < layerSizes.length - 1; layerIdx++) {
      const fromSize = layerSizes[layerIdx];
      const toSize = layerSizes[layerIdx + 1];
      const layerWeights: number[][] = [];
      for (let fromIdx = 0; fromIdx < fromSize; fromIdx++) {
        const neuronWeights: number[] = [];
        for (let toIdx = 0; toIdx < toSize; toIdx++) {
          neuronWeights.push(Math.random() * 2 - 1);
        }
        layerWeights.push(neuronWeights);
      }
      weights.push(layerWeights);
    }
    return weights;
  };

  const initializeBiases = (hiddenLayers: number[], outputNeurons: number) => {
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

    return { activations, neuronValuesMap };
  };

  const trainModel = async (numEpochs: number) => {
    console.log(`trainModel called with numEpochs: ${numEpochs}`);
    if (dataset.inputs.length === 0 || dataset.outputs.length === 0) {
      alert("Please upload a valid dataset first.");
      return;
    }

    if (isTraining || trainingInitiated.current) {
      console.log("Training already in progress, skipping...");
      return;
    }

    console.log(`Starting trainModel for ${numEpochs} epochs, sessionId: ${sessionId.current}`);
    trainingInitiated.current = true;
    setIsTraining(true);
    stopTrainingRef.current = false;
    cancelAnimationRef.current = false;
    trainingStateRef.current = { loss: null, lossHistory: [], outputs: [] };

    const initialWeights = initializeWeights(inputNeurons, hiddenLayers, outputNeurons);
    const initialBiases = initializeBiases(hiddenLayers, outputNeurons);
    setWeights(initialWeights);
    setBiases(initialBiases);
    setNeuronEquations(new Map());
    setNeuronValues(new Map());
    setNeuronGradients(new Map());
    setEpochDisplay(0);
    setLossHistory([]);
    setPulses([]);
    setDisplayedConnections(new Set());
    setCurrentInputs([]);

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

    const normalizedInputs = preprocessData(dataset.inputs);
    const xs = tf.tensor2d(normalizedInputs);
    const ys = tf.tensor2d(dataset.outputs);

    let epochCount = 0;

    try {
      for (let epoch = 0; epoch < numEpochs; epoch++) {
        epochCount++;
        if (stopTrainingRef.current) {
          console.log("Training stopped by user");
          break;
        }
        console.log(`Starting Epoch ${epoch + 1} of ${numEpochs}`);
        setEpochDisplay(epoch + 1);
        setDisplayedConnections(new Set());

        // Store weights before training
        const preTrainWeights = model.getWeights();
        const preTrainParsedWeights = await Promise.all(
          preTrainWeights.filter((_, idx) => idx % 2 === 0).map((w) => w.array())
        );
        const preTrainParsedBiases = await Promise.all(
          preTrainWeights.filter((_, idx) => idx % 2 !== 0).map((b) => b.array())
        );

        // Train the model
        const batchSize = Math.min(32, normalizedInputs.length);
        const history = await model.fit(xs, ys, {
          epochs: 1,
          batchSize: batchSize,
          shuffle: true,
          verbose: 0,
        });

        const currentLoss = history.history.loss[0] as number;
        const currentAccuracy = (history.history as any).acc ? (history.history as any).acc[0] : 0;
        console.log(`Epoch ${epoch + 1} - Loss: ${currentLoss}, Accuracy: ${currentAccuracy}`);

        // Fetch weights after training
        const postTrainWeights = model.getWeights();
        const postTrainParsedWeights = await Promise.all(
          postTrainWeights.filter((_, idx) => idx % 2 === 0).map((w) => w.array())
        );
        const postTrainParsedBiases = await Promise.all(
          postTrainWeights.filter((_, idx) => idx % 2 !== 0).map((b) => b.array())
        );

        // Log weight differences
        console.log("Weight differences after training:");
        preTrainParsedWeights.forEach((layerWeights, layerIdx) => {
          const postLayerWeights = postTrainParsedWeights[layerIdx] as number[][];
          const diffs = layerWeights.map((neuronWeights, fromIdx) =>
            neuronWeights.map((weight, toIdx) => {
              const newWeight = postLayerWeights[fromIdx][toIdx];
              return Math.abs(newWeight - weight);
            })
          );
          console.log(`Layer ${layerIdx} weight diffs:`, diffs.map(row => row.map(val => val.toFixed(4))));
        });

        trainingStateRef.current = {
          loss: currentLoss.toFixed(4),
          lossHistory: [
            ...trainingStateRef.current.lossHistory,
            { loss: parseFloat(currentLoss.toFixed(4)), accuracy: parseFloat(currentAccuracy.toFixed(4)) },
          ],
          outputs: trainingStateRef.current.outputs,
        };
        setLoss(trainingStateRef.current.loss);
        setLossHistory(trainingStateRef.current.lossHistory);

        console.log(`Starting forward visualization for Epoch ${epoch + 1}`);
        const { neuronValuesMap: forwardValues } = await computeActivations(model, [normalizedInputs[0]]);
        await playForward({ model, neuronValuesMap: forwardValues, parsedWeights: postTrainParsedWeights, parsedBiases: postTrainParsedBiases });
        if (stopTrainingRef.current) break;

        console.log(`Starting backward visualization for Epoch ${epoch + 1}`);
        const { neuronValuesMap: backwardValues } = await computeActivations(model, [normalizedInputs[0]]);
        setShowBackpropBanner(true);
        await playBackward({
          model,
          neuronValuesMap: backwardValues,
          preTrainWeights: preTrainParsedWeights as number[][][],
          preTrainBiases: postTrainParsedBiases as number[][],
          postTrainWeights: postTrainParsedWeights as number[][][],
          postTrainBiases: postTrainParsedBiases as number[][],
        });
        if (stopTrainingRef.current) break;
        setShowBackpropBanner(false);

        const pred = model.predict(xs) as tf.Tensor;
        const predArray = await pred.array();
        setOutputs(predArray.flatMap((v: any) => v.map((val: number) => parseFloat(val.toFixed(3)))));
        pred.dispose();
        if (stopTrainingRef.current) break;
        console.log(`Completed Epoch ${epoch + 1} of ${numEpochs}`);
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
    } catch (error) {
      console.error("Training failed:", error);
    } finally {
      setModel(model);
      if (epochCount >= numEpochs || stopTrainingRef.current) {
        setIsTraining(false);
        trainingInitiated.current = false;
        cancelAnimationRef.current = true;
      }
      setPulses([]);
      setShowBackpropBanner(false);
      setNeuronValues(new Map());
      setNeuronGradients(new Map());
      setCurrentInputs([]);
      setDisplayedConnections(new Set());
      xs.dispose();
      ys.dispose();
      console.log("Training completed, sessionId:", sessionId.current);
    }
  };

  const playForward = async (epochLog: { model: tf.Sequential; neuronValuesMap: Map<string, number>; parsedWeights: any[]; parsedBiases: any[] }) => {
    const { model, neuronValuesMap: valueMap, parsedWeights, parsedBiases } = epochLog;
    const positions = getNeuronPositions();
    const equationMap = new Map<string, string>();
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const newDisplayedConnections = new Set<string>();

    // Initialize weights with current model weights
    let currentWeights: number[][][] = parsedWeights.map(layer => layer.map((row: number[]) => [...row]));
    let currentBiases: number[][] = parsedBiases.map((b: number[]) => [...b]);

    // Generate layers for visualization
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
        newDisplayedConnections.add(`${layer.from}-${step.fromIdx}-${step.toIdx}`);
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
        };
      });

      let progress = 0;
      while (progress < 1) {
        if (cancelAnimationRef.current) {
          console.log("Forward animation cancelled");
          setPulses([]);
          setDisplayedConnections(new Set());
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 16));
        progress += 0.025;
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
      }

      // Update weights and biases only when pulses reach the destination
      for (const step of layer.steps) {
        currentWeights[layer.from][step.fromIdx][step.toIdx] = step.weight;
        currentBiases[layer.to - 1][step.toIdx] = step.bias;

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

      console.log("Forward pass weights:", currentWeights.map(w => w.map(row => row.map(val => val.toFixed(2)))));
      console.log("Forward pass biases:", currentBiases.map(b => b.map(val => val.toFixed(2))));
      setWeights([...currentWeights]);
      setBiases([...currentBiases]);
      setNeuronEquations(new Map(equationMap));
      setNeuronValues(new Map(valueMap));
      setPulses([]);
      setDisplayedConnections(new Set(newDisplayedConnections));
      await new Promise((resolve) => setTimeout(resolve, 150));
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
    const { model, neuronValuesMap: valueMap, preTrainWeights, preTrainBiases, postTrainWeights, postTrainBiases } = epochLog;
    const positions = getNeuronPositions();
    const gradientMap = new Map<string, number>();
    const equationMap = new Map<string, string>();
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    const newDisplayedConnections = new Set<string>();

    // Initialize with pre-training weights
    let currentWeights: number[][][] = JSON.parse(JSON.stringify(preTrainWeights));
    let currentBiases: number[][] = JSON.parse(JSON.stringify(preTrainBiases));
    setWeights([...currentWeights]);
    setBiases([...currentBiases]);

    // Generate layers for visualization
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
        newDisplayedConnections.add(`${layer.from}-${step.fromIdx}-${step.toIdx}`);
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
        };
      });

      let progress = 0;
      while (progress < 1) {
        if (cancelAnimationRef.current) {
          console.log("Backward animation cancelled");
          setPulses([]);
          setDisplayedConnections(new Set());
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 16));
        progress += 0.025;
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
      }

      // Update weights and biases when pulses reach the destination
      for (const step of layer.steps) {
        currentWeights[layer.from][step.fromIdx][step.toIdx] = step.weight;
        currentBiases[layer.to - 1][step.toIdx] = step.bias;

        const gradientKey = `${layer.from}-${step.fromIdx}`;
        const prev = gradientMap.get(gradientKey) || 0;
        const prevWeight = preTrainWeights[layer.from]?.[step.fromIdx]?.[step.toIdx] ?? step.weight;
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

      console.log(`Backward pass weights for layer ${layer.from} -> ${layer.to}:`, 
        currentWeights[layer.from].map(row => row.map(val => val.toFixed(2))));
      console.log(`Backward pass biases for layer ${layer.to - 1}:`, 
        currentBiases[layer.to - 1].map(val => val.toFixed(2)));
      setWeights([...currentWeights]);
      setBiases([...currentBiases]);
      setNeuronEquations(new Map(equationMap));
      setNeuronValues(new Map(valueMap));
      setNeuronGradients(new Map(gradientMap));
      setPulses([]);
      setDisplayedConnections(new Set(newDisplayedConnections));
      await new Promise((resolve) => setTimeout(resolve, 150));
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

    const modelWeights = model.getWeights();
    const parsedWeights = await Promise.all(
      modelWeights.filter((_, i) => i % 2 === 0).map((w) => w.array())
    );
    const parsedBiases = await Promise.all(
      modelWeights.filter((_, i) => i % 2 !== 0).map((b) => b.array())
    );

    const layers = parsedWeights.map((layerWeights, layerIdx) => {
      const steps = (layerWeights as number[][]).flatMap((neuronWeights, fromIdx) =>
        neuronWeights.map((weight, toIdx) => ({
          fromIdx,
          toIdx,
          weight,
          bias: parsedBiases[layerIdx]?.[toIdx],
        }))
      );
      return { from: layerIdx, to: layerIdx + 1, steps };
    });

    cancelAnimationRef.current = false;
    const { neuronValuesMap } = await computeActivations(model, [inputs]);
    await playForward({ model, neuronValuesMap, parsedWeights, parsedBiases });
    xs.dispose();
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

  const handleTrain = async (epochs: number) => {
    stopTrainingRef.current = false;
    cancelAnimationRef.current = false;
    await trainModel(epochs);
  };

  const handleStop = () => {
    stopTrainingRef.current = true;
    cancelAnimationRef.current = true;
    setPulses([]);
    setShowBackpropBanner(false);
    setDisplayedConnections(new Set());
    setCurrentInputs([]);
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
            onTrain={handleTrain}
            isTraining={isTraining}
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
          {inputNeurons > 0 && outputNeurons > 0 && weights.length > 0 && biases.length > 0 ? (
            <NetworkGraph
              key={`${inputNeurons}-${outputNeurons}`}
              inputNeurons={inputNeurons}
              hiddenLayers={hiddenLayers}
              outputNeurons={outputNeurons}
              weights={weights}
              biases={biases}
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
            />
          ) : (
            <div style={{ textAlign: "center", marginTop: "20px", color: "#666" }}>
              <p>Error: Unable to render the neural network.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

export default NeuralNetworkVisualizer;