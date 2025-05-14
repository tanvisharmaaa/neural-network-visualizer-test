import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import NetworkControls from "./NetworkControls";
import NetworkGraph from "./NetworkGraph";
import LossChart from "./LossChart";

const NeuralNetworkVisualizer = () => {
  const [inputNeurons, setInputNeurons] = useState(2);
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([2]);
  const [outputNeurons, setOutputNeurons] = useState(1);
  const [weights, setWeights] = useState<number[][][]>([]);
  const [biases, setBiases] = useState<number[][]>([]);
  const [activationFunction, setActivationFunction] = useState("sigmoid");
  const [problemType, setProblemType] = useState("Classification");
  const [loss, setLoss] = useState<string | null>(null);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [outputs, setOutputs] = useState<number[]>([]);
  const [pulses, setPulses] = useState<any[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [neuronEquations, setNeuronEquations] = useState<Map<string, string>>(new Map());
  const [neuronValues, setNeuronValues] = useState<Map<string, number>>(new Map());
  const [neuronGradients, setNeuronGradients] = useState<Map<string, number>>(new Map());
  const [epochDisplay, setEpochDisplay] = useState(0);
  const [showBackpropBanner, setShowBackpropBanner] = useState(false);
  const [showWeights, setShowWeights] = useState(true);

  const xorData = {
    inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
    outputs: [[0], [1], [1], [0]],
  };

  const getNeuronPositions = () => {
    const layerSizes = [inputNeurons, ...hiddenLayers, outputNeurons];
    return layerSizes.map((count, layerIdx) => {
      const totalHeight = (count - 1) * 70;
      return Array.from({ length: count }, (_, neuronIdx) => ({
        x: layerIdx * 120 + 50,
        y: neuronIdx * 70 - totalHeight / 2 + 200,
      }));
    });
  };

  const trainModel = async () => {
    setIsTraining(true);
    setLoss(null);
    setOutputs([]);
    setWeights([]);
    setBiases([]);
    setLossHistory([]);
    setNeuronEquations(new Map());
    setNeuronValues(new Map());
    setNeuronGradients(new Map());

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [inputNeurons], units: hiddenLayers[0], activation: activationFunction }));
    hiddenLayers.slice(1).forEach((units) =>
      model.add(tf.layers.dense({ units, activation: "sigmoid" }))
    );
    model.add(tf.layers.dense({ units: outputNeurons, activation: outputNeurons === 1 ? "sigmoid" : "softmax" }));

    model.compile({ optimizer: tf.train.adam(0.1), loss: "binaryCrossentropy", metrics: ["accuracy"] });

    const xs = tf.tensor2d(xorData.inputs);
    const ys = tf.tensor2d(xorData.outputs);

    const logs: any[] = [];
    const numEpochs = parseInt(localStorage.getItem("numEpochs") || "3");

    for (let epoch = 0; epoch < numEpochs; epoch++) {
      setEpochDisplay(epoch + 1);
      await model.fit(xs, ys, {
        epochs: 1,
        callbacks: {
          onEpochEnd: async (_, logsData) => {
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

            logs.push({ epoch, layers });
            const currentLoss = logsData?.loss ?? 0;
            setLoss(currentLoss.toFixed(4));
            setLossHistory((prev) => [...prev, parseFloat(currentLoss.toFixed(4))]);

            const pred = model.predict(xs) as tf.Tensor;
            const predArray = await pred.array();
            setOutputs(predArray.map((v: any) => parseFloat(v[0].toFixed(3))));
          },
        },
      });

      const epochLog = logs[logs.length - 1];
      await playForward(epochLog);
      setShowBackpropBanner(true);
      await playBackward(epochLog);
      setShowBackpropBanner(false);
    }

    setIsTraining(false);
    xs.dispose();
    ys.dispose();
  };

  const playForward = async (epochLog: any) => {
    const positions = getNeuronPositions();
    const equationMap = new Map<string, string>();
    const valueMap = new Map<string, number>();
    const allWeights: number[][][] = Array(hiddenLayers.length + 1).fill(null).map(() => []);
    const allBiases: number[][] = Array(hiddenLayers.length + 1).fill(null).map(() => []);

    for (const layer of epochLog.layers) {
      const pulses = layer.steps.map((step: any) => ({
        from: positions[layer.from][step.fromIdx],
        to: positions[layer.to][step.toIdx],
        progress: 0,
        direction: "forward" as const,
      }));

      setPulses(pulses);
      let progress = 0;
      while (progress < 1) {
        await new Promise((res) => setTimeout(res, 16));
        progress += 0.025;
        setPulses(pulses.map((p) => ({ ...p, progress: Math.min(1, progress) })));
      }

      for (const step of layer.steps) {
        if (!allWeights[layer.from][step.fromIdx]) allWeights[layer.from][step.fromIdx] = [];
        allWeights[layer.from][step.fromIdx][step.toIdx] = step.weight;

        if (!allBiases[layer.to]) allBiases[layer.to] = [];
        allBiases[layer.to][step.toIdx] = step.bias;
      }

      setWeights([...allWeights]);
      setBiases([...allBiases]);

      for (const step of layer.steps) {
        const key = `${layer.to}-${step.toIdx}`;
        const weightsToNeuron = allWeights[layer.from].map((row) => row?.[step.toIdx] ?? 0);
        const bias = step.bias ?? 0;
        const terms = weightsToNeuron.map((w, i) => `${w.toFixed(2)}·x${i + 1}`).join(" + ");
        const equation = `z = ${terms} + ${bias.toFixed(2)}`;
        equationMap.set(key, equation);

        const z = weightsToNeuron.reduce((acc, w) => acc + w, 0) + bias;
        valueMap.set(key, z);
      }

      setNeuronEquations(new Map(equationMap));
      setNeuronValues(new Map(valueMap));
      setPulses([]);
      await new Promise((res) => setTimeout(res, 300));
    }
  };

  const playBackward = async (epochLog: any) => {
    const positions = getNeuronPositions();
    const gradientMap = new Map<string, number>();

    for (const layer of [...epochLog.layers].reverse()) {
      const pulses = layer.steps.map((step: any) => ({
        from: positions[layer.to][step.toIdx],
        to: positions[layer.from][step.fromIdx],
        progress: 0,
        direction: "backward" as const,
      }));

      setPulses(pulses);
      let progress = 0;
      while (progress < 1) {
        await new Promise((res) => setTimeout(res, 16));
        progress += 0.025;
        setPulses(pulses.map((p) => ({ ...p, progress: Math.min(1, progress) })));
      }

      // Approximate ∂L/∂z with |weight| for visual cue
      for (const step of layer.steps) {
        const key = `${layer.from}-${step.fromIdx}`;
        const prev = gradientMap.get(key) || 0;
        gradientMap.set(key, prev + Math.abs(step.weight));
      }

      setPulses([]);
      await new Promise((res) => setTimeout(res, 300));
    }

    setNeuronGradients(gradientMap);
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
          />
          <button onClick={trainModel} disabled={isTraining} style={{ marginTop: "20px" }}>
            {isTraining ? "Training..." : "Train"}
          </button>
          {loss && <p style={{ marginTop: "10px", color: "#27ae60" }}>Loss: {loss}</p>}
          <LossChart lossHistory={lossHistory} />
        </div>

        <div className="graph-panel" style={{ flex: 1, position: "relative" }}>
          <NetworkGraph
            inputNeurons={inputNeurons}
            hiddenLayers={hiddenLayers}
            outputNeurons={outputNeurons}
            weights={weights}
            biases={biases}
            activationFunction={activationFunction}
            pulses={pulses}
            neuronEquations={neuronEquations}
            neuronValues={neuronValues}
            neuronGradients={neuronGradients}
            showWeights={showWeights}
          />
        </div>
      </div>
    </div>
  );
};

export default NeuralNetworkVisualizer;
