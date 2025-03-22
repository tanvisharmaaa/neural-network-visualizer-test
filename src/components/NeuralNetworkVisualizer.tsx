import React, { useState, useEffect } from "react"
import * as tf from "@tensorflow/tfjs"
import NetworkControls from "./NetworkControls"
import NetworkGraph from "./NetworkGraph"

const NeuralNetworkVisualizer = () => {
  const [inputNeurons, setInputNeurons] = useState(2)
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([2])
  const [outputNeurons, setOutputNeurons] = useState(1)
  const [weights, setWeights] = useState<number[][][]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [loss, setLoss] = useState<string | null>(null)
  const [epoch, setEpoch] = useState(0)
  const [outputs, setOutputs] = useState<number[]>([])
  const [activationFunction, setActivationFunction] = useState("sigmoid")
  const [problemType, setProblemType] = useState("Classification")
  const [biases, setBiases] = useState<number[][]>([])



  const xorData = {
    inputs: [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ],
    outputs: [[0], [1], [1], [0]],
  }

  const trainModel = async () => {
    setIsTraining(true)
    setLoss(null)
    setEpoch(0)

    const model = tf.sequential()

    model.add(
      tf.layers.dense({
        inputShape: [inputNeurons],
        units: hiddenLayers[0],
        activation: activationFunction
,
      })
    )

    hiddenLayers.slice(1).forEach((units) => {
      model.add(tf.layers.dense({ units, activation: "sigmoid" }))
    })

    model.add(
      tf.layers.dense({
        units: outputNeurons,
        activation: outputNeurons === 1 ? "sigmoid" : "softmax",
      })
    )

    model.compile({
      optimizer: tf.train.adam(0.1),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    })

    const xs = tf.tensor2d(xorData.inputs)
    const ys = tf.tensor2d(xorData.outputs)

    await model.fit(xs, ys, {
      epochs: 30,
      callbacks: {
        onEpochEnd: async (ep, logs) => {
          setEpoch(ep + 1)
          setLoss(logs?.loss?.toFixed(4) ?? null)

          const modelWeights = model.getWeights()

          // Separate weights and biases
          const parsedWeights = await Promise.all(
            modelWeights.filter((_, i) => i % 2 === 0).map((w) => w.array())
          )
          const parsedBiases = await Promise.all(
            modelWeights.filter((_, i) => i % 2 !== 0).map((b) => b.array())
          )
          
          setWeights(parsedWeights as number[][][])
          setBiases(parsedBiases as number[][])

          // Predict outputs
          const pred = model.predict(xs) as tf.Tensor
          const predArray = await pred.array()
          setOutputs(predArray.map((v: any) => parseFloat(v[0].toFixed(3))))
        },
      },
    })

    xs.dispose()
    ys.dispose()
    setIsTraining(false)
  }

  useEffect(() => {
    const dummyWeights = hiddenLayers.length === 1
      ? [
          [
            [0.5, -0.2],
            [0.8, 0.3],
          ],
          [
            [0.1],
            [-0.6],
          ],
        ]
      : []
    setWeights(dummyWeights)
  }, [inputNeurons, hiddenLayers, outputNeurons])

  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>Neural Network Visualizer</h1>
      <div style={styles.panels}>
        <div style={styles.leftPanel}>
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
/>

          <button onClick={trainModel} disabled={isTraining} style={styles.trainBtn}>
            {isTraining ? "Training..." : "Train"}
          </button>

          {loss && <p style={styles.loss}>Loss: {loss}</p>}

          {outputs.length > 0 && (
            <div style={styles.outputBox}>
              <h3>Network Output</h3>
              {xorData.inputs.map((inp, i) => (
                <div key={i}>
                  [{inp.join(", ")}] → <strong>{outputs[i]}</strong>
                </div>
              ))}
              <p style={{ marginTop: 10 }}>Epoch: {epoch}</p>
            </div>
          )}
        </div>

        <div style={styles.rightPanel}>
        <NetworkGraph
          inputNeurons={inputNeurons}
          hiddenLayers={hiddenLayers}
          outputNeurons={outputNeurons}
          weights={weights}
          biases={biases}
/>
        </div>
      </div>
    </div>
  )
}

const styles: { [key: string]: React.CSSProperties } = {
  container: { padding: "20px" },
  heading: {
    textAlign: "center",
    fontSize: "24px",
    marginBottom: "20px",
  },
  panels: {
    display: "flex",
    gap: "20px",
  },
  leftPanel: {
    flex: 1,
    backgroundColor: "#1e1e2f",
    padding: "20px",
    borderRadius: "8px",
  },
  rightPanel: {
    flex: 2,
    backgroundColor: "#1e1e2f",
    padding: "20px",
    borderRadius: "8px",
  },
  trainBtn: {
    marginTop: "16px",
    padding: "10px 20px",
    fontWeight: "bold",
    backgroundColor: "#3498db",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },
  loss: {
    marginTop: "10px",
    color: "#2ecc71",
    fontWeight: "bold",
  },
  outputBox: {
    marginTop: "16px",
    padding: "10px",
    backgroundColor: "#2c2c3e",
    borderRadius: 6,
    fontSize: "14px",
  },
}

export default NeuralNetworkVisualizer
