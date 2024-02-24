using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FaNNtasyRaceResultPredictions
{
    /// <summary>
    /// Backpropagation NeuralNetwork
    /// </summary>
    public class NeuralNetwork
    {
        //store layer information
        int[] layer;
        //store all layers from network
        Layer[] layers;
        //set learning rate for network
        public static float learningRate = initLearningRate;
        public static float initLearningRate = 0.008f;

        /// <summary>
        /// Constructor for NeuralNetwork
        /// </summary>
        /// <param name="layer">Layers of this network</param>
        public NeuralNetwork(int[] layer)
        {
            //deep copy layers
            this.layer = new int[layer.Length];
            for (int i = 0; i < layer.Length; i++)
                this.layer[i] = layer[i];

            //creates neural layers
            layers = new Layer[layer.Length - 1];

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(layer[i], layer[i + 1]);
            }
        }

        /// <summary>
        /// High level feedforward for this network
        /// </summary>
        /// <param name="inputs">Inputs to be feed forwared</param>
        /// <returns></returns>
        public float[] FeedForward(float[] inputs)
        {
            //feed forward
            layers[0].FeedForwardLayer(inputs);

            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForwardLayer(layers[i - 1].outputs);
            }

            return layers[layers.Length - 1].outputs; //return output of last layer
        }

        /// <summary>
        /// High level back porpagation
        /// </summary>
        /// <param name="expected">The expected output form the last feedforward</param>
        public void BackProp(float[] expected)
        {
            // run over all layers backwards
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                if (i == layers.Length - 1)
                {
                    layers[i].BackPropOutput(expected); //back prop output
                }
                else
                {
                    layers[i].BackPropHidden(layers[i + 1].gamma, layers[i + 1].weights); //back prop hidden
                }
            }

            //Update weights
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].UpdateWeights();
            }
        }

        /// <summary>
        /// Lower LearningRate by [difference]
        /// </summary>
        public void LowerLearningRate(float difference)
        {
            learningRate -= difference;
        }

        public void ResetLearningRate()
        {
            learningRate = initLearningRate;
        }

        /// <summary>
        /// Individual layer from network
        /// </summary>
        public class Layer
        {
            //number of neurons in previous
            //and following layer
            int numberOfInputs;
            int numberOfOutputs;

            public float[] outputs; //outputs of this layer
            public float[] inputs; //inputs in into this layer
            public float[,] weights; //weights of this layer
            public float[,] weightsDelta; //deltas of this layer
            public float[] gamma; //gamma of this layer
            public float[] error; //error of the output layer

            //generate random number
            public static Random random = new Random();

            public Layer(int numberOfInputs, int numberOfOutputs)
            {
                this.numberOfInputs = numberOfInputs;
                this.numberOfOutputs = numberOfOutputs;

                //initialize datastructures
                outputs = new float[numberOfOutputs];
                inputs = new float[numberOfInputs];
                weights = new float[numberOfOutputs, numberOfInputs];
                weightsDelta = new float[numberOfOutputs, numberOfInputs];
                gamma = new float[numberOfOutputs];
                error = new float[numberOfOutputs];

                //initilize weights
                InitilizeWeights();
            }

            /// <summary>
            /// Initilize weights between -0.5 and 0.5
            /// </summary>
            public void InitilizeWeights()
            {
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    for (int j = 0; j < numberOfInputs; j++)
                    {
                        weights[i, j] = (float)random.NextDouble() - 0.5f;
                    }
                }
            }

            /// <summary>
            /// Feedforward this layer with a given input
            /// </summary>
            /// <param name="inputs">The output values of the previous layer</param>
            /// <returns></returns>
            public float[] FeedForwardLayer(float[] inputs)
            {
                return FeedForwardLayernsideSpl(inputs);
            }

            /// <summary>
            /// Feedforward help function
            /// </summary>
            /// <param name="inputs"></param>
            /// <returns></returns>
            public float[] FeedForwardLayernside(float[] inputs, int index_start, int index_end)
            {
                this.inputs = inputs;// keep shallow copy which can be used for back propagation
                index_end = (index_end - index_start) + 1;

                //feed forwards
                for (int i = index_start; i < index_end; i++)
                {
                    outputs[i] = index_start;
                    //outputs[i] = 0;
                    for (int j = index_start; j < numberOfInputs; j++)
                    {
                        outputs[i] += inputs[j] * weights[i, j];
                    }

                    outputs[i] = (float)Math.Tanh(outputs[i]);
                    //outputs[i] = (float)TanH(outputs[i]);

                    //tanh fix
                    if (outputs[i] < 0)
                        outputs[i] = 0;
                }

                return outputs;
            }

            /// <summary>
            /// Feedforward help function
            /// </summary>
            /// <param name="inputs"></param>
            /// <returns></returns>
            public float[] FeedForwardLayernsideSpl(float[] inputs)
            {
                this.inputs = inputs;// keep shallow copy which can be used for back propagation

                //feed forwards
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    outputs[i] = 0;
                    for (int j = 0; j < numberOfInputs; j++)
                    {
                        outputs[i] += inputs[j] * weights[i, j];
                    }

                    outputs[i] = (float)Math.Tanh(outputs[i]);

                    // Experimental math
                    //outputs[i] = (float)TanH(outputs[i]);

                    //tanh fix
                    //if (outputs[i] < 0)
                    //outputs[i] = 0;
                }

                return outputs;
            }

            /// <summary>
            /// TanH derivate 
            /// </summary>
            /// <param name="value">An already computed TanH value</param>
            /// <returns></returns>
            public float TanHDer(float value)
            {
                return 1 - (value * value);
            }

            // Another experimental math
            ////SIGMOID AC.Fx
            //public static float TanH(double x)
            //{
            //    return (float)(1.0 / (1.0 + Math.Pow(Math.E, -x)));
            //}

            //public static float TanHDer(double x)
            //{
            //    //Math.Pow(Math.E, -x)/Math.Pow(1+ Math.Pow(Math.E, -x),2)
            //    return (float)(1 - Math.Pow(TanH(x),2));
            //}

            /// <summary>
            /// Back propagation for the output layer
            /// </summary>
            /// <param name="expected">The expected output</param>
            public void BackPropOutput(float[] expected)
            {
                //Error dervative of the cost function
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    if (outputs[i] > 0) //[ABS-FIX]
                        error[i] = outputs[i] - expected[i];
                    else
                        error[i] = outputs[i] + expected[i];

                    // Yet another experimental math
                    //error[i] = Math.Abs(outputs[i]) - expected[i];
                    //error[i] = Math.Max(0, outputs[i]) - expected[i];
                }

                //Gamma calculation
                for (int i = 0; i < numberOfOutputs; i++)
                    gamma[i] = error[i] * TanHDer(outputs[i]);

                //Caluclating detla weights
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    for (int j = 0; j < numberOfInputs; j++)
                    {
                        weightsDelta[i, j] = gamma[i] * inputs[j];
                    }
                }
            }

            /// <summary>
            /// Back propagation for the hidden layers
            /// </summary>
            /// <param name="gammaForward">the gamma value of the forward layer</param>
            /// <param name="weightsFoward">the weights of the forward layer</param>
            public void BackPropHidden(float[] gammaForward, float[,] weightsFoward)
            {
                //Caluclate new gamma using gamma sums of the forward layer
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    gamma[i] = 0;

                    for (int j = 0; j < gammaForward.Length; j++)
                    {
                        gamma[i] += gammaForward[j] * weightsFoward[j, i];
                    }

                    gamma[i] *= TanHDer(outputs[i]);
                }

                //Caluclating detla weights
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    for (int j = 0; j < numberOfInputs; j++)
                    {
                        weightsDelta[i, j] = gamma[i] * inputs[j];
                    }
                }
            }

            /// <summary>
            /// Updating weights
            /// </summary>
            public void UpdateWeights()
            {
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    for (int j = 0; j < numberOfInputs; j++)
                    {
                        //debug
                        //Console.WriteLine(weights[i, j].ToString());

                        weights[i, j] -= weightsDelta[i, j] * learningRate;
                    }
                }
            }

            //get direct access to weights
            public float[,] LayerWeights()
            {
                return weights;
            }

            //get encapsulating access to weights
            public float[,] Weights   // property
            {
                get { return weights; }   // get method
                set { weights = value; }  // set method
            }

            //get encapsulating access to the number of inputs
            public int Inputs   // property
            {
                get { return numberOfInputs; }   // get method
                set { numberOfInputs = value; }  // set method
            }

            //get encapsulating access to the number of outputs
            public int Outputs   // property
            {
                get { return numberOfOutputs; }   // get method
                set { numberOfOutputs = value; }  // set method
            }
        }

        //get encapsulating access to learning rate
        public float GetLearningRate
        {
            get { return learningRate; }
        }

        //get direct access to weights
        public float[,] GetLayerWeights(int layer)
        {
            return this.layers[layer].LayerWeights();
        }

        //set weigts using encapsulating
        public void LoadWeights(int index_layer, float[,] new_weights)
        {
            this.layers[index_layer].Weights = new_weights;
        }

        //get number of inputs using encapsulating
        public int GetNumberOfInputes(int index_layer)
        {
            return this.layers[index_layer].Inputs;
        }

        //get number of outputs using encapsulating
        public int GetNumberOfOutputs(int index_layer)
        {
            return this.layers[index_layer].Outputs;
        }

    }

}









