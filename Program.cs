using FaNNtasyRaceResultPredictions;
using System.Globalization;

//store loss function
double lossfunction;

//number of eras
int eras = 220;

//training set
int ts_size = 80;
int ts_count = 52; //! 11-12||17-18||23-24||49-50,52
int ts_check = 4; ts_count -= ts_check;
float[] trainingSet;
float[] trainingSetSliced, expectedSliced;

//save training results
bool saveTraining = true;

//load saved nn
bool loadNet = true;

//mutate preloaded nn
bool mutateNet = true;

//store training set data
trainingSet = new float[ts_count * ts_size + ts_check * ts_size];
//sliced training set array
trainingSetSliced = new float[ts_size - ts_size / 4 + 1];
//sliced results for training
expectedSliced = new float[ts_size - ts_size / 4 + 1];

//nn structure
int[] structure = { 60, 125, 125, 20 };

//intialize network [175 max]
NeuralNetwork net = new NeuralNetwork(structure);


//RESULTS AREA
//calculate results
//multidemensional array [driver,points]
int[,] drivers = { { 1, 25, 0 }, { 2, 0, 0 }, { 4, 18, 0 }, { 10, 1, 0 }, { 11, 0, 0 }, { 14, 4, 0 }, { 16, 12, 0 }, { 18, 0, 0 }, { 20, 0, 0 }, { 22, 0, 0 }, { 23, 0, 0 }, { 24, 0, 0 }, { 27, 0, 0 }, { 31, 2, 0 }, { 40, 0, 0 }, { 44, 10, 0 }, { 55, 8, 0 }, { 63, 6, 0 }, { 77, 0, 0 }, { 81, 15, 0 } };
//store output info
float[,] output_temp = new float[drivers.Length / 2, drivers.Length / 2];
//number of top drivers
int podium = 5;
//total points for top drivers
float score = 0;
//store info about corrupted data
int corrupted_lines = 0;
//store info about duplicates
int duplicated_lines = 0;

//store run's info
float topResult = 0;
float average_score = 0;
float average_duplicates = 0;

//number of nn's
int main_loop = 5000;


void TrainingFanntasy()
{
    //reset loss function
    lossfunction = 0;

    //reset learning rate
    net.ResetLearningRate();

    for (int j = 0; j < ts_count * ts_size; j += ts_size)
    {
        //slice the training set array [0-60]
        Array.Copy(trainingSet, j, trainingSetSliced, 0, 60);

        //slice the expected array [60-80]
        Array.Copy(trainingSet, j + 60, expectedSliced, 0, 20);

        float result = net.FeedForward(trainingSetSliced)[0];
        net.BackProp(expectedSliced);

        lossfunction += Math.Pow(Math.Abs(expectedSliced[0] - result), 2);
    }

    lossfunction = lossfunction / 2;
}

//find driver index
static int FindDriverIndex(float driverNumber, int[,] drivers)
{
    for (int i = 0; i < drivers.GetLength(0); i++)
    {
        if (drivers[i, 0] == driverNumber)
        {
            return i;
        }
    }
    return 0; // Driver not found
}

//save and display results
void SaveResults(int number, int ts_check_index)
{
    //final set for feedforward 
    Array.Copy(trainingSet, ts_count * ts_size + ts_check_index * ts_size, trainingSetSliced, 0, 60); //! ts_count * ts_size || 880

    //slice the expected (real results) array
    Array.Copy(trainingSet, ts_count * ts_size + ts_check_index * ts_size + 60, expectedSliced, 0, 20);

    int[,] points = { { 1, 25 }, { 2, 18 }, { 3, 15 }, { 4, 12 }, { 5, 10 }, { 6, 8 }, { 7, 6 }, { 8, 4 }, { 9, 2 }, { 10, 1 } };

    //overwrite points for each driver
    for (int i = 0; i < expectedSliced.GetLength(0); i++)
    {
        if (i < 10)
            drivers[FindDriverIndex(expectedSliced[i], drivers), 1] = points[i, 1];
        else
            drivers[FindDriverIndex(expectedSliced[i], drivers), 1] = 0;
    }

    //save results
    using (StreamWriter writetext = new StreamWriter("nn_training_results" + "/nn" + number + "/nn_output.txt"))
    {
        float result;

        for (int i = 0; i < 20; i++)
        {
            result = net.FeedForward(trainingSetSliced)[i];

            //save results (output) of training
            writetext.WriteLine(result.ToString());
        }
    }
}

//read training set
void ReadTrainingSet(bool is_active)
{
    if (is_active)
    {
        //read training set text
        string text = File.ReadAllText("trainingset.txt");
        string[] lines = text.Split(Environment.NewLine);

        //auxiliary variables
        float read_temp = 0;
        int index = 0;

        foreach (string line in lines)
        {
            //convert string to int
            if (!String.IsNullOrEmpty(line))
                read_temp = Convert.ToSingle(int.Parse(line));
            else
                break;

            trainingSet[index] = read_temp / 100;
            index++;
        }
    }
}

//save training results
void SaveNeuralNetwork(int number, bool is_active)
{
    if (is_active)
    {
        using (StreamWriter writetext = new StreamWriter("nn_training_results" + "/nn" + number + "/learning_rate.txt"))
        {
            //save current learning rate
            writetext.WriteLine(net.GetLearningRate.ToString());
        }

        for (int i = 1; i < structure.Length; i++)
        {
            using (StreamWriter writetext = new StreamWriter("nn_training_results" + "/nn" + number + "/weights_layer" + i + ".txt"))
            {
                float[,] weights = net.GetLayerWeights(i - 1);

                //save nbr of inputs and outputs
                writetext.WriteLine(net.GetNumberOfInputes(i - 1).ToString());
                writetext.WriteLine(net.GetNumberOfOutputs(i - 1).ToString());

                for (int j = 0; j < weights.GetLength(0); j++)
                {
                    for (int k = 0; k < weights.GetLength(1); k++)
                    {
                        //save current learning rate
                        writetext.WriteLine(weights[j, k].ToString());
                    }
                }
            }
        }

    }
}

//load saved neural network
void LoadNeuralNetwork(bool is_active)
{
    if (is_active)
    {
        for (int i = 1; i < structure.Length; i++)
        {
            //read training set text
            string text = File.ReadAllText("nn/weights_layer" + i + ".txt");
            string[] lines = text.Split(Environment.NewLine);

            int numberOfInputs = int.Parse(lines[0]);
            int numberOfOutputs = int.Parse(lines[1]);

            float[,] weights; float read_temp;
            weights = new float[numberOfOutputs, numberOfInputs];
            int index = 2;

            for (int j = 0; j < numberOfOutputs; j++)
            {
                for (int k = 0; k < numberOfInputs; k++)
                {
                    //convert string to int
                    if (!String.IsNullOrEmpty(lines[index]))
                        read_temp = float.Parse(lines[index]);
                    else
                        break;

                    weights[j, k] = read_temp; index++;
                }
            }

            //load weights
            net.LoadWeights(i - 1, weights);
        }
    }
}

//mutate neural network
void MutateNeuralNetwork(bool is_active)
{
    if (is_active)
    {
        //generate random number
        Random random = new Random();
        Random random_small = new Random();

        for (int i = 1; i < structure.Length; i++)
        {
            //weights;
            float[,] weights = new float[structure[i], structure[i - 1]];

            for (int j = 0; j < structure[i]; j++)
            {
                for (int k = 0; k < structure[i - 1]; k++)
                {
                    int random_small_int = random_small.Next(0, 10);

                    if (random_small_int == 1 || random_small_int == 2)
                        weights[j, k] = (float)random.NextDouble() - 0.5f;
                }
            }

            //load weights
            net.LoadWeights(i - 1, weights);
        }
    }
}


//RESULTS AREA
///Dynamic duplicates count and assignation
void DynamicAssignNCount(ref float driver, ref float value, int index)
{
    driver = drivers[index, 0];
    value = drivers[index, 1];

    if (drivers[index, 2] != 0)
    {
        //count duplicates
        duplicated_lines++;
        //don't bring points
        value = 0;
    }

    Console.Write(driver + "\n");
    drivers[index, 2]++;
}


//main-inner void
void MainInnerVoid(int number)
{
    string directoryPath = "nn_training_results/nn" + number;
    Directory.CreateDirectory(directoryPath);

    string filePath = Path.Combine(directoryPath, "lossfunction.txt");

    using (StreamWriter writetext = new StreamWriter(filePath))
    {
        writetext.WriteLine(DateTime.Now.ToString());

        //initial NN training
        TrainingFanntasy();

        writetext.WriteLine(lossfunction.ToString());

        //manage learning rate
        int reductionRateInt = (int)Math.Round(eras / 5.0);
        float learningRate = net.GetLearningRate;
        float lrReductionRateFloat = (eras < 10000) ? (learningRate / 7) : (learningRate / 7) * (10000 / eras);
        int saveRateInt = (int)Math.Round(eras / 100.0);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        int temp = 0; //lossfunction saving rate

        for (int i = 0; i < eras; i += 4)
        {
            TrainingFanntasy();

            if (i == reductionRateInt)
            {
                net.LowerLearningRate(lrReductionRateFloat);
                reductionRateInt += reductionRateInt;
                Console.Write("=");
            }

            if (temp == saveRateInt)
            {
                writetext.WriteLine(lossfunction.ToString());
                saveRateInt += saveRateInt;
            }
            temp++;
        }

        watch.Stop(); var elapsedMs = watch.ElapsedMilliseconds;
        Console.WriteLine("runtime: " + elapsedMs + " eras: " + eras);

        SaveNeuralNetwork(number, saveTraining);
    }

    Console.Write("NeuralNetwork." + number + " stopped learning \n");

    //iterate over every test set
    for (int check_set_index = 0; check_set_index < ts_check; check_set_index++)
    {
        SaveResults(number, check_set_index);

        //load training results [nn_output.txt]
        string contenst = File.ReadAllText("nn_training_results" + "/nn" + number + "/nn_output.txt");

        int l = 0; //find the nearest neighbour of output
        using (StringReader reader = new StringReader(contenst))
        {
            string line = string.Empty;
            do
            {
                line = reader.ReadLine();

                if (line != null)
                {
                    Thread.CurrentThread.CurrentCulture = new CultureInfo("de-DE");
                    float temp_driver = (float)Math.Round((decimal)float.Parse(line, Thread.CurrentThread.CurrentCulture.NumberFormat), 5) * 100;
                    float temp_value = 0;

                    if (temp_driver < 0) //mirroring
                        temp_driver = -temp_driver;

                    if (temp_driver == 0)
                    {
                        temp_driver = 0;//drivers[rand, 0]; 
                        temp_value = 0;//drivers[rand, 1];
                        corrupted_lines++; // corrupted lines
                    }
                    else
                    {
                        for (int j = 0; j < drivers.Length / 3 - 1; j++)
                        {
                            if (temp_driver > drivers[j, 0] && temp_driver < drivers[j + 1, 0])
                            {
                                float bottom = temp_driver - drivers[j, 0];
                                float top = drivers[j + 1, 0] - temp_driver;

                                if (bottom >= top)
                                    DynamicAssignNCount(ref temp_driver, ref temp_value, j + 1);
                                else
                                    DynamicAssignNCount(ref temp_driver, ref temp_value, j);
                            }
                            else if (temp_driver > drivers[drivers.Length / 3 - 1, 0])
                                DynamicAssignNCount(ref temp_driver, ref temp_value, drivers.Length / 3 - 1);
                            else if (temp_driver > 0 && temp_driver < drivers[0, 0])
                                DynamicAssignNCount(ref temp_driver, ref temp_value, 0);
                        }
                    }

                    output_temp[l, 0] = temp_driver; output_temp[l, 1] = temp_value; l++;
                }
            } while (line != null);
        }

        //sum total points for 'podium drivers
        for (int j = 0; j < podium; j++)
        {
            score += output_temp[j, 1];
        }

        //score selection fix [SEL-FIX]
        score -= duplicated_lines;

        //reset duplicats count
        for (int h = 0; h < drivers.Length / 3; h++)
            drivers[h, 2] = 0;
    }

    using (StreamWriter writetext = new StreamWriter("nn_training_results" + "/nn" + number + "/results.txt"))
    {
        //save results to file
        writetext.WriteLine(DateTime.Now.ToString());
        writetext.WriteLine("results for top " + podium + " drivers: " + score.ToString());
        writetext.WriteLine("number of corrupted lines: " + corrupted_lines.ToString());
        writetext.WriteLine("number of duplicated lines: " + duplicated_lines.ToString());
    }

    Console.Write("Score: " + score.ToString() + " Duplicated lines: " + duplicated_lines.ToString() + "\n\n");

    //store average info
    using (StreamWriter writetext = new StreamWriter("average.txt"))
    {
        //calculate average score
        average_score += score;
        float temp_score = average_score / number;
        average_duplicates += duplicated_lines;
        float temp_duplicates = average_duplicates / number;

        writetext.WriteLine(DateTime.Now.ToString());
        writetext.WriteLine("average score: " + temp_score);
        writetext.WriteLine("average duplicates: " + temp_duplicates);
    }

    //store info about the best net
    if (score > topResult)
    {
        //overwrite current top result
        topResult = score;

        using (StreamWriter writetext = new StreamWriter("top_result.txt"))
        {
            writetext.WriteLine(DateTime.Now.ToString());
            writetext.WriteLine("nn: " + number);
            writetext.WriteLine("results for top " + podium + " drivers: " + score.ToString());
            writetext.WriteLine("number of corrupted lines: " + corrupted_lines.ToString());
            writetext.WriteLine("number of duplicated lines: " + duplicated_lines.ToString());
        }
    }

    //reset values
    score = 0;
    corrupted_lines = 0;
    duplicated_lines = 0;

    //reset duplicats count
    for (int h = 0; h < drivers.Length / 3; h++)
        drivers[h, 2] = 0;

    //reset (reinitialise) the net
    net = new NeuralNetwork(structure);
}


///MAIN LOOP
for (int number = 1; number < main_loop; number++)
{
    //read ts from file
    ReadTrainingSet(true);

    //load saved nn
    LoadNeuralNetwork(loadNet);

    //mutate preloaded nn
    MutateNeuralNetwork(mutateNet);

    //train nn's and save results
    MainInnerVoid(number);
}
///end of main loop

Console.Write("Press any key to exit... \n");
Console.ReadKey(); Console.ReadKey();



